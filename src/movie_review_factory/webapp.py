"""Local web UI + JSON API for the movie-review-factory pipeline.

Python-native and dependency-free: it uses only the standard library
(``http.server`` + ``threading``) so it adds nothing to pyproject and needs no
JavaScript build step. The page is a single inline HTML document that talks to
a small JSON API. Bind to localhost - this is a single-operator tool, not a
public service.

Publishing safety is built into the shape of the API, not just the UI:

* The web "run" action stops the pipeline at the ``thumbnail`` stage
  (``until="thumbnail"``) and never reaches ``publish`` on its own.
* Metadata approval (``approved=true``) is a separate, explicit endpoint.
* Publishing requires an explicit ``confirm`` and, even then, only runs the
  ``publish`` stage - which merely writes the handoff record
  ``publish_record.json`` when approvals are complete. Nothing is ever uploaded.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import localization, pipeline
from .models import JobConfig

# A job id / artifact name must be a single safe path segment. This is the only
# thing standing between a URL and the filesystem, so it is deliberately strict.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")

# The web "run" button intentionally stops here; publish is a separate action.
RUN_UNTIL_STAGE = "thumbnail"

# Optional bearer-token auth.  Set DASHBOARD_TOKEN in the environment to require
# a token on every request.  When the variable is absent or empty every request
# is allowed — explicit local-dev mode, no silent open access on a deployed host.
_DASHBOARD_TOKEN: str = os.environ.get("DASHBOARD_TOKEN", "")

# Maximum JSON request-body size accepted before returning 413.
_MAX_BODY_BYTES = 1_048_576  # 1 MiB

# Conservative security headers added to every response.
_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # The page uses only inline scripts/styles and self-hosted media.
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "media-src 'self' blob:; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'"
    ),
}

_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
    ".json": "application/json; charset=utf-8",
    ".srt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

_ARTIFACT_KINDS = {
    ".mp4": "video",
    ".mp3": "audio",
    ".json": "json",
    ".srt": "subtitle",
    ".md": "markdown",
    ".txt": "text",
    ".jpg": "image",
    ".jpeg": "image",
}


def _is_safe_segment(name: str) -> bool:
    return bool(_SAFE_SEGMENT.match(name)) and name not in (".", "..")


def _content_type(name: str) -> str:
    return _CONTENT_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")


def _artifact_kind(name: str) -> str:
    return _ARTIFACT_KINDS.get(Path(name).suffix.lower(), "other")


class JobsService:
    """Filesystem-backed operations behind the API, decoupled from HTTP.

    Wraps a ``jobs_root`` directory and tracks in-memory run state per job so
    the UI can poll progress. All heavy pipeline work is delegated to
    ``pipeline``; this class only adds job discovery, localization, background
    running, and the explicit approval/publish gates.
    """

    def __init__(self, jobs_root: Path):
        self.jobs_root = Path(jobs_root)
        self._runs: dict[str, dict] = {}
        self._lock = threading.Lock()

    # -- helpers -------------------------------------------------------------

    def _job_root(self, job_id: str) -> Path:
        if not _is_safe_segment(job_id):
            raise ValueError(f"job_id không hợp lệ: {job_id!r}")
        return self.jobs_root / job_id

    def _require_job(self, job_id: str) -> Path:
        root = self._job_root(job_id)
        if not pipeline.manifest_path(root).exists():
            raise FileNotFoundError(f"không tìm thấy job: {job_id}")
        return root

    @staticmethod
    def _read_json(root: Path, name: str) -> dict:
        path = root / name
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}

    def _is_running(self, job_id: str) -> bool:
        with self._lock:
            return bool(self._runs.get(job_id, {}).get("running"))

    # -- read ----------------------------------------------------------------

    def list_jobs(self) -> list[dict]:
        jobs = pipeline.list_jobs(self.jobs_root)
        for job in jobs:
            job["running"] = self._is_running(job["job_id"])
        return jobs

    def status(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        info = pipeline.job_status(root)
        for stage in info["stages"]:
            stage["stage_label"] = localization.stage_label(stage["stage"])
            stage["status_label"] = localization.status_label(stage["status"])
            stage["status_hint"] = localization.status_hint(stage["status"])
            stage["message_vi"] = localization.localize_message(stage.get("message", ""))

        with self._lock:
            run_state = dict(self._runs.get(job_id, {}))
        run_error = run_state.get("error")
        info["running"] = bool(run_state.get("running"))
        info["run_error"] = run_error
        info["run_error_vi"] = localization.localize_message(run_error) if run_error else None

        script = self._read_json(root, "script.json")
        meta = self._read_json(root, pipeline.METADATA_NAME)
        info["approvals"] = {
            "script_present": bool(script),
            "script_approved": bool(script.get("approved")),
            "metadata_present": bool(meta),
            "metadata_approved": bool(meta.get("approved")),
        }
        info["artifacts"] = self.list_artifacts(job_id)
        info["has_final_video"] = (root / "final.mp4").exists()
        info["has_thumbnail"] = (root / "thumbnail.jpg").exists()
        return info

    def list_artifacts(self, job_id: str) -> list[dict]:
        root = self._job_root(job_id)
        if not root.exists():
            return []
        items: list[dict] = []
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.name == pipeline.MANIFEST_NAME or path.suffix == ".tmp":
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            items.append({
                "name": path.name,
                "size": size,
                "kind": _artifact_kind(path.name),
                "href": f"/api/jobs/{job_id}/artifacts/{path.name}",
            })
        return items

    def get_metadata(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        meta = self._read_json(root, pipeline.METADATA_NAME)
        return {"present": bool(meta), "metadata": meta}

    def get_thumbnails(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        thumbnails = self._read_json(root, "thumbnails.json")
        return {"present": bool(thumbnails), "thumbnails": thumbnails}

    def artifact_path(self, job_id: str, name: str) -> Path:
        root = self._require_job(job_id)
        if not _is_safe_segment(name):
            raise ValueError(f"tên artifact không hợp lệ: {name!r}")
        path = root / name
        # Defence in depth: the resolved file must stay inside the job dir.
        if path.resolve().parent != root.resolve():
            raise ValueError("đường dẫn artifact không hợp lệ")
        if not path.is_file():
            raise FileNotFoundError(f"không tìm thấy artifact: {name}")
        return path

    # -- write ---------------------------------------------------------------

    def create_job(self, payload: dict) -> dict:
        job_id = str(payload.get("job_id", "")).strip()
        if not _is_safe_segment(job_id):
            raise ValueError("job_id chỉ gồm chữ, số, dấu chấm, gạch ngang, gạch dưới")
        root = self.jobs_root / job_id
        if pipeline.manifest_path(root).exists():
            raise FileExistsError(f"job đã tồn tại: {job_id}")
        source_video = payload.get("source_video") or None
        content_agent = str(payload.get("content_agent") or "scaffold")
        if content_agent not in {"scaffold", "claude"}:
            raise ValueError("content_agent phải là scaffold hoặc claude")
        config = JobConfig(
            job_id=job_id,
            language=str(payload.get("language") or "vi"),
            target_minutes=float(payload.get("target_minutes") or 10),
            aspect_ratio=str(payload.get("aspect_ratio") or "16:9"),
            source_video=Path(source_video) if source_video else None,
            movie_title=str(payload.get("movie_title") or "").strip() or None,
            content_agent=content_agent,
        )
        pipeline.create_job(root, config)
        return self.status(job_id)

    def start_run(self, job_id: str, *, until: str | None = RUN_UNTIL_STAGE) -> dict:
        """Run the pipeline in a background thread up to ``until`` (default
        ``thumbnail``). Progress is observed by polling ``status`` because
        ``run_job`` persists the manifest after every stage."""
        root = self._require_job(job_id)
        with self._lock:
            if self._runs.get(job_id, {}).get("running"):
                raise RuntimeError("job đang chạy")
            self._runs[job_id] = {"running": True, "error": None, "until": until}

        def _worker() -> None:
            error: str | None = None
            try:
                pipeline.run_job(root, until=until)
            except Exception as exc:  # unexpected; run_job already handles stage failures
                error = str(exc)
            with self._lock:
                self._runs[job_id] = {"running": False, "error": error, "until": until}

        threading.Thread(target=_worker, name=f"mrf-run-{job_id}", daemon=True).start()
        return {"started": True, "until": until}

    def approve_metadata(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        meta = pipeline.approve_metadata(root)
        return {"approved": True, "metadata": meta}

    def update_metadata(self, job_id: str, fields: dict) -> dict:
        root = self._require_job(job_id)
        meta = pipeline.update_metadata(root, fields)
        return {"approved": bool(meta.get("approved")), "metadata": meta}

    def get_script(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        script = self._read_json(root, "script.json")
        return {"present": bool(script), "script": script}

    def update_script(self, job_id: str, fields: dict) -> dict:
        root = self._require_job(job_id)
        script = pipeline.update_script(root, fields)
        return {"approved": False, "script": script}

    def select_thumbnail(self, job_id: str, candidate: str) -> dict:
        root = self._require_job(job_id)
        thumbnails = pipeline.select_thumbnail(root, candidate)
        return {
            "selected": thumbnails.get("primary_candidate", ""),
            "thumbnails": thumbnails,
        }

    def approve_script(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        script = pipeline.approve_script(root)
        return {"approved": True, "script": script}

    def prepare_publish(self, job_id: str, *, confirm: bool) -> dict:
        """The publishing gate. Without ``confirm`` it does nothing and reports
        the current gate state. With ``confirm`` it runs only the ``publish``
        stage, which writes ``publish_record.json`` when approvals are complete
        and otherwise stays ``skipped``. It never uploads anything."""
        root = self._require_job(job_id)
        meta = self._read_json(root, pipeline.METADATA_NAME)
        if not confirm:
            return {
                "confirmed": False,
                "published": False,
                "metadata_approved": bool(meta.get("approved")),
                "message_vi": "Cần xác nhận rõ ràng để tạo bản ghi xuất bản (không tải lên).",
            }
        manifest = pipeline.run_job(root, until="publish")
        stage = manifest.stage("publish")
        status = stage.status if stage else "pending"
        message = stage.message if stage else ""
        return {
            "confirmed": True,
            "published": status == "ready",
            "status": status,
            "status_label": localization.status_label(status),
            "message": message,
            "message_vi": localization.localize_message(message),
        }


# --- HTTP layer --------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: JobsService):
        self.service = service
        super().__init__(address, MRFRequestHandler)


class MRFRequestHandler(BaseHTTPRequestHandler):
    server_version = "MovieReviewFactory/0.2"

    @property
    def service(self) -> JobsService:
        return self.server.service  # type: ignore[attr-defined]

    # -- low-level responders ------------------------------------------------

    def _send_json(self, status: int, obj: object) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for hk, hv in _SECURITY_HEADERS.items():
            self.send_header(hk, hv)
        if status in (401, 403):
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for hk, hv in _SECURITY_HEADERS.items():
            self.send_header(hk, hv)
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > _MAX_BODY_BYTES:
            raise OverflowError(f"request body too large ({length} > {_MAX_BODY_BYTES})")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise ValueError(f"JSON không hợp lệ: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("body phải là một đối tượng JSON")
        return data

    @staticmethod
    def _error_payload(exc: Exception) -> dict:
        message = str(exc)
        return {"error": message, "error_vi": localization.localize_message(message)}

    def _dispatch(self, handler) -> None:
        """Run a route handler, mapping domain exceptions to HTTP status codes."""
        try:
            handler()
        except ValueError as exc:
            self._send_json(400, self._error_payload(exc))
        except FileNotFoundError as exc:
            self._send_json(404, self._error_payload(exc))
        except FileExistsError as exc:
            self._send_json(409, self._error_payload(exc))
        except RuntimeError as exc:
            self._send_json(409, self._error_payload(exc))
        except OverflowError:
            self._send_json(413, {"error": "request body too large", "error_vi": "body yêu cầu quá lớn"})
        except Exception:  # last resort — never leak a stack trace or exception text
            self._send_json(500, {"error": "internal server error", "error_vi": "lỗi máy chủ nội bộ"})

    # -- file streaming with HTTP range support (for video/audio preview) ----

    def _serve_file(self, path: Path) -> None:
        file_size = path.stat().st_size
        start, end, status = 0, file_size - 1, 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw = range_header.split("=", 1)[1].split(",")[0].strip()
            begin, _, finish = raw.partition("-")
            try:
                if begin:
                    start = int(begin)
                    end = int(finish) if finish else file_size - 1
                else:
                    start = max(0, file_size - int(finish))
                    end = file_size - 1
            except ValueError:
                start, end, status = 0, file_size - 1, 200
            else:
                end = min(end, file_size - 1)
                if start > end or start >= file_size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    for hk, hv in _SECURITY_HEADERS.items():
                        self.send_header(hk, hv)
                    self.end_headers()
                    return
                status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", _content_type(path.name))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        for hk, hv in _SECURITY_HEADERS.items():
            self.send_header(hk, hv)
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # -- auth ----------------------------------------------------------------

    def _check_auth(self) -> bool:
        """True if the request is authorized.

        When DASHBOARD_TOKEN is empty every request is allowed (local-dev
        mode).  When it is set the Authorization header must carry a matching
        Bearer token; comparison is constant-time via hmac.compare_digest so
        the token value is never logged or compared with ==.
        """
        if not _DASHBOARD_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False
        token = auth[len("Bearer "):]
        return hmac.compare_digest(
            token.encode("utf-8"),
            _DASHBOARD_TOKEN.encode("utf-8"),
        )

    def _send_401(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Bearer realm="dashboard"')
        self.send_header("Content-Length", "0")
        for hk, hv in _SECURITY_HEADERS.items():
            self.send_header(hk, hv)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    # -- routing -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if not self._check_auth():
            self._send_401()
            return
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
            return
        if route == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        parts = [unquote(p) for p in route.split("/") if p]
        if not parts or parts[0] != "api":
            self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})
            return
        self._dispatch(lambda: self._route_get(parts[1:]))

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if not self._check_auth():
            self._send_401()
            return
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if not parts or parts[0] != "api":
            self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})
            return
        self._dispatch(lambda: self._route_post(parts[1:]))

    def _route_get(self, parts: list[str]) -> None:
        # parts: [] | ["jobs"] | ["jobs", id] | ["jobs", id, "artifacts"] |
        #        ["jobs", id, "artifacts", name] | ["jobs", id, "metadata"]
        if parts == ["jobs"]:
            self._send_json(200, {"jobs": self.service.list_jobs()})
            return
        if len(parts) == 2 and parts[0] == "jobs":
            self._send_json(200, self.service.status(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "artifacts":
            self._send_json(200, {"artifacts": self.service.list_artifacts(parts[1])})
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "artifacts":
            self._serve_file(self.service.artifact_path(parts[1], parts[3]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "metadata":
            self._send_json(200, self.service.get_metadata(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "script":
            self._send_json(200, self.service.get_script(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "thumbnails":
            self._send_json(200, self.service.get_thumbnails(parts[1]))
            return
        self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})

    def _route_post(self, parts: list[str]) -> None:
        if parts == ["jobs"]:
            self._send_json(201, self.service.create_job(self._read_body()))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "run":
            self._send_json(202, self.service.start_run(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "metadata":
            self._send_json(200, self.service.update_metadata(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "metadata" and parts[3] == "approve":
            self._send_json(200, self.service.approve_metadata(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "script":
            self._send_json(200, self.service.update_script(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "script" and parts[3] == "approve":
            self._send_json(200, self.service.approve_script(parts[1]))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "thumbnails" and parts[3] == "select":
            body = self._read_body()
            candidate = str(body.get("candidate") or "").strip()
            if not candidate:
                raise ValueError("thiếu candidate ảnh bìa")
            self._send_json(200, self.service.select_thumbnail(parts[1], candidate))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "publish":
            body = self._read_body()
            result = self.service.prepare_publish(parts[1], confirm=bool(body.get("confirm")))
            self._send_json(200, result)
            return
        self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Keep the console (and test output) quiet; override if debugging.
        return


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  jobs_root: Path | str = "jobs") -> _Server:
    """Build (but do not start) the local web server."""
    return _Server((host, port), JobsService(Path(jobs_root)))


def run_server(host: str = "127.0.0.1", port: int = 8765,
               jobs_root: Path | str = "jobs") -> None:
    """Start the local web server and serve until interrupted."""
    server = create_server(host, port, jobs_root)
    bound_host, bound_port = server.server_address[:2]
    print(f"Movie Review Factory UI: http://{bound_host}:{bound_port}  (jobs: {Path(jobs_root)})")
    print("Nhấn Ctrl+C để dừng.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# --- single-page UI ----------------------------------------------------------
# Vanilla HTML + JS (no framework, no build step). Kept as one inline document
# so the whole UI ships with the package and needs no static-file plumbing.

INDEX_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Xưởng Review Phim — Bảng điều khiển</title>
<style>
  :root { color-scheme: light dark; --gap: 16px; --accent: #4f7cff; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, "Segoe UI", Roboto, sans-serif;
         line-height: 1.5; background: #0f1115; color: #e6e8ee; }
  header { padding: 14px 20px; background: #171a21; border-bottom: 1px solid #262b36; }
  header h1 { margin: 0; font-size: 18px; }
  header .sub { color: #9aa3b2; font-size: 13px; }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: var(--gap); padding: var(--gap); }
  @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 10px; padding: 14px; margin-bottom: var(--gap); }
  .card h2 { margin: 0 0 10px; font-size: 15px; }
  label { display: block; font-size: 12px; color: #9aa3b2; margin: 8px 0 2px; }
  input, select, textarea, button { font: inherit; }
  input, select, textarea { width: 100%; padding: 7px 9px; background: #0f1115; color: #e6e8ee;
      border: 1px solid #333a48; border-radius: 6px; }
  textarea { resize: vertical; }
  button { cursor: pointer; padding: 8px 12px; border-radius: 6px; border: 1px solid #333a48;
      background: #232838; color: #e6e8ee; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .jobrow { padding: 8px 10px; border: 1px solid #262b36; border-radius: 8px; margin-bottom: 6px; cursor: pointer; }
  .jobrow:hover { border-color: var(--accent); }
  .jobrow.active { border-color: var(--accent); background: #1c2438; }
  .muted { color: #9aa3b2; font-size: 12px; }
  .stage { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid #20242e; }
  .stage:last-child { border-bottom: 0; }
  .stage .name { width: 150px; font-weight: 600; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
  .badge.pending { background: #2b303c; color: #b9c0cc; }
  .badge.running { background: #3a2f13; color: #f5c451; }
  .badge.ready   { background: #133a22; color: #57d98a; }
  .badge.failed  { background: #43181c; color: #ff7a86; }
  .badge.skipped { background: #2b303c; color: #9aa3b2; }
  .progress { height: 8px; background: #0f1115; border: 1px solid #333a48; border-radius: 999px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--accent); width: 0; transition: width .3s; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .arts a { color: #9db4ff; text-decoration: none; }
  .arts li { margin: 3px 0; }
  video { width: 100%; border-radius: 8px; background: #000; margin-top: 8px; }
  .thumb-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }
  .thumb-item { border: 1px solid #333a48; border-radius: 8px; padding: 8px; }
  .thumb-item.selected { border-color: var(--accent); background: #1c2438; }
  .thumb-item img { width: 100%; aspect-ratio: 16/9; object-fit: cover; border-radius: 6px; background: #000; }
  .thumb-item button { width: 100%; margin-top: 6px; }
  .gate { border: 1px dashed #6b5324; background: #1c1706; border-radius: 8px; padding: 10px; }
  .ok { color: #57d98a; } .warn { color: #f5c451; } .err { color: #ff7a86; }
  .notice { font-size: 12px; color: #9aa3b2; margin-top: 6px; }
  code { background: #0f1115; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<header>
  <h1>Xưởng Review Phim — Bảng điều khiển</h1>
  <div class="sub">Chạy cục bộ · Không tự động xuất bản · Duyệt thủ công trước khi bàn giao</div>
</header>
<div class="layout">
  <aside>
    <div class="card">
      <h2>Tạo job mới</h2>
      <form id="createForm">
        <label>Mã job (job_id)</label>
        <input name="job_id" placeholder="vd: review-abc" required>
        <label>Tên phim / truy vấn nghiên cứu</label>
        <input name="movie_title" placeholder="vd: The Matrix (1999)">
        <label>Ngôn ngữ</label>
        <input name="language" value="vi">
        <label>Bộ tạo nội dung</label>
        <select name="content_agent">
          <option value="scaffold">Scaffold (offline)</option>
          <option value="claude">Claude Code (research → outline → script)</option>
        </select>
        <label>Thời lượng mục tiêu (phút)</label>
        <input name="target_minutes" type="number" value="10" min="1" max="60" step="0.5">
        <label>Tỷ lệ khung hình</label>
        <select name="aspect_ratio"><option>16:9</option><option>9:16</option></select>
        <label>Video nguồn (tuỳ chọn, đường dẫn cục bộ)</label>
        <input name="source_video" placeholder="data\\raw\\....mp4">
        <div class="row" style="margin-top:10px">
          <button class="primary" type="submit">Tạo job</button>
        </div>
        <div id="createMsg" class="notice"></div>
      </form>
    </div>
    <div class="card">
      <h2>Danh sách job</h2>
      <div id="jobList" class="muted">Đang tải…</div>
    </div>
  </aside>
  <main>
    <div id="empty" class="card muted">Chọn một job ở cột trái để xem tiến trình, artifact và cổng xuất bản.</div>
    <div id="detail" style="display:none">
      <div class="card">
        <div class="row" style="justify-content:space-between">
          <h2 id="jobTitle" style="margin:0"></h2>
          <div class="row">
            <button id="runBtn" class="primary">Chạy pipeline (đến bước Ảnh bìa)</button>
            <button id="refreshBtn">Làm mới</button>
          </div>
        </div>
        <div class="notice">Nút "Chạy" dừng ở bước <code>thumbnail</code>; bước <code>publish</code> không bao giờ tự chạy.</div>
        <div style="margin-top:10px" class="progress"><div id="progBar"></div></div>
        <div id="progText" class="muted" style="margin-top:6px"></div>
        <div id="runErr" class="err" style="margin-top:6px"></div>
      </div>

      <div class="card">
        <h2>Tiến trình các bước</h2>
        <div id="stages"></div>
      </div>

      <div class="card" id="videoCard" style="display:none">
        <h2>Xem trước bản dựng cuối (final.mp4)</h2>
        <video id="video" controls preload="metadata"></video>
      </div>

      <div class="card" id="thumbnailCard" style="display:none">
        <h2>Chọn ảnh bìa</h2>
        <div class="muted">Chọn một trong các frame đã tạo. Ảnh được chọn sẽ trở thành <code>thumbnail.jpg</code>.</div>
        <div id="thumbnailGrid" class="thumb-grid" style="margin-top:10px"></div>
        <div id="thumbnailMsg" class="notice"></div>
      </div>

      <div class="card">
        <h2>Artifact</h2>
        <ul id="artifacts" class="arts muted"></ul>
      </div>

      <div class="card">
        <h2>Kịch bản &amp; Duyệt</h2>
        <div id="scriptState" class="muted"></div>
        <label>Nội dung các phần (JSON sections)</label>
        <textarea id="scriptSections" rows="8"></textarea>
        <div class="row" style="margin-top:10px">
          <button id="saveScriptBtn" disabled>Lưu kịch bản</button>
          <button id="approveScriptBtn" class="primary" disabled>Duyệt kịch bản</button>
        </div>
        <div class="notice">Lưu thay đổi sẽ đặt lại trạng thái duyệt (approved=false) — phải duyệt lại sau khi sửa.</div>
        <div id="scriptMsg" class="notice"></div>
      </div>

      <div class="card">
        <h2>Siêu dữ liệu &amp; Duyệt</h2>
        <div id="metaState" class="muted"></div>
        <label>Tiêu đề</label>
        <input id="metaTitle">
        <label>Mô tả</label>
        <textarea id="metaDesc" rows="5"></textarea>
        <label>Thẻ (phân tách bằng dấu phẩy)</label>
        <input id="metaTags">
        <div class="row" style="margin-top:10px">
          <button id="saveMetaBtn">Lưu thay đổi</button>
          <button id="approveBtn" class="primary">Duyệt siêu dữ liệu</button>
        </div>
        <div class="notice">Lưu thay đổi sẽ đặt lại trạng thái duyệt (approved=false) — phải duyệt lại sau khi sửa.</div>
        <div id="metaMsg" class="notice"></div>
      </div>

      <div class="card">
        <h2>Cổng xuất bản</h2>
        <div class="gate">
          <div>Thao tác này <strong>không tải lên</strong> bất kỳ đâu. Nó chỉ ghi tệp bàn giao
            <code>publish_record.json</code> khi kịch bản và siêu dữ liệu đã được duyệt.</div>
          <label class="row" style="margin-top:8px; gap:6px; align-items:center">
            <input id="confirmPub" type="checkbox" style="width:auto"> Tôi xác nhận tạo bản ghi xuất bản
          </label>
          <div class="row" style="margin-top:8px">
            <button id="publishBtn" disabled>Tạo bản ghi xuất bản</button>
          </div>
          <div id="pubMsg" class="notice"></div>
        </div>
      </div>
    </div>
  </main>
</div>
<script>
const $ = (id) => document.getElementById(id);
let current = null;
let poller = null;
let thumbnailObjectUrls = [];

// Read bearer token from URL fragment (#token=...) — fragment is never sent
// to the server so the token never appears in access logs.  Not persisted.
let _tok = '';
(function () {
  const m = location.hash.replace(/^#/, '').match(/(?:^|&)token=([^&]*)/);
  if (m) _tok = decodeURIComponent(m[1]);
})();

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  if (_tok) opts.headers['Authorization'] = 'Bearer ' + _tok;
  const res = await fetch(path, opts);
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) { throw new Error(data.error_vi || data.error || ('HTTP ' + res.status)); }
  return data;
}

async function authFetch(url) {
  if (!_tok) return url;
  try {
    const resp = await fetch(url, { headers: { Authorization: 'Bearer ' + _tok } });
    if (!resp.ok) return url;
    return URL.createObjectURL(await resp.blob());
  } catch (e) { return url; }
}

function clearThumbnailObjectUrls() {
  for (const url of thumbnailObjectUrls) URL.revokeObjectURL(url);
  thumbnailObjectUrls = [];
}

async function renderThumbnails() {
  const card = $('thumbnailCard');
  const grid = $('thumbnailGrid');
  const message = $('thumbnailMsg');
  clearThumbnailObjectUrls();

  let result;
  try {
    result = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/thumbnails');
  } catch (e) {
    card.style.display = '';
    grid.replaceChildren();
    message.innerHTML = '<span class="err">Lỗi tải ảnh bìa: ' + e.message + '</span>';
    return;
  }

  const doc = result.thumbnails || {};
  const candidates = Array.isArray(doc.candidates) ? doc.candidates : [];
  if (!result.present || !candidates.length) {
    card.style.display = 'none';
    grid.replaceChildren();
    message.textContent = '';
    return;
  }

  card.style.display = '';
  grid.replaceChildren();
  message.textContent = '';

  for (const candidate of candidates) {
    if (!candidate || typeof candidate.file !== 'string') continue;

    const item = document.createElement('div');
    item.className = 'thumb-item' + (candidate.file === doc.primary_candidate ? ' selected' : '');

    const img = document.createElement('img');
    img.alt = 'Ảnh bìa ứng viên ' + (candidate.index || '');
    const artifactUrl = '/api/jobs/' + encodeURIComponent(current)
      + '/artifacts/' + encodeURIComponent(candidate.file);
    const resolved = await authFetch(artifactUrl);
    if (resolved.startsWith('blob:')) thumbnailObjectUrls.push(resolved);
    img.src = resolved;

    const meta = document.createElement('div');
    meta.className = 'muted';
    const seconds = Number(candidate.source_seconds || 0).toFixed(1);
    meta.textContent = candidate.file + ' · ' + seconds + 's';

    const button = document.createElement('button');
    const selected = candidate.file === doc.primary_candidate;
    button.textContent = selected ? 'Đang dùng' : 'Chọn ảnh này';
    button.disabled = selected;
    button.onclick = async () => {
      try {
        await api(
          'POST',
          '/api/jobs/' + encodeURIComponent(current) + '/thumbnails/select',
          { candidate: candidate.file }
        );
        message.innerHTML = '<span class="ok">Đã chọn ' + candidate.file + ' làm ảnh bìa.</span>';
        await loadStatus();
      } catch (e) {
        message.innerHTML = '<span class="err">' + e.message + '</span>';
      }
    };

    item.append(img, meta, button);
    grid.appendChild(item);
  }
}

async function loadJobs() {
  try {
    const { jobs } = await api('GET', '/api/jobs');
    const el = $('jobList');
    if (!jobs.length) { el.innerHTML = '<div class="muted">Chưa có job nào.</div>'; return; }
    el.innerHTML = '';
    for (const j of jobs) {
      const div = document.createElement('div');
      div.className = 'jobrow' + (j.job_id === current ? ' active' : '');
      div.innerHTML = `<div><strong>${j.job_id}</strong> ${j.running ? '· <span class="warn">đang chạy</span>' : ''}</div>`
        + `<div class="muted">${j.ready_count}/${j.stage_count} bước · ${j.aspect_ratio} · ${j.language}`
        + `${j.complete ? ' · <span class="ok">hoàn tất</span>' : ''}</div>`;
      div.onclick = () => selectJob(j.job_id);
      el.appendChild(div);
    }
  } catch (e) { $('jobList').textContent = 'Lỗi tải danh sách: ' + e.message; }
}

function selectJob(id) {
  clearThumbnailObjectUrls();
  current = id;
  $('empty').style.display = 'none';
  $('detail').style.display = '';
  loadStatus();
  loadJobs();
}

function badge(stage) {
  return `<span class="badge ${stage.status}" title="${stage.status_hint||''}">${stage.status_label}</span>`;
}

async function loadStatus() {
  if (!current) return;
  let s;
  try { s = await api('GET', '/api/jobs/' + encodeURIComponent(current)); }
  catch (e) { $('progText').textContent = 'Lỗi: ' + e.message; return; }

  $('jobTitle').textContent = 'Job: ' + s.job_id;
  const total = s.stages.length;
  const ready = s.counts.ready || 0;
  $('progBar').style.width = Math.round(ready * 100 / total) + '%';
  $('progText').textContent = `${ready}/${total} bước hoàn tất · ${s.complete ? 'đã xong' : 'đang tiến hành'}`
    + (s.running ? ' · đang chạy…' : '');
  $('runErr').textContent = s.run_error_vi ? ('Lỗi chạy: ' + s.run_error_vi) : '';
  $('runBtn').disabled = !!s.running;

  $('stages').innerHTML = s.stages.map(st =>
    `<div class="stage"><div class="name">${st.stage_label}</div>${badge(st)}`
    + `<div class="muted">${st.message_vi || ''}</div></div>`).join('');

  const vc = $('videoCard');
  if (s.has_final_video) {
    vc.style.display = '';
    const src = '/api/jobs/' + encodeURIComponent(current) + '/artifacts/final.mp4';
    const v = $('video');
    if (v.dataset.src !== src) {
      v.dataset.src = src;
      authFetch(src).then(resolved => { v.src = resolved; });
    }
  } else { vc.style.display = 'none'; }

  if (s.has_thumbnail) {
    await renderThumbnails();
  } else {
    clearThumbnailObjectUrls();
    $('thumbnailCard').style.display = 'none';
    $('thumbnailGrid').replaceChildren();
    $('thumbnailMsg').textContent = '';
  }

  const artHtml = s.artifacts.length
    ? s.artifacts.map(a => `<li><a href="${a.href}" data-auth-href="${a.href}" target="_blank">${a.name}</a> `
        + `<span class="muted">(${a.kind}, ${a.size} B)</span></li>`).join('')
    : '<li class="muted">Chưa có artifact.</li>';
  $('artifacts').innerHTML = artHtml;
  if (_tok) {
    $('artifacts').querySelectorAll('a[data-auth-href]').forEach(a => {
      a.onclick = async (e) => {
        e.preventDefault();
        const resolved = await authFetch(a.dataset.authHref);
        const tmp = document.createElement('a');
        tmp.href = resolved; tmp.download = a.textContent.trim(); tmp.click();
      };
    });
  }

  renderMeta(s.approvals);
  if (poller) { clearInterval(poller); poller = null; }
  if (s.running) { poller = setInterval(loadStatus, 1500); }
}

let metaLoaded = null;
let scriptLoaded = null;
async function renderMeta(approvals) {
  $('scriptState').innerHTML = approvals.script_present
    ? (approvals.script_approved ? '<span class="ok">Kịch bản đã được duyệt.</span>'
        : '<span class="warn">Kịch bản chưa được duyệt.</span>')
    : '<span class="muted">Chưa có kịch bản (chạy pipeline tới bước Kịch bản).</span>';
  if (scriptLoaded !== current) {
    scriptLoaded = current;
    try {
      const { present, script } = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/script');
      if (present && script.sections !== undefined) {
        $('scriptSections').value = JSON.stringify(script.sections, null, 2);
      }
    } catch (e) { /* ignore */ }
  }
  $('saveScriptBtn').disabled = !approvals.script_present;
  $('approveScriptBtn').disabled = !approvals.script_present;

  $('metaState').innerHTML = approvals.metadata_present
    ? (approvals.metadata_approved ? '<span class="ok">Siêu dữ liệu đã được duyệt.</span>'
        : '<span class="warn">Siêu dữ liệu chưa được duyệt.</span>')
    : '<span class="muted">Chưa có siêu dữ liệu (chạy pipeline tới bước Siêu dữ liệu).</span>';
  if (metaLoaded !== current) {
    metaLoaded = current;
    try {
      const { present, metadata } = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/metadata');
      if (present) {
        $('metaTitle').value = metadata.title || '';
        $('metaDesc').value = metadata.description || '';
        $('metaTags').value = (metadata.tags || []).join(', ');
      }
    } catch (e) { /* ignore */ }
  }
  const canPublish = approvals.script_approved && approvals.metadata_approved && $('confirmPub').checked;
  $('publishBtn').disabled = !canPublish;
  $('approveBtn').disabled = !approvals.metadata_present;
  $('saveMetaBtn').disabled = !approvals.metadata_present;
}

$('createForm').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  try {
    const s = await api('POST', '/api/jobs', payload);
    $('createMsg').innerHTML = '<span class="ok">Đã tạo job ' + s.job_id + '.</span>';
    e.target.reset();
    await loadJobs();
    selectJob(s.job_id);
  } catch (err) { $('createMsg').innerHTML = '<span class="err">' + err.message + '</span>'; }
};

$('runBtn').onclick = async () => {
  try { await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/run', {}); loadStatus(); }
  catch (e) { $('runErr').textContent = e.message; }
};
$('refreshBtn').onclick = loadStatus;

$('saveMetaBtn').onclick = async () => {
  const tags = $('metaTags').value.split(',').map(t => t.trim()).filter(Boolean);
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/metadata',
      { title: $('metaTitle').value, description: $('metaDesc').value, tags });
    $('metaMsg').innerHTML = '<span class="ok">Đã lưu. Cần duyệt lại trước khi xuất bản.</span>';
    metaLoaded = null; loadStatus();
  } catch (e) { $('metaMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

$('approveBtn').onclick = async () => {
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/metadata/approve', {});
    $('metaMsg').innerHTML = '<span class="ok">Đã duyệt siêu dữ liệu.</span>';
    loadStatus();
  } catch (e) { $('metaMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

$('confirmPub').onchange = () => loadStatus();

$('publishBtn').onclick = async () => {
  try {
    const r = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/publish', { confirm: true });
    $('pubMsg').innerHTML = r.published
      ? '<span class="ok">' + (r.message_vi || 'Đã tạo bản ghi xuất bản.') + '</span>'
      : '<span class="warn">' + (r.message_vi || 'Chưa thể xuất bản.') + '</span>';
    loadStatus();
  } catch (e) { $('pubMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

$('saveScriptBtn').onclick = async () => {
  try {
    const sections = JSON.parse($('scriptSections').value || '[]');
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/script', { sections });
    $('scriptMsg').innerHTML = '<span class="ok">Đã lưu. Cần duyệt lại trước khi xuất bản.</span>';
    scriptLoaded = null; loadStatus();
  } catch (e) { $('scriptMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

$('approveScriptBtn').onclick = async () => {
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/script/approve', {});
    $('scriptMsg').innerHTML = '<span class="ok">Đã duyệt kịch bản.</span>';
    loadStatus();
  } catch (e) { $('scriptMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

loadJobs();
</script>
</body>
</html>
"""
