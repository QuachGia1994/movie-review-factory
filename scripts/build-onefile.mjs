import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageRoot = path.join(repoRoot, "src", "movie_review_factory");
const templatePath = path.join(repoRoot, "scripts", "onefile-launcher-template.js");
const outputPath = path.join(repoRoot, "movie-review-factory.js");

const runtimeFiles = [
  "__init__.py",
  "agy_agent.py",
  "agy_vision.py",
  "branding.py",
  "creative_brief.py",
  "creator_library.py",
  "chunked_tts.py",
  "analytics.py",
  "audio_mix.py",
  "cli.py",
  "content_agent.py",
  "content_scout.py",
  "copyright_bypass.py",
  "cancellation.py",
  "editor_ops.py",
  "editorial_qa.py",
  "media_qa.py",
  "handoff.py",
  "hook_crafter.py",
  "licensing.py",
  "link_download.py",
  "localization.py",
  "media_store.py",
  "media_intelligence.py",
  "midroll.py",
  "narration_alignment.py",
  "tts_providers.py",
  "batch_queue.py",
  "models.py",
  "pipeline.py",
  "pool_scheduler.py",
  "propainter_setup.py",
  "quick_preview.py",
  "scene_scoring.py",
  "scene_validation.py",
  "watermark_removal.py",
  "mask_detection.py",
  "short_variants.py",
  "thumbnail_editor.py",
  "versions.py",
  "semantic_search.py",
  "visual_rhythm.py",
  "webapp.py",
];

const bundle = {};
for (const name of runtimeFiles) {
  const bytes = fs.readFileSync(path.join(packageRoot, name));
  bundle["movie_review_factory/" + name] = bytes.toString("base64");
}
for (const name of ["man-ke.svg", "man-ke.png"]) {
  bundle["movie_review_factory/assets/" + name] = fs.readFileSync(path.join(packageRoot, "assets", name)).toString("base64");
}
const migrationRoot = path.join(packageRoot, "migrations");
const migrationFiles = fs.readdirSync(migrationRoot)
  .filter(name => name === "__init__.py" || name.endsWith(".sql"))
  .sort();
for (const name of migrationFiles) {
  const bytes = fs.readFileSync(path.join(migrationRoot, name));
  bundle["movie_review_factory/migrations/" + name] = bytes.toString("base64");
}
bundle["pyproject.toml"] = fs.readFileSync(path.join(repoRoot, "pyproject.toml")).toString("base64");

const pyproject = fs.readFileSync(path.join(repoRoot, "pyproject.toml"), "utf8");
const versionMatch = pyproject.match(/^version\s*=\s*"([^"]+)"/m);
const version = versionMatch ? versionMatch[1] : "0.0.0";
const bundleJson = JSON.stringify(bundle);
const hash = crypto.createHash("sha256").update(bundleJson).digest("hex");

let launcher = fs.readFileSync(templatePath, "utf8");
launcher = launcher
  .replace("__MRF_VERSION__", JSON.stringify(version))
  .replace("__MRF_HASH__", JSON.stringify(hash))
  .replace("__MRF_BUNDLE__", bundleJson);

if (/__MRF_(VERSION|HASH|BUNDLE)__/.test(launcher)) {
  throw new Error("one-file launcher template contains unresolved placeholders");
}

const asciiLauncher = launcher.replace(
  /[^\x00-\x7F]/g,
  (char) => "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0"),
);
fs.writeFileSync(outputPath, asciiLauncher, "ascii");
const info = {
  output: outputPath,
  bytes: fs.statSync(outputPath).size,
  version,
  hash,
  embeddedFiles: Object.keys(bundle),
};
console.log(JSON.stringify(info, null, 2));
