"""``python -m movie_review_factory`` entry point.

Lets the Windows desktop installer launch the dashboard with the bundled
interpreter (``python.exe -m movie_review_factory serve``) without depending on a
console-script wrapper being on PATH. Delegates to the same Typer app as the
``mrf`` console command.
"""
from __future__ import annotations

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
