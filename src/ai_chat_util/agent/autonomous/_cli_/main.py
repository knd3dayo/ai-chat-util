"""Backward-compatible CLI entrypoint.

`docker_main.py` contains the Typer app used by this package.
This module keeps the historical import path and module entrypoint stable.
"""

from __future__ import annotations

from .docker_main import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
