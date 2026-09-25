"""Tool: recent git status/log for a configured project's local repo."""

from __future__ import annotations

import subprocess

from devdesk.config import PROJECTS

_TIMEOUT_SECONDS = 5


def _run(args: list[str], cwd: str) -> str:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "(timed out)"
    if proc.returncode != 0:
        return f"(git error: {proc.stderr.strip()[:200]})"
    return proc.stdout.strip()


def git_status_lookup(project: str) -> dict:
    """Get recent commits and working-tree status for a project's local repo.

    Useful for "where did I leave off" style questions.

    Args:
        project: One of the configured project names.

    Returns:
        A dict with `status`. If the project has no local repo path
        configured, `status` explains that cleanly instead of raising an
        error.
    """
    cfg = PROJECTS.get(project)
    if cfg is None:
        return {"status": f"unknown project '{project}'"}
    if cfg.source_path is None:
        return {"status": "no local repo configured for this project"}

    path = str(cfg.source_path)
    recent_commits = _run(["git", "log", "-5", "--oneline"], cwd=path)
    working_tree = _run(["git", "status", "--short"], cwd=path)
    return {
        "status": "ok",
        "recent_commits": recent_commits,
        "working_tree": working_tree or "(clean)",
    }
