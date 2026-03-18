from __future__ import annotations

import os
import signal
import subprocess
import logging
from typing import Any, Callable, cast


logger = logging.getLogger("ai_platform_samplelib")


def _parse_int_env(var_name: str) -> int | None:
    """Parse an int environment variable; warn and return None on invalid values."""
    value = os.getenv(var_name)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        logger.warning("Invalid %s=%r; falling back to auto-detection", var_name, value)
        return None


def get_host_uid_gid(*, default_uid: int = 1000, default_gid: int = 1000) -> tuple[int, int]:
    """Return host UID/GID in a cross-platform, best-effort way.

    Priority:
    1) AI_PLATFORM_HOST_UID / AI_PLATFORM_HOST_GID (env overrides)
    2) psutil (POSIX-only APIs: Process().uids/gids)
    3) os.getuid / os.getgid (POSIX)
    4) defaults (Windows-safe): 1000/1000 unless overridden
    """

    uid: int | None = None
    gid: int | None = None

    # Prefer YAML-defined overrides.
    try:
        from ai_chat_util_base.config.ai_chat_util_runtime import get_autonomous_runtime_config

        cfg = get_autonomous_runtime_config()
        if cfg.host.uid is not None:
            uid = int(cfg.host.uid)
        if cfg.host.gid is not None:
            gid = int(cfg.host.gid)
    except Exception:
        # Keep best-effort behavior; fall back to env/autodetection.
        pass

    # Backward-compatible env overrides.
    if uid is None:
        uid = _parse_int_env("AI_PLATFORM_HOST_UID")
    if gid is None:
        gid = _parse_int_env("AI_PLATFORM_HOST_GID")

    # psutil (POSIX) fallback
    if uid is None or gid is None:
        try:
            import psutil  # type: ignore

            proc = psutil.Process()
            if uid is None:
                uids_fn = getattr(proc, "uids", None)
                if callable(uids_fn):
                    uids_obj = uids_fn()
                    real_uid = getattr(uids_obj, "real", None)
                    if real_uid is not None:
                        uid = int(real_uid)
            if gid is None:
                gids_fn = getattr(proc, "gids", None)
                if callable(gids_fn):
                    gids_obj = gids_fn()
                    real_gid = getattr(gids_obj, "real", None)
                    if real_gid is not None:
                        gid = int(real_gid)
        except Exception as e:
            # On Windows these APIs don't exist; keep best-effort behavior.
            logger.warning("Failed to detect uid/gid via psutil: %s", e)

    # os.getuid/getgid fallback (POSIX)
    if uid is None:
        getuid = getattr(os, "getuid", None)
        if callable(getuid):
            try:
                getuid_fn = cast(Callable[[], int], getuid)
                uid = int(getuid_fn())
            except Exception as e:
                logger.warning("Failed to detect uid via os.getuid: %s", e)
    if gid is None:
        getgid = getattr(os, "getgid", None)
        if callable(getgid):
            try:
                getgid_fn = cast(Callable[[], int], getgid)
                gid = int(getgid_fn())
            except Exception as e:
                logger.warning("Failed to detect gid via os.getgid: %s", e)

    # Final Windows-safe defaults
    if uid is None:
        uid = int(default_uid)
    if gid is None:
        gid = int(default_gid)

    return uid, gid


def popen_new_process_group_kwargs() -> dict[str, Any]:
    """Return Popen kwargs to create an independent process group/session.

    - POSIX: start a new session (so the child is a session leader and has its own process group).
    - Windows: create a new process group (so it can be targeted independently).
    """

    if os.name == "nt":
        # On Windows, start_new_session is not supported; use creationflags instead.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_process_tree(pid: int) -> None:
    """Best-effort kill for a process tree.

    This is used for cancelling detached tasks.

    - Windows: taskkill /T /F
    - POSIX: kill the whole process group (SIGKILL)

    The function is intentionally best-effort and should not raise if the process
    already exited.
    """

    if not isinstance(pid, int) or pid <= 1:
        return

    if os.name == "nt":
        try:
            # /T: terminate child processes; /F: force.
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            return
        except FileNotFoundError:
            # taskkill should exist on typical Windows installs, but keep fallback.
            pass
        except Exception:
            # If taskkill fails for any reason, fall back to os.kill below.
            pass

        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return

    # POSIX
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    getpgid = getattr(os, "getpgid", None)
    killpg = getattr(os, "killpg", None)

    pgid = pid
    if callable(getpgid):
        try:
            pgid = int(getpgid(pid))  # type: ignore[arg-type]
        except Exception:
            pgid = pid

    if callable(killpg):
        try:
            killpg(pgid, sigkill)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        except Exception:
            # Fall through to per-pid kill.
            pass

    # Fallback to killing just the pid.
    try:
        os.kill(pid, sigkill)
    except ProcessLookupError:
        return


def pid_is_running(pid: int) -> bool:
    """Return True if a pid appears to be alive (best-effort, cross-platform)."""

    if not isinstance(pid, int) or pid <= 1:
        return False

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            OpenProcess = kernel32.OpenProcess
            OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            OpenProcess.restype = wintypes.HANDLE

            GetExitCodeProcess = kernel32.GetExitCodeProcess
            GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            GetExitCodeProcess.restype = wintypes.BOOL

            CloseHandle = kernel32.CloseHandle
            CloseHandle.argtypes = [wintypes.HANDLE]
            CloseHandle.restype = wintypes.BOOL

            handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                # Can't open process handle: assume not running.
                return False
            try:
                code = wintypes.DWORD()
                ok = GetExitCodeProcess(handle, ctypes.byref(code))
                if not ok:
                    return False
                return int(code.value) == STILL_ACTIVE
            finally:
                CloseHandle(handle)
        except Exception:
            # If detection fails, assume not running to avoid tasks getting stuck in "running".
            return False

    # POSIX best-effort
    proc_path = f"/proc/{pid}"
    try:
        if os.path.exists(proc_path):
            return True
    except Exception:
        pass

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True
