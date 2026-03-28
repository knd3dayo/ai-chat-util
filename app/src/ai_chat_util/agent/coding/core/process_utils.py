from coding_agent_util.core.process_utils import (
    _parse_int_env,
    get_host_uid_gid,
    kill_process_tree,
    pid_is_running,
    popen_new_process_group_kwargs,
)

__all__ = [
    "_parse_int_env",
    "get_host_uid_gid",
    "kill_process_tree",
    "pid_is_running",
    "popen_new_process_group_kwargs",
]
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
