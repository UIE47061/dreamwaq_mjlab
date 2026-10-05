"""Wait for MuJoCo's asynchronous native-window destruction before process exit."""

import threading


def join_native_viewer_threads():
    # MuJoCo 3.11 Handle.close() only requests exit. The daemon thread later
    # destroys its GLFW resources; interpreter atexit must not terminate GLFW
    # concurrently. The handle provides no public join API in this version.
    for thread in threading.enumerate():
        target = getattr(thread, "_target", None)
        if (
            getattr(target, "__module__", None) == "mujoco.viewer"
            and getattr(target, "__name__", None) == "_launch_internal"
        ):
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError("MuJoCo native viewer did not finish closing")
