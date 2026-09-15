"""Process-environment fixes that must run *before* Isaac Sim's ``AppLauncher`` is created.

This box is shared by several desktop users, and ``~/.bashrc`` exports ``DISPLAY=:2`` which belongs
to a different login session. The NVIDIA Vulkan ICD cannot initialise against another user's X
server, so ``vkCreateInstance`` returns ``VK_ERROR_INCOMPATIBLE_DRIVER`` and Kit dies with::

    [carb.graphics-vulkan.plugin] vkCreateInstance failed. Vulkan 1.1 is not supported, ...

:func:`prepare_display` picks the X display that actually belongs to the current user (from
``who``) and exports it, which makes Isaac Sim start normally — headless included, because Kit
still needs a working Vulkan instance to render.

Override with ``V2S2R_DISPLAY=:N`` (or ``V2S2R_DISPLAY=none`` to unset ``DISPLAY`` entirely).
"""

from __future__ import annotations

import ctypes
import getpass
import os
import subprocess


def _vulkan_ok() -> bool:
    """True when ``vkCreateInstance`` succeeds in this process' current environment."""
    try:
        lib = ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return False

    class _AppInfo(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
            ("pApplicationName", ctypes.c_char_p), ("applicationVersion", ctypes.c_uint32),
            ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
            ("apiVersion", ctypes.c_uint32),
        ]

    class _CreateInfo(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
            ("pApplicationInfo", ctypes.POINTER(_AppInfo)),
            ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
            ("enabledExtensionCount", ctypes.c_uint32), ("ppEnabledExtensionNames", ctypes.c_void_p),
        ]

    app = _AppInfo(0, None, b"v2s2r", 1, b"v2s2r", 1, (1 << 22) | (3 << 12))
    info = _CreateInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    instance = ctypes.c_void_p()
    try:
        result = lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance))
    except Exception:
        return False
    if result == 0:
        try:
            lib.vkDestroyInstance(instance, None)
        except Exception:
            pass
        return True
    return False


def _x_display_ok(display: str | None = None) -> bool:
    """True when an X client can actually *connect* to the display (auth included).

    Vulkan can initialise against a display this user is not authorised for, so ``_vulkan_ok`` is
    not enough when a window is wanted: GLFW then fails with "Invalid MIT-MAGIC-COOKIE-1 key" and
    Kit runs on with no window. ``XOpenDisplay`` is the same test GLFW does.
    """
    name = display if display is not None else os.environ.get("DISPLAY")
    if not name:
        return False
    try:
        x11 = ctypes.CDLL("libX11.so.6")
    except OSError:
        return False
    x11.XOpenDisplay.restype = ctypes.c_void_p
    handle = x11.XOpenDisplay(name.encode())
    if not handle:
        return False
    try:
        x11.XCloseDisplay(ctypes.c_void_p(handle))
    except Exception:
        pass
    return True


def _user_displays(user: str) -> list[str]:
    """X displays owned by *user*, newest first, according to ``who``."""
    try:
        out = subprocess.run(["who"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    displays = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == user and parts[1].startswith(":"):
            displays.append(parts[1])
    return displays


def prepare_display(verbose: bool = True, require_window: bool = False) -> str | None:
    """Make sure ``DISPLAY`` points at an X server this user can actually use.

    With ``require_window=True`` the display must also accept an X connection (``XOpenDisplay``),
    which is what GLFW needs to open a window. Vulkan alone can succeed on a display owned by
    another user, and Kit then starts windowless with only a "GLFW initialization failed" warning.

    Returns the DISPLAY value in effect afterwards (``None`` if unset).
    """

    def _usable() -> bool:
        return _vulkan_ok() and (not require_window or _x_display_ok())

    override = os.environ.get("V2S2R_DISPLAY")
    if override:
        if override.lower() in {"none", "unset", ""}:
            os.environ.pop("DISPLAY", None)
            if verbose:
                print("[v2s2r] V2S2R_DISPLAY=none -> DISPLAY unset")
            return None
        os.environ["DISPLAY"] = override
        if verbose:
            print(f"[v2s2r] V2S2R_DISPLAY -> DISPLAY={override}")
        return override

    if _usable():
        return os.environ.get("DISPLAY")

    current = os.environ.get("DISPLAY")
    user = getpass.getuser()
    reason = "no window could be opened on" if require_window else "Vulkan could not initialise with"
    for candidate in _user_displays(user):
        if candidate == current:
            continue
        os.environ["DISPLAY"] = candidate
        if _usable():
            if verbose:
                print(
                    f"[v2s2r] {reason} DISPLAY={current!r}; "
                    f"switched to {candidate!r} (owned by {user})."
                )
            return candidate

    if require_window:
        # a window was asked for and no display can provide one - say so instead of running blind
        if current is not None:
            os.environ["DISPLAY"] = current
        if verbose:
            print(
                f"[v2s2r][WARN] no X display this user can open (tried {current!r} and "
                f"{_user_displays(user) or 'none from `who`'}). A window cannot be created — Kit "
                "will warn 'GLFW initialization failed' and run windowless.\n"
                "              Run from a terminal inside your own desktop session, or set "
                "V2S2R_DISPLAY=:<display>.",
                flush=True,
            )
        return current

    # last resort: no display at all
    os.environ.pop("DISPLAY", None)
    if _vulkan_ok():
        if verbose:
            print(f"[v2s2r] Vulkan could not initialise with DISPLAY={current!r}; unset DISPLAY.")
        return None

    if current is not None:
        os.environ["DISPLAY"] = current
    else:
        os.environ.pop("DISPLAY", None)
    if verbose:
        print(
            "[v2s2r][WARN] Vulkan cannot create an instance in this environment. Isaac Sim will "
            "likely fail with 'vkCreateInstance failed'. Try running from a terminal inside your "
            "own desktop session, or set V2S2R_DISPLAY=:<your display>."
        )
    return current


def hard_exit(simulation_app, status: int = 0, close_timeout: float = 20.0) -> None:
    """Close Isaac Sim and make sure the process actually exits.

    Kit 107 (Isaac Sim 5.1) regularly hangs inside ``SimulationApp.close()`` on this machine —
    the app finishes its work, then blocks forever in shutdown (telemetry / cache locks), which
    would stall any batch script. Close on a daemon thread, wait a bounded time, then exit hard.
    """
    import sys
    import threading

    thread = threading.Thread(target=simulation_app.close, daemon=True)
    thread.start()
    thread.join(timeout=close_timeout)
    if thread.is_alive():
        print("[v2s2r] Isaac Sim shutdown is hanging; exiting the process directly.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def check_memory(required_gb: float = 12.0, verbose: bool = True) -> float:
    """Warn when there is not enough free RAM to run a replay.

    A replay of a 5-object scene peaks around 10 GB resident. This box is shared, and when it runs
    out the kernel OOM-kills the process ("Killed") after the desktop has already frozen while
    swapping - so it is worth saying something *before* starting Isaac Sim.

    Returns the available memory in GB (``-1`` if it could not be determined).
    """
    try:
        with open("/proc/meminfo") as f:
            info = {
                line.split(":")[0]: int(line.split()[1])
                for line in f
                if line.split(":")[0] in ("MemAvailable", "SwapFree", "SwapTotal")
            }
    except Exception:
        return -1.0

    available_gb = info.get("MemAvailable", 0) / 1024**2
    swap_free_gb = info.get("SwapFree", 0) / 1024**2
    swap_total_gb = info.get("SwapTotal", 0) / 1024**2

    if verbose and available_gb < required_gb:
        print(
            f"[v2s2r][WARN] only {available_gb:.1f} GB of RAM available "
            f"(swap {swap_free_gb:.1f}/{swap_total_gb:.1f} GB free); a replay peaks around 10 GB.\n"
            "              Free some memory first (browser/editor windows are usually the biggest\n"
            "              offenders) or the kernel will OOM-kill the run with a bare 'Killed'.\n"
            "              `--no-render` roughly halves the requirement.",
            flush=True,
        )
    return available_gb
