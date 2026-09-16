"""CPU placement for consistent timing on hybrid laptops.

Intel 12th-gen and later laptop CPUs mix fast performance cores (P-cores) with
slower efficiency cores (E-cores). For a single recommendation request the GPU
finishes almost instantly, so latency mostly measures how fast the CPU hands work
to the GPU - and Windows may run Python on either core type. Measured on an
i7-12650H: ~1.5 ms on P-cores vs ~3.2 ms on E-cores for the same model.

So the benchmark pins itself to the performance cores. On a CPU without core
types (or outside Windows) nothing changes.
"""

import ctypes
import platform

_CPU_SET_INFORMATION = 0


def cpu_name():
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown CPU"


def _efficiency_classes():
    """{logical CPU index: efficiency class} for processor group 0, via Windows."""
    k32 = ctypes.windll.kernel32
    fn = k32.GetSystemCpuSetInformation
    fn.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                   ctypes.c_void_p, ctypes.c_ulong]
    needed = ctypes.c_ulong(0)
    fn(None, 0, ctypes.byref(needed), None, 0)          # first call reports the size
    if not needed.value:
        return {}
    buf = (ctypes.c_ubyte * needed.value)()
    if not fn(ctypes.cast(buf, ctypes.c_void_p), needed, ctypes.byref(needed), None, 0):
        return {}
    raw, classes, off = bytes(buf), {}, 0
    # SYSTEM_CPU_SET_INFORMATION: Size(4) Type(4) Id(4) Group(2) LogicalProcessorIndex(1)
    # CoreIndex(1) LastLevelCacheIndex(1) NumaNodeIndex(1) EfficiencyClass(1) ...
    while off + 19 <= len(raw):
        size = int.from_bytes(raw[off:off + 4], "little")
        if size == 0:
            break
        if int.from_bytes(raw[off + 4:off + 8], "little") == _CPU_SET_INFORMATION:
            if int.from_bytes(raw[off + 12:off + 14], "little") == 0:
                classes[raw[off + 14]] = raw[off + 18]
        off += size
    return classes


def pin_to_performance_cores():
    """Pin this process to the highest-efficiency-class cores.

    Returns a short description of what was done, for the report.
    """
    if platform.system() != "Windows":
        return "not pinned (not Windows)"
    try:
        classes = _efficiency_classes()
    except Exception as exc:                         # never let this stop a benchmark
        return f"not pinned (could not read core types: {exc.__class__.__name__})"
    if len(set(classes.values())) < 2:
        return f"not pinned (no P-core/E-core split among {len(classes)} logical CPUs)"

    top = max(classes.values())
    perf = sorted(lp for lp, c in classes.items() if c == top)
    mask = sum(1 << lp for lp in perf)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
        return "not pinned (Windows refused the affinity change)"
    return (f"pinned to {len(perf)} performance-core threads "
            f"(logical CPUs {perf[0]}-{perf[-1]} of {len(classes)})")
