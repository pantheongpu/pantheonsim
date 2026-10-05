"""NVML NVLink, fabric, topology, peer-to-peer, drain state and system events, against the shim.

Run by run_nvml_category.sh once per profile (h100: 18 NVLink ports; gh200: the same plus a C2C link;
rtx3060: no NVLink). The simulated machine has no NVLink fabric -- every GPU shares one host bridge --
so the ports exist but none is connected.
"""
import ctypes, sys
from nvml_support import *

lib = Lib(sys.argv[1])
profile = sys.argv[2]
PORTS = {"nvidia/h100": (18, 6), "nvidia/gh200-480gb": (18, 6), "nvidia/rtx3060": (0, 0)}[profile]
GEFORCE = profile == "nvidia/rtx3060"
check("nvmlInit_v2", lib.lib.nvmlInit_v2() == SUCCESS)
h0, h1 = lib.handle(0), lib.handle(1)


def val(name, h, ctype=ctypes.c_uint, *extra):
    x = ctype()
    rc = lib(name, h, *extra, ref(x))
    return rc, x.value


# ---- NVLink ports -------------------------------------------------------------------------------------------------------------------
links, ver = PORTS
if links:
    for link in (0, links - 1):
        check(f"link {link}: inactive (nothing is cabled)", val("nvmlDeviceGetNvLinkState", h1, ctypes.c_int, u(link)) == (SUCCESS, 0))
        check(f"link {link}: the generation's version", val("nvmlDeviceGetNvLinkVersion", h1, ctypes.c_uint, u(link)) == (SUCCESS, ver))
    check("a link past the last is invalid", val("nvmlDeviceGetNvLinkState", h1, ctypes.c_int, u(links))[0] == INVALID_ARGUMENT)
    check("capability: valid link", val("nvmlDeviceGetNvLinkCapability", h1, ctypes.c_uint, u(0), ctypes.c_int(5)) == (SUCCESS, 1))
    check("capability: P2P over NVLink", val("nvmlDeviceGetNvLinkCapability", h1, ctypes.c_uint, u(0), ctypes.c_int(0)) == (SUCCESS, 1))
    check("capability: no SLI bridge", val("nvmlDeviceGetNvLinkCapability", h1, ctypes.c_uint, u(0), ctypes.c_int(4)) == (SUCCESS, 0))
    check("a capability that does not exist", val("nvmlDeviceGetNvLinkCapability", h1, ctypes.c_uint, u(0), ctypes.c_int(99))[0] == INVALID_ARGUMENT)
    check("remote device of an unconnected link is unknown", val("nvmlDeviceGetNvLinkRemoteDeviceType", h1, ctypes.c_int, u(0)) == (SUCCESS, 0xFF))
    pci = ctypes.create_string_buffer(128)
    check("no remote PCI info", lib("nvmlDeviceGetNvLinkRemotePciInfo_v2", h1, u(0), pci) == NOT_SUPPORTED)
    check("no remote PCI info (v1 name)", lib("nvmlDeviceGetNvLinkRemotePciInfo", h1, u(0), pci) == NOT_SUPPORTED)
    check("no error counters without a connected link", val("nvmlDeviceGetNvLinkErrorCounter", h1, ctypes.c_ulonglong, u(0), ctypes.c_int(0))[0] == NOT_SUPPORTED)
    check("a counter that does not exist", val("nvmlDeviceGetNvLinkErrorCounter", h1, ctypes.c_ulonglong, u(0), ctypes.c_int(77))[0] == INVALID_ARGUMENT)
    rx, tx = ull(), ull()
    check("utilization counters are gone (deprecated)", lib("nvmlDeviceGetNvLinkUtilizationCounter", h1, u(0), u(0), ref(rx), ref(tx)) == NOT_SUPPORTED)
    check("utilization counter 2 does not exist", lib("nvmlDeviceGetNvLinkUtilizationCounter", h1, u(0), u(2), ref(rx), ref(tx)) == INVALID_ARGUMENT)
else:
    for name, args in [("nvmlDeviceGetNvLinkState", [u(0), ref(ctypes.c_int())]), ("nvmlDeviceGetNvLinkVersion", [u(0), ref(u())]),
                       ("nvmlDeviceGetNvLinkCapability", [u(0), ctypes.c_int(0), ref(u())]),
                       ("nvmlDeviceGetNvLinkRemoteDeviceType", [u(0), ref(ctypes.c_int())]),
                       ("nvmlDeviceGetNvLinkErrorCounter", [u(0), ctypes.c_int(0), ref(ull())])]:
        check(f"{name}: a GeForce card has no NVLink (as measured)", lib(name, h1, *args) == NOT_SUPPORTED)


# The version words of nvmlNvLinkInfo are the header's sizes (the v2 struct carries the firmware table),
# so only the refusal of a wrong one is checked here.
ni = ctypes.create_string_buffer(4096)
ctypes.memset(ni, 0, 4096)
ctypes.c_uint.from_buffer(ni).value = 7
rc = lib("nvmlDeviceGetNvLinkInfo", h1, ni)
if rc is not None:
    check("NVLink info with a wrong version", rc == VERSION_MISMATCH, rc)
    check("NVLink info with a NULL pointer", lib("nvmlDeviceGetNvLinkInfo", h1, None) == INVALID_ARGUMENT)

# ---- C2C ----------------------------------------------------------------------------------------------------------------------------------
c2c = ctypes.create_string_buffer(8)
rc = lib("nvmlDeviceGetC2cModeInfoV", h1, c2c)
if rc is not None:
    check("C2C: the Grace link of a GH200, nothing elsewhere", rc == (SUCCESS if profile == "nvidia/gh200-480gb" else NOT_SUPPORTED), rc)

# ---- the fabric ---------------------------------------------------------------------------------------------------------------------------
fi = ctypes.create_string_buffer(b"\xaa" * 64, 64)
rc = lib("nvmlDeviceGetGpuFabricInfo", h1, fi)
check("fabric info: no fabric, nothing set", rc == SUCCESS and fi.raw[:28] == b"\x00" * 28, (rc, fi.raw[:28]))
fv = ctypes.create_string_buffer(64)
rc = lib("nvmlDeviceGetGpuFabricInfoV", h1, fv)
if rc is not None:
    check("fabric info, versioned: a zero version is a mismatch", rc == VERSION_MISMATCH, rc)

# ---- topology and peer-to-peer ------------------------------------------------------------------------------------------------------------------
level = ctypes.c_int(-1)
check("two GPUs share a host bridge (30)", lib("nvmlDeviceGetTopologyCommonAncestor", h0, h1, ref(level)) == SUCCESS and level.value == 30, level.value)
check("a GPU with itself: N/A, as on the card", lib("nvmlDeviceGetTopologyCommonAncestor", h1, h1, ref(level)) == NOT_SUPPORTED)
check("a bad handle", lib("nvmlDeviceGetTopologyCommonAncestor", h0, BAD, ref(level)) == INVALID_ARGUMENT)
n = u(0)
arr = (ctypes.c_void_p * 8)()
check("GPUs at the host bridge: the sizing call", lib("nvmlDeviceGetTopologyNearestGpus", h0, ctypes.c_int(30), ref(n), arr) == SUCCESS and n.value == 1, n.value)
n = u(8)
check("and the array", lib("nvmlDeviceGetTopologyNearestGpus", h0, ctypes.c_int(30), ref(n), arr) == SUCCESS and n.value == 1 and arr[0] == h1.value, (n.value, arr[0]))
n = u(0)
check("none nearer than a host bridge", lib("nvmlDeviceGetTopologyNearestGpus", h0, ctypes.c_int(20), ref(n), arr) == SUCCESS and n.value == 0)
check("a level that is none of NVML's", lib("nvmlDeviceGetTopologyNearestGpus", h0, ctypes.c_int(25), ref(n), arr) == INVALID_ARGUMENT)
n = u(0)
import os
cpu = min(os.sched_getaffinity(0))
check("GPUs near a CPU the process may use", lib("nvmlSystemGetTopologyGpuSet", u(cpu), ref(n), arr) == SUCCESS and n.value == 2, n.value)
check("a CPU that does not exist", lib("nvmlSystemGetTopologyGpuSet", u(100000), ref(n), arr) == INVALID_ARGUMENT)

CAPS = {"READ": 0, "WRITE": 1, "NVLINK": 2, "ATOMICS": 3, "PCI": 4, "UNKNOWN": 5}
want = {"READ": 1, "WRITE": 1, "NVLINK": 5, "ATOMICS": 5, "PCI": 5, "UNKNOWN": 0} if GEFORCE else \
       {"READ": 0, "WRITE": 0, "NVLINK": 5, "ATOMICS": 0, "PCI": 0, "UNKNOWN": 0}
for name, idx in CAPS.items():
    st = ctypes.c_int(-1)
    rc = lib("nvmlDeviceGetP2PStatus", h0, h1, ctypes.c_int(idx), ref(st))
    check(f"P2P {name}", (rc, st.value) == (SUCCESS, want[name]), (rc, st.value))
check("P2P index 6 does not exist", lib("nvmlDeviceGetP2PStatus", h0, h1, ctypes.c_int(6), ref(ctypes.c_int())) == INVALID_ARGUMENT)

# ---- drain state -----------------------------------------------------------------------------------------------------------------------------------
class Pci(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


p1 = Pci()
lib.lib.nvmlDeviceGetPciInfo_v3(h1, ref(p1))
st = ctypes.c_int(-1)
check("drain state: off", lib("nvmlDeviceQueryDrainState", ref(p1), ref(st)) == SUCCESS and st.value == 0)
check("drain state of an address that is no GPU", lib("nvmlDeviceQueryDrainState", ref(Pci()), ref(st)) == NOT_FOUND)
check("drain state with NULL", lib("nvmlDeviceQueryDrainState", None, ref(st)) == INVALID_ARGUMENT)

# ---- system events -----------------------------------------------------------------------------------------------------------------------------------
class SetReq(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("set", ctypes.c_void_p)]


class RegReq(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("types", ctypes.c_ulonglong), ("set", ctypes.c_void_p)]


class EvData(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulonglong), ("gpu", ctypes.c_uint)]


class WaitReq(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("timeoutms", ctypes.c_uint), ("set", ctypes.c_void_p),
                ("data", ctypes.c_void_p), ("dataSize", ctypes.c_uint), ("numEvent", ctypes.c_uint)]


if lib.has("nvmlSystemEventSetCreate"):
    sr = SetReq(version(SetReq, 1), None)
    check("create a system event set", lib("nvmlSystemEventSetCreate", ref(sr)) == SUCCESS and sr.set)
    rr = RegReq(version(RegReq, 1), 3, sr.set)
    check("register driver bind and unbind", lib("nvmlSystemRegisterEvents", ref(rr)) == SUCCESS)
    rr.types = 0x100
    check("an event type that does not exist", lib("nvmlSystemRegisterEvents", ref(rr)) == INVALID_ARGUMENT)
    rr.version = 3
    check("register with a wrong version", lib("nvmlSystemRegisterEvents", ref(rr)) == VERSION_MISMATCH)
    ev = (EvData * 4)()
    wr = WaitReq(version(WaitReq, 1), 50, sr.set, ctypes.cast(ev, ctypes.c_void_p), 4, 9)
    check("nothing binds or unbinds a simulated GPU: the wait times out", lib("nvmlSystemEventSetWait", ref(wr)) == TIMEOUT and wr.numEvent == 0)
    wr.version = 1
    check("wait with a wrong version", lib("nvmlSystemEventSetWait", ref(wr)) == VERSION_MISMATCH)
    fr = SetReq(version(SetReq, 1), sr.set)
    check("free the set", lib("nvmlSystemEventSetFree", ref(fr)) == SUCCESS)

# ---- the driver --------------------------------------------------------------------------------------------------------------------------------------
class Branch(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("branch", ctypes.c_char * 80)]


br = Branch(version(Branch, 1), b"")
rc = lib("nvmlSystemGetDriverBranch", ref(br), u(80))
if rc is not None:
    check("driver branch: r<major>_00", rc == SUCCESS and br.branch.startswith(b"r") and br.branch.endswith(b"_00"), (rc, br.branch))
    check("driver branch in a short buffer", lib("nvmlSystemGetDriverBranch", ref(br), u(2)) == INSUFFICIENT_SIZE)
    br.version = 4
    check("driver branch with a wrong version", lib("nvmlSystemGetDriverBranch", ref(br), u(80)) == VERSION_MISMATCH)
mode = u()
rc = lib("nvmlSystemGetNvlinkBwMode", ref(mode))
if rc is not None:
    check("the NVLink bandwidth mode belongs to a fabric manager", rc == NOT_SUPPORTED, rc)
check("shutdown", lib.lib.nvmlShutdown() == SUCCESS)
finish()
