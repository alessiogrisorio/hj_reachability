"""NumPy/ctypes interface to the SFF-based signed XY distance in sff.cpp.

Place both files in hj_reachability/vehicle/metrics. Linux, C++17 and OpenMP;
only NumPy is required in Python. The native library compiles on first use.
The result is in metres: positive outside, negative inside, zero on the
boundary of the union of equal-time collision configurations.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import ctypes as ct
import fcntl
import os
import shlex
import subprocess
import tempfile
from time import perf_counter

import numpy as np


_FIELDS = (
    "dt", "ego_braking", "human_braking", "ego_reaction_time",
    "human_reaction_time", "lf", "lr", "ego_length", "ego_width",
    "human_length", "human_width",
)


class _Config(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in _FIELDS] + [
        ("threads", ct.c_int32), ("use_symmetry", ct.c_int32),
    ]


class _Stats(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in (
        "total_groups", "evaluated_groups", "max_time_samples",
        "total_boundary_segments",
    )] + [("symmetry_used", ct.c_int32), ("threads_used", ct.c_int32)]


@dataclass
class SFFMetricResult:
    terminal_values: np.ndarray
    parameters: dict


@lru_cache(maxsize=1)
def _library():
    directory = Path(__file__).resolve().parent
    source, target = directory / "sff.cpp", directory / "_sff.so"
    if not source.is_file():
        raise FileNotFoundError(f"Missing C++ source: {source}")
    # Atomic replacement and a process lock protect concurrent first use.
    with (directory / "._sff_build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
            compiler = shlex.split(os.environ.get("CXX", "g++"))
            fd, temporary = tempfile.mkstemp(prefix="_sff_", suffix=".so", dir=directory)
            os.close(fd)
            try:
                command = compiler + ["-O3", "-march=native", "-DNDEBUG", "-std=c++17",
                                      "-fopenmp", "-fPIC", "-shared", str(source), "-o", temporary]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError("sff.cpp compilation failed:\n" + result.stderr)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    native = ct.CDLL(str(target))
    for name, restype in (("sff_abi_version", ct.c_int),
                          ("sff_config_size", ct.c_size_t),
                          ("sff_stats_size", ct.c_size_t)):
        function = getattr(native, name)
        function.argtypes, function.restype = [], restype
    if (native.sff_abi_version() != 1 or native.sff_config_size() != ct.sizeof(_Config)
            or native.sff_stats_size() != ct.sizeof(_Stats)):
        raise RuntimeError("SFF ABI mismatch; delete _sff.so and restart Python")
    native.sff_last_error.argtypes, native.sff_last_error.restype = [], ct.c_char_p
    native.sff_default_config.argtypes = [ct.POINTER(_Config)]
    native.sff_default_config.restype = None
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    sizes = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    native.sff_compute_grid.argtypes = [doubles, sizes, ct.POINTER(_Config),
                                       doubles, ct.POINTER(_Stats)]
    native.sff_compute_grid.restype = ct.c_int
    native.sff_compute_points.argtypes = [doubles, ct.c_uint64, doubles,
                                         ct.POINTER(_Config), doubles, ct.POINTER(_Stats)]
    native.sff_compute_points.restype = ct.c_int
    return native


def _config(native, parameters):
    config = _Config()
    native.sff_default_config(ct.byref(config))
    unknown = set(parameters) - set(_FIELDS) - {"threads", "use_symmetry"}
    if unknown:
        raise TypeError(f"Unknown SFF parameters: {sorted(unknown)}")
    for name, value in parameters.items():
        if name == "threads":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("threads must be an integer >= 0")
            if not 0 <= value <= np.iinfo(np.int32).max:
                raise ValueError("threads is outside the int32 range")
            value = int(value)
        elif name == "use_symmetry":
            if value not in (True, False, 0, 1):
                raise ValueError("use_symmetry must be a boolean")
            value = int(value)
        else:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        setattr(config, name, value)
    metadata = {name: getattr(config, name) for name, _ in config._fields_}
    metadata.update(
        model="SFF-based deterministic stopping-procedure conflict set",
        terminal_value_definition="Signed Euclidean XY distance to the entire sampled unsafe union",
        unsafe_set_definition="Equal-time vehicle footprints intersect during the stopping procedures",
        distance_units="m", distance_coordinates=["x_rel", "y_rel"],
        fixed_coordinates=["theta_rel", "v_H", "delta_E", "v_E"],
        sign_convention="V0 <= 0 unsafe",
        prediction="Constant speed during delay; constant braking to rest; frozen ego steering; human yaw=0",
        horizon_policy="Until both vehicles stop; then both remain stationary",
        time_discretization="Uniform samples plus exact delay and stopping events",
        geometry_backend="Convex polygon edge clipping and boundary BVH",
        spatial_rasterization=False, grid_clipping=False,
        footprint_dilation=0.0, regularization="none", normalization="none",
        geometric_roundoff_tolerance_m=1e-10,
        original_nvidia_potential=False,
    )
    return config, metadata


def _check(native, status):
    if status:
        raise RuntimeError(native.sff_last_error().decode("utf-8", errors="replace"))


def _metadata(metadata, stats, elapsed):
    metadata.update({name: getattr(stats, name) for name, _ in stats._fields_})
    metadata["symmetry_used"] = bool(stats.symmetry_used)
    metadata["native_compute_seconds"] = elapsed


def metricSFF(grid, dynamics, ego=None, human=None, **parameters):
    """Compute V0 on axes [x,y,theta,vH,deltaE,vE], in metres.

    Defaults: dt=.01 s; braking=7 m/s² for both; reaction times=0 s.
    threads=0 uses the OpenMP runtime / OMP_NUM_THREADS. Geometry is shared
    across XY positions and, when the axes allow it, reflected slices.
    The sampled union approximates the continuous-time unsafe set; check
    temporal convergence separately. No interpolation of XY distances.
    """
    axes = tuple(np.asarray(a, dtype=np.float64) for a in grid.coordinate_vectors)
    if len(axes) != 6 or any(a.ndim != 1 or a.size == 0 for a in axes):
        raise ValueError("Expected six nonempty one-dimensional coordinate vectors")
    parameters = dict(parameters, lf=float(dynamics.lf), lr=float(dynamics.lr))
    if ego is not None:
        parameters.update(ego_length=float(ego.length), ego_width=float(ego.width))
    if human is not None:
        parameters.update(human_length=float(human.length), human_width=float(human.width))
    native = _library()
    config, metadata = _config(native, parameters)
    shape_tuple = tuple(a.size for a in axes)
    shape = np.asarray(shape_tuple, dtype=np.uint64)
    axis_data = np.ascontiguousarray(np.concatenate(axes))
    values = np.empty(shape_tuple, dtype=np.float64)
    stats = _Stats()
    start = perf_counter()
    _check(native, native.sff_compute_grid(axis_data, shape, ct.byref(config),
                                         values.ravel(), ct.byref(stats)))
    _metadata(metadata, stats, perf_counter() - start)
    metadata.update(total_states=int(values.size), grid_shape=list(shape_tuple))
    return SFFMetricResult(values, metadata)


def sffDistanceXY(points, *, theta_rel, v_h, delta_e, v_e, **parameters):
    """Signed distances for XY points (...,2) sharing four kinematic values.

    Builds the complete sampled unsafe union once. Useful for off-grid
    evaluation and temporal convergence checks. Returns an array in metres.
    lf/lr and rectangle sizes can be supplied as keyword parameters.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("points must have shape (..., 2)")
    eta = np.asarray([theta_rel, v_h, delta_e, v_e], dtype=np.float64)
    native = _library()
    config, _ = _config(native, parameters)
    values = np.empty(points.shape[:-1], dtype=np.float64)
    stats = _Stats()
    _check(native, native.sff_compute_points(points.ravel(), values.size, eta,
        ct.byref(config), values.ravel(), ct.byref(stats)))
    return values
