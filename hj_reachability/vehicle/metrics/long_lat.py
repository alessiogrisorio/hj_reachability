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


_FLOAT_FIELDS = (
    "horizon", "dt", "buffer", "ego_braking", "steering_rate",
    "delta_min", "delta_max", "lf", "lr", "ego_length", "ego_width",
    "human_length", "human_width",
)


class _Config(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in _FLOAT_FIELDS] + [
        ("threads", ct.c_int32), ("use_symmetry", ct.c_int32),
    ]


class _Stats(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in (
        "total_groups", "evaluated_groups", "time_samples", "ego_trajectories",
    )] + [("symmetry_used", ct.c_int32), ("threads_used", ct.c_int32)]


@dataclass
class LongLatMetricResult:
    terminal_values: np.ndarray
    longitudinal: np.ndarray
    lateral: np.ndarray
    parameters: dict


@lru_cache(maxsize=1)
def _library():
    directory = Path(__file__).resolve().parent
    source, target = directory / "long_lat.cpp", directory / "_long_lat.so"
    with (directory / "._long_lat_build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
            compiler = shlex.split(os.environ.get("CXX", "g++"))
            fd, temporary = tempfile.mkstemp(prefix="_long_lat_", suffix=".so", dir=directory)
            os.close(fd)
            try:
                command = compiler + ["-O3", "-march=native", "-DNDEBUG", "-std=c++17",
                                      "-fopenmp", "-fPIC", "-shared", str(source), "-o", temporary]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError("long_lat.cpp compilation failed:\n" + result.stderr)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    native = ct.CDLL(str(target))
    for name, restype in (("ll_abi_version", ct.c_int),
                          ("ll_config_size", ct.c_size_t),
                          ("ll_stats_size", ct.c_size_t)):
        function = getattr(native, name)
        function.argtypes, function.restype = [], restype
    if (native.ll_abi_version() != 1 or native.ll_config_size() != ct.sizeof(_Config)
            or native.ll_stats_size() != ct.sizeof(_Stats)):
        raise RuntimeError("Long/lat ABI mismatch; delete _long_lat.so and restart Python")
    native.ll_last_error.argtypes, native.ll_last_error.restype = [], ct.c_char_p
    native.ll_default_config.argtypes, native.ll_default_config.restype = [ct.POINTER(_Config)], None
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    sizes = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    native.ll_compute_grid.argtypes = [doubles, sizes, ct.POINTER(_Config),
                                      doubles, doubles, doubles, ct.POINTER(_Stats)]
    native.ll_compute_grid.restype = ct.c_int
    native.ll_compute_points.argtypes = [doubles, ct.c_uint64, doubles,
                                        ct.POINTER(_Config), doubles, doubles, doubles]
    native.ll_compute_points.restype = ct.c_int
    return native


def _configuration(native, parameters):
    config = _Config()
    native.ll_default_config(ct.byref(config))
    unknown = set(parameters) - set(_FLOAT_FIELDS) - {"threads", "use_symmetry"}
    if unknown:
        raise TypeError(f"Unknown long/lat parameters: {sorted(unknown)}")
    for name, value in parameters.items():
        if name == "threads":
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or not 0 <= value <= np.iinfo(np.int32).max):
                raise ValueError("threads must be an integer in [0, int32_max]")
        elif name == "use_symmetry":
            if value not in (True, False, 0, 1):
                raise ValueError("use_symmetry must be a boolean")
            value = int(value)
        else:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        setattr(config, name, value)
    return config


def _check(native, status):
    if status:
        raise RuntimeError(native.ll_last_error().decode("utf-8", errors="replace"))


def metricLongLat(grid, dynamics, ego=None, human=None, **parameters):
    """Compute sampled manoeuvre clearance in metres on the project's 6D grid.

    Bounds come from delta_E coordinates; braking and steering rates come
    from dynamics.control_space. Human speed/heading stay constant. No XY
    clipping or speed-axis clipping is applied during prediction. Ego stops
    at v=0 under braking. Rectangles are centred at each model reference point.
    """
    axes = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in grid.coordinate_vectors)
    if len(axes) != 6 or any(a.ndim != 1 or a.size == 0 for a in axes):
        raise ValueError("Expected six nonempty one-dimensional coordinate vectors")
    controlled = {"lf", "lr", "ego_braking", "steering_rate", "delta_min", "delta_max"}
    if controlled.intersection(parameters):
        raise TypeError("Dynamics and grid determine: " + ", ".join(sorted(controlled)))
    lower = np.asarray(dynamics.control_space.lo, dtype=np.float64)
    upper = np.asarray(dynamics.control_space.hi, dtype=np.float64)
    if lower.shape != (2,) or upper.shape != (2,):
        raise ValueError("Expected controls [steering_rate, acceleration]")
    if lower[0] >= 0 or upper[0] <= 0 or abs(lower[0] + upper[0]) > 1e-8:
        raise ValueError("Maximum steering rates must be positive and symmetric")
    parameters = dict(parameters, lf=float(dynamics.lf), lr=float(dynamics.lr),
                      ego_braking=float(-lower[1]), steering_rate=float(upper[0]),
                      delta_min=float(axes[4][0]), delta_max=float(axes[4][-1]))
    if ego is not None:
        parameters.update(ego_length=float(ego.length), ego_width=float(ego.width))
    if human is not None:
        parameters.update(human_length=float(human.length), human_width=float(human.width))
    native = _library()
    config = _configuration(native, parameters)
    shape_tuple = tuple(a.size for a in axes)
    shape = np.asarray(shape_tuple, dtype=np.uint64)
    axis_data = np.ascontiguousarray(np.concatenate(axes))
    combined, longitudinal, lateral = [np.empty(shape_tuple, dtype=np.float64) for _ in range(3)]
    stats = _Stats()
    start = perf_counter()
    _check(native, native.ll_compute_grid(axis_data, shape, ct.byref(config),
        combined.ravel(), longitudinal.ravel(), lateral.ravel(), ct.byref(stats)))
    metadata = {name: getattr(config, name) for name, _ in config._fields_}
    metadata.update({name: getattr(stats, name) for name, _ in stats._fields_})
    metadata.update(
        native_compute_seconds=perf_counter()-start,
        symmetry_used=bool(stats.symmetry_used), total_states=int(combined.size),
        grid_shape=list(shape_tuple), distance_units="m",
        terminal_value_definition="min(V0_long, V0_lat)",
        longitudinal_definition="min_t signed_clearance(maximum_braking) - buffer",
        lateral_definition="max(min_t signed_clearance(left), min_t signed_clearance(right)) - buffer",
        prediction="Human speed/heading constant; brake with frozen initial steering; steer at constant speed",
        geometry="Centred rectangles; Euclidean separation outside, negative minimum translation depth inside",
        time_discretization="Samples at 0, dt, ... and H; RK4 steering ramps, exact constant-steering arcs",
        symmetry="(y,theta,delta) -> (-y,-theta,-delta); axis matching tolerance 1e-6",
        model="Custom sampled maximum-manoeuvre spatial margin; not BTN/STN",
        sign_convention="V0 <= 0 unsafe", normalization="none",
    )
    return LongLatMetricResult(combined, longitudinal, lateral, metadata)


def longLatDistanceXY(points, *, theta_rel, v_h, delta_e, v_e, **parameters):
    """Evaluate XY points (...,2) sharing one kinematic tuple.

    Returns (combined, longitudinal, lateral). Steering bounds and dynamics
    may be supplied explicitly; defaults match LLConfig, not a grid object.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("points must have shape (..., 2)")
    eta = np.ascontiguousarray([theta_rel, v_h, delta_e, v_e], dtype=np.float64)
    native = _library()
    config = _configuration(native, parameters)
    outputs = [np.empty(points.shape[:-1], dtype=np.float64) for _ in range(3)]
    _check(native, native.ll_compute_points(points.ravel(), outputs[0].size, eta,
        ct.byref(config), *(a.ravel() for a in outputs)))
    return tuple(outputs)