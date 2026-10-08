"""Minimal NumPy/ctypes interface to time_eggert.cpp (Linux, C++17).

Place both files beside eggert.py. The native library is built on first use
and rebuilt when the C++ source changes. Set CXX to choose the compiler.
No change to eggert.py is needed.

In scripts/compute_brt.py:
    from hj_reachability.vehicle.metrics.time_eggert import metricTimeEggert
    # In compute_terminal_metric:
    if METRIC_NAME == "time_eggert":
        return metricTimeEggert(grid, dynamics, **parameters)

Add a "time_eggert" entry in METRIC_PARAMETERS, e.g.:
    "time_eggert": {"horizon": 3.0, "time_max": 3.0, "p_crit": 0.7,
                    "dt": 0.05, "search_dt": 0.05,
                    "time_tolerance": 0.01, "backend": "interpolated"}
Use METRIC_NAME = "time_eggert". The existing BRT code reads terminal_values.
To save diagnostics, add probability as P_H and first_contact_time to the
additional_metric_arrays dictionary, as for the original Eggert metric.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import ctypes as ct
import os
import shlex
import subprocess
import tempfile

import numpy as np


_DOUBLE_FIELDS = (
    "horizon", "time_max", "dt", "search_dt", "time_tolerance",
    "beta_d", "tau_d0", "escape_rate", "p_crit", "collision_tolerance",
    "contact_time_tolerance", "lf", "lr", "ego_length", "ego_width",
    "human_length", "human_width", "theta_period",
)


class _Config(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in _DOUBLE_FIELDS] + [
        ("threads", ct.c_int32), ("interpolate", ct.c_int32),
        ("use_symmetry", ct.c_int32),
    ]


@dataclass
class TimeEggertMetricResult:
    probability: np.ndarray
    terminal_values: np.ndarray
    first_contact_time: np.ndarray
    parameters: dict


@lru_cache(maxsize=1)
def _library():
    directory = Path(__file__).resolve().parent
    source = directory / "time_eggert.cpp"
    target = directory / "_time_eggert.so"
    if not source.is_file():
        raise FileNotFoundError(f"Missing native source: {source}")
    if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
        compiler = shlex.split(os.environ.get("CXX", "g++"))
        fd, temporary = tempfile.mkstemp(prefix="_time_eggert_", suffix=".so", dir=directory)
        os.close(fd)
        try:
            command = compiler + ["-O3", "-march=native", "-std=c++17", "-fopenmp",
                                  "-fPIC", "-shared", str(source), "-o", temporary]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("time_eggert.cpp compilation failed:\n" + result.stderr)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    native = ct.CDLL(str(target))
    native.te_abi_version.argtypes = []
    native.te_abi_version.restype = ct.c_int
    native.te_config_size.argtypes = []
    native.te_config_size.restype = ct.c_size_t
    if native.te_abi_version() != 1 or native.te_config_size() != ct.sizeof(_Config):
        raise RuntimeError("Native ABI mismatch; delete _time_eggert.so and restart Python")
    native.te_default_config.argtypes = [ct.POINTER(_Config)]
    native.te_default_config.restype = None
    native.te_last_error.argtypes = []
    native.te_last_error.restype = ct.c_char_p
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    shape = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    native.te_compute_grid.argtypes = [doubles, shape, ct.POINTER(_Config),
                                      doubles, doubles, doubles]
    native.te_compute_grid.restype = ct.c_int
    native.te_probability_states.argtypes = [doubles, ct.c_uint64, ct.POINTER(_Config),
                                            doubles, doubles]
    native.te_probability_states.restype = ct.c_int
    native.te_time_states.argtypes = [doubles, ct.c_uint64, ct.POINTER(_Config),
                                     doubles, doubles]
    native.te_time_states.restype = ct.c_int
    return native


def _config(native, *, backend="interpolated", **parameters):
    if backend not in ("interpolated", "direct"):
        raise ValueError("backend must be 'interpolated' or 'direct'")
    config = _Config()
    native.te_default_config(ct.byref(config))
    allowed = set(_DOUBLE_FIELDS) | {"threads", "use_symmetry"}
    unknown = set(parameters) - allowed
    if unknown:
        raise TypeError(f"Unknown parameters: {sorted(unknown)}")
    for name, value in parameters.items():
        if name == "threads":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("threads must be an integer >= 0")
            if not 0 <= int(value) <= np.iinfo(np.int32).max:
                raise ValueError("threads is outside the int32 range")
            value = int(value)
        elif name == "use_symmetry":
            if value not in (True, False, 0, 1):
                raise ValueError("use_symmetry must be a boolean")
            value = int(value)
        else:
            value = float(value)
        setattr(config, name, value)
    config.interpolate = int(backend == "interpolated")
    metadata = {name: getattr(config, name) for name, _ in config._fields_}
    metadata.update({
        "backend": backend,
        "time_definition": "First forward/backward entry into the opposite rolling-risk class",
        "contact_policy": "absorb_survivors_at_first_contact",
        "nominal_prediction": "Constant v_H, v_E, delta_E; zero human yaw rate",
        "time_units": "s",
        "regularization": "none",
        "out_of_grid_policy": "direct_probability_evaluation",
    })
    return config, metadata


def _check(native, status):
    if status:
        raise RuntimeError(native.te_last_error().decode("utf-8", errors="replace"))


def metricTimeEggert(grid, dynamics, ego=None, human=None, *,
                     backend="interpolated", **parameters):
    """Compute signed V0 and rolling Eggert probability on a six-dimensional grid.

    Defaults: horizon=time_max=3 s, p_crit=.7, dt=search_dt=.05 s,
    time_tolerance=.01 s, beta_d=2, tau_d0=.3 s, escape_rate=.1/s.
    threads=0 respects the OpenMP runtime / OMP_NUM_THREADS setting.
    Interpolated mode is approximate; direct mode is intended for validation.
    """
    axes = tuple(np.asarray(a, dtype=np.float64) for a in grid.coordinate_vectors)
    if len(axes) != 6 or any(a.ndim != 1 or a.size == 0 for a in axes):
        raise ValueError("Expected six nonempty one-dimensional coordinate vectors")
    parameters = dict(parameters)
    parameters.update(lf=float(dynamics.lf), lr=float(dynamics.lr))
    if ego is not None:
        parameters.update(ego_length=float(ego.length), ego_width=float(ego.width))
    if human is not None:
        parameters.update(human_length=float(human.length), human_width=float(human.width))
    native = _library()
    config, metadata = _config(native, backend=backend, **parameters)
    shape_tuple = tuple(a.size for a in axes)
    shape = np.asarray(shape_tuple, dtype=np.uint64)
    axis_data = np.ascontiguousarray(np.concatenate(axes))
    probability = np.empty(shape_tuple, dtype=np.float64)
    values = np.empty_like(probability)
    contact = np.empty_like(probability)
    _check(native, native.te_compute_grid(axis_data, shape, ct.byref(config),
        probability.ravel(), values.ravel(), contact.ravel()))
    metadata.update(total_states=int(probability.size), grid_shape=shape_tuple)
    return TimeEggertMetricResult(probability, values, contact, metadata)


def probabilityTimeEggert(states, **parameters):
    """Direct Eggert risk/contact time for states with final dimension six."""
    states = np.ascontiguousarray(states, dtype=np.float64)
    if states.ndim < 1 or states.shape[-1] != 6:
        raise ValueError("states must have shape (..., 6)")
    shape = states.shape[:-1]
    native = _library()
    config, _ = _config(native, **parameters)
    probability, contact = np.empty(shape), np.empty(shape)
    _check(native, native.te_probability_states(states.ravel(), probability.size,
        ct.byref(config), probability.ravel(), contact.ravel()))
    return probability, contact


def directTimeEggert(states, **parameters):
    """Direct signed times/risk for a sample, without grid interpolation."""
    states = np.ascontiguousarray(states, dtype=np.float64)
    if states.ndim < 1 or states.shape[-1] != 6:
        raise ValueError("states must have shape (..., 6)")
    shape = states.shape[:-1]
    native = _library()
    parameters = dict(parameters, backend="direct")
    config, _ = _config(native, **parameters)
    probability, values = np.empty(shape), np.empty(shape)
    _check(native, native.te_time_states(states.ravel(), probability.size,
        ct.byref(config), probability.ravel(), values.ravel()))
    return values, probability
