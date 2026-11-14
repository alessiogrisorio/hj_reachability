"""Pursuit-evasion simulation and thesis figures.

Run: python scripts/pursuit_th.py
Dependencies: numpy, scipy, matplotlib.
All relative paths are resolved from this script's directory.
The saved BRT slice and its stored gradients are used throughout the game.
Each selected metric generates a pursuit_<metric> folder.
frame.pdf is a multipage PDF, not a video. Existing output names are replaced.
"""
from __future__ import annotations

import gc
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, transforms
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker
from matplotlib.text import Text
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import gaussian_filter1d
try:
    from .plotting_common import METRIC_LABELS, format_colorbars
except ImportError:
    from plotting_common import METRIC_LABELS, format_colorbars

# ---------------- Configuration ----------------
DATA_DIRECTORY = Path("../results/brt")
# NPZ filenames without extension; one simulation and output folder per metric.
METRICS = (
    "euclidean",
    #"dce",
    #"btn",
    #"stn",
    #"long_lat",
    #"ttc",
    #"time_eggert",
    "rss",
    #"eggert",
)
OUTPUT_ROOT = Path(".")  # Creates pursuit_euclidean, pursuit_dce, etc. here.
SIMULATION_TIME = 5.0       # s; independent of the saved BRT horizon.
SIMULATION_FREQUENCY = 30.0  # Hz
ANIMATION_FREQUENCY = 2.0    # frames per simulated second; multipage frame.pdf.
FONT_SIZE = 19
# Solo rigenerazione dei plot salvati: nessuna nuova simulazione.
REPLOT_ONLY = True
REPLOT_FOLDERS = ("pursuit_euclidean_pos", "pursuit_rss_neg")
FIRST_NEGATIVE_BODY_ALPHA = 0.20

# Absolute states; headings and steering angle in radians.
EGO_INITIAL_STATE = np.array([0.0, 0.0, 0.0, 0.0, 7.0])  # x,y,psi,delta,v
HUMAN_INITIAL_STATE = np.array([-6.0, 4.0, 0.0, 6.0])     # x,y,psi,v

EGO_LENGTH, EGO_WIDTH = 4.68, 2.20
HUMAN_LENGTH, HUMAN_WIDTH = 4.28, 1.80
EGO_COLOR, HUMAN_COLOR = "tab:blue", "tab:orange"
TIME_COLORMAP = "viridis"
DPI = 200
RECOVERY_ALPHA = 0.14
RECOVERY_POSITION_GAIN = 0.5
RECOVERY_ACCELERATION_WEIGHT = 1.0
RECOVERY_YAW_WEIGHT = 1.0
RECOVERY_LOW_SPEED = 0.2
RECOVERY_HEADING_GAIN = 1.0
# ------------------------------------------------
BASE = Path(__file__).resolve().parent
NAMES = ("x_rel", "y_rel", "theta_rel", "v_H", "delta_E", "v_E")
CORNERS = np.array(list(product((0, 1), repeat=6)), dtype=int)


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else BASE / path


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        axes = tuple(np.asarray(data[name], dtype=float) for name in NAMES)
        if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
               or not np.all(np.diff(a) > 0) for a in axes):
            raise ValueError("NPZ contains invalid grid axes.")
        if "state_names" in data and tuple(data["state_names"].tolist()) != NAMES:
            raise ValueError("Incompatible state order.")
        brt, v0, gradient = data["BRT"], data["V0"], data["gradients"]
        shape = tuple(len(a) for a in axes)
        if brt.shape != shape or v0.shape != shape or gradient.shape != shape + (6,):
            raise ValueError("Invalid field or gradient shape.")
        # Check slabs rather than allocate a full-size finite mask for gradients.
        for k in range(shape[0]):
            if not all(np.isfinite(a[k]).all() for a in (brt, v0, gradient)):
                raise ValueError("Non-finite input fields or gradients.")
        lo = np.asarray(data["grid_lo"], dtype=float)
        hi = np.asarray(data["grid_hi"], dtype=float)
        periodic = tuple(int(i) for i in data["periodic_dims"])
        if lo.shape != (6,) or hi.shape != (6,) or not np.all(hi > lo):
            raise ValueError("Invalid grid bounds.")
        if periodic != (2,):
            raise ValueError("This vehicle simulation expects periodic_dims=(2,).")
        for dim, axis in enumerate(axes):
            if not np.isclose(axis[0], lo[dim]):
                raise ValueError(f"Axis {NAMES[dim]} does not match grid_lo.")
            if dim == 2:
                if not np.isclose(hi[dim] - lo[dim], 2 * np.pi) or axis[-1] >= hi[dim]:
                    raise ValueError("Heading must cover an endpoint-excluded 2*pi period.")
            elif not np.isclose(axis[-1], hi[dim]):
                raise ValueError(f"Axis {NAMES[dim]} does not match grid_hi.")
        metadata = json.loads(str(data["metadata_json"].item()))
        parameters = metadata["dynamics"]["parameters"]
        target_time = float(data["target_time"])
    if not np.isfinite(target_time):
        raise ValueError("Invalid target_time.")
    for mode, expected in (("control_mode", "max"), ("disturbance_mode", "min")):
        if parameters.get(mode, expected) != expected:
            raise ValueError(f"Expected {mode}={expected}.")
    if any(parameters.get(k) is not None for k in ("control_space", "disturbance_space")):
        raise ValueError("Custom input spaces are not supported by this script.")
    lf, lr = float(parameters["lf"]), float(parameters["lr"])
    if lf <= 0 or lr <= 0:
        raise ValueError("Vehicle lengths lf and lr must be positive.")
    # Defaults match RelativeVehicle6D; saved parameters override them.
    steering = parameters.get("ego_max_steering_rate", 0.087)
    yaw = parameters.get("human_max_yaw_rate", 1.0)
    u_lo = np.array([-steering, parameters.get("ego_min_acceleration", -7.0)], float)
    u_hi = np.array([steering, parameters.get("ego_max_acceleration", 2.5)], float)
    d_lo = np.array([-yaw, parameters.get("human_min_acceleration", -7.0)], float)
    d_hi = np.array([yaw, parameters.get("human_max_acceleration", 2.5)], float)
    if not all(np.isfinite(a).all() for a in (u_lo, u_hi, d_lo, d_hi)):
        raise ValueError("Non-finite input bounds.")
    if np.any(u_lo > u_hi) or np.any(d_lo > d_hi):
        raise ValueError("Invalid input bounds.")
    return dict(axes=axes, brt=brt, v0=v0, gradient=gradient, lo=lo, hi=hi,
                lf=lf, lr=lr, u_lo=u_lo, u_hi=u_hi, d_lo=d_lo, d_hi=d_hi,
                target_time=target_time, metadata=metadata)


def relative_state(z, model):
    dx, dy = z[5] - z[0], z[6] - z[1]
    c, s = np.cos(z[2]), np.sin(z[2])
    theta = (z[7] - z[2] - model["lo"][2]) % (2 * np.pi) + model["lo"][2]
    return np.array([c * dx + s * dy, -s * dx + c * dy,
                     theta, z[8], z[3], z[4]])


def inside_grid(state, model):
    dims = [0, 1, 3, 4, 5]
    return bool(np.all(state[dims] >= model["lo"][dims])
                and np.all(state[dims] <= model["hi"][dims]))


def interpolate(model, state):
    """Multilinear interpolation, including the periodic heading seam."""
    lower, upper, weights = [], [], []
    for dim, axis in enumerate(model["axes"]):
        value = state[dim]
        if dim == 2:
            value = (value - model["lo"][2]) % (2 * np.pi) + model["lo"][2]
        i = int(np.clip(np.searchsorted(axis, value, side="right") - 1, 0, len(axis) - 1))
        if dim == 2 and i == len(axis) - 1:
            j, right = 0, axis[0] + 2 * np.pi
        else:
            i = min(i, len(axis) - 2)
            j, right = i + 1, axis[i + 1]
        lower.append(i)
        upper.append(j)
        weights.append((value - axis[i]) / (right - axis[i]))
    indices = np.where(CORNERS, upper, lower)
    w = np.prod(np.where(CORNERS, weights, 1 - np.asarray(weights)), axis=1)
    index = tuple(indices[:, dim] for dim in range(6))
    value = float(w @ model["brt"][index])
    terminal = float(w @ model["v0"][index])
    gradient = w @ model["gradient"][index]
    return value, terminal, gradient


def joint_rate(z, inputs, model):
    steer, accel_e, yaw_h, accel_h = inputs
    beta = np.arctan(model["lr"] / (model["lf"] + model["lr"]) * np.tan(z[3]))
    omega_e = z[4] * np.cos(beta) / (model["lf"] + model["lr"]) * np.tan(z[3])
    return np.array([z[4] * np.cos(z[2] + beta), z[4] * np.sin(z[2] + beta),
                     omega_e, steer, accel_e,
                     z[8] * np.cos(z[7]), z[8] * np.sin(z[7]), yaw_h, accel_h])


def relative_rate(state, inputs, model):
    x, y, theta, vh, delta, ve = state
    steer, accel_e, yaw_h, accel_h = inputs
    beta = np.arctan(model["lr"] / (model["lf"] + model["lr"]) * np.tan(delta))
    omega_e = ve * np.cos(beta) / (model["lf"] + model["lr"]) * np.tan(delta)
    return np.array([vh * np.cos(theta) - ve * np.cos(beta) + y * omega_e,
                     vh * np.sin(theta) - ve * np.sin(beta) - x * omega_e,
                     yaw_h - omega_e, accel_h, steer, accel_e])


def restrict_state_rates(inputs, state, model, dt):
    inputs = inputs.copy()
    for command, dim in ((0, 4), (1, 5), (3, 3)):
        inputs[command] = np.clip(inputs[command],
                                  (model["lo"][dim] - state[dim]) / dt,
                                  (model["hi"][dim] - state[dim]) / dt)
    return inputs


def recovery_inputs(z, state, reference_speed, model, dt):
    u = np.clip([-state[4] / dt, (reference_speed - state[5]) / dt],
                model["u_lo"], model["u_hi"])
    beta = np.arctan(model["lr"] / (model["lf"] + model["lr"]) * np.tan(state[4]))
    ego_velocity = state[5] * np.array([np.cos(z[2] + beta), np.sin(z[2] + beta)])
    desired = ego_velocity + RECOVERY_POSITION_GAIN * (z[:2] - z[5:7])
    heading = np.array([np.cos(z[7]), np.sin(z[7])])
    normal = np.array([-np.sin(z[7]), np.cos(z[7])])
    error = state[3] * heading - desired
    accel = -error @ heading / RECOVERY_ACCELERATION_WEIGHT
    yaw = -state[3] * (error @ normal) / RECOVERY_YAW_WEIGHT
    if abs(state[3]) < RECOVERY_LOW_SPEED:
        desired_heading = np.arctan2(desired[1], desired[0])
        yaw = (RECOVERY_HEADING_GAIN * np.arctan2(
            np.sin(desired_heading - z[7]), np.cos(desired_heading - z[7]))
            if np.linalg.norm(desired) > 1e-9 else 0.0)
    d = np.clip([yaw, accel], model["d_lo"], model["d_hi"])
    return restrict_state_rates(np.r_[u, d], state, model, dt)


def simulate(model):
    settings = [SIMULATION_TIME, SIMULATION_FREQUENCY, ANIMATION_FREQUENCY, FONT_SIZE]
    if not np.isfinite(settings).all() or min(settings) <= 0:
        raise ValueError("Duration, frequencies and font size must be finite and positive.")
    ego, human = np.asarray(EGO_INITIAL_STATE, float), np.asarray(HUMAN_INITIAL_STATE, float)
    if ego.shape != (5,) or human.shape != (4,) or not np.isfinite(np.r_[ego, human]).all():
        raise ValueError("Invalid absolute initial states.")
    for value, dim in ((ego[3], 4), (ego[4], 5), (human[3], 3)):
        if not model["lo"][dim] <= value <= model["hi"][dim]:
            raise ValueError(f"Initial {NAMES[dim]} lies outside the grid.")
    times = np.arange(int(np.floor(SIMULATION_TIME * SIMULATION_FREQUENCY)) + 1) / SIMULATION_FREQUENCY
    if not np.isclose(times[-1], SIMULATION_TIME, atol=1e-12, rtol=0):
        times = np.r_[times, SIMULATION_TIME]
    else:
        times[-1] = SIMULATION_TIME
    n = len(times)
    states, relative = np.empty((n, 9)), np.empty((n, 6))
    states[0] = np.r_[ego, human]
    values, v0, hamiltonian = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    h_unrestricted = np.full(n, np.nan)
    inputs, recovery = np.empty((n, 4)), np.zeros(n, bool)
    reference_speed = None
    last_speed = ego[4]
    for k, time in enumerate(times):
        z = states[k]
        state = relative_state(z, model)
        relative[k] = state
        dt = times[k + 1] - time if k < n - 1 else 1 / SIMULATION_FREQUENCY
        inside = inside_grid(state, model)
        recovery[k] = not inside
        if inside:
            reference_speed = None
            last_speed = state[5]
            values[k], v0[k], gradient = interpolate(model, state)
            # Same endpoint selection as hj.sets.Box.extreme_point, including ties.
            u = np.where(gradient[[4, 5]] < 0, model["u_lo"], model["u_hi"])
            d = np.where(-gradient[[2, 3]] < 0, model["d_lo"], model["d_hi"])
            optimum = np.r_[u, d]
            h_unrestricted[k] = gradient @ relative_rate(state, optimum, model)
            inputs[k] = restrict_state_rates(optimum, state, model, dt)
            # Hamiltonian of the commands actually applied after state-bound limiting.
            hamiltonian[k] = gradient @ relative_rate(state, inputs[k], model)
        else:
            if reference_speed is None:
                reference_speed = last_speed
                print(f"Recovery activated at t={time:.4f} s")
            inputs[k] = recovery_inputs(z, state, reference_speed, model, dt)
        if k < n - 1:
            states[k + 1] = z + dt * joint_rate(z, inputs[k], model)
            for absolute_dim, dim in ((3, 4), (4, 5), (8, 3)):
                states[k + 1, absolute_dim] = np.clip(
                    states[k + 1, absolute_dim], model["lo"][dim], model["hi"][dim])
    derivative = np.full(n, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    for segment in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
        if len(segment) >= 2:
            derivative[segment] = np.gradient(values[segment], times[segment],
                                             edge_order=2 if len(segment) >= 3 else 1)
    return dict(time=times, states=states, relative_states=relative, brt=values, v0=v0,
                hamiltonian=hamiltonian, hamiltonian_unrestricted=h_unrestricted,
                dv_dt=derivative, commands=inputs, recovery=recovery)


def recovery_intervals(result):
    mask, t = result["recovery"], result["time"]
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return [(float(t[a]), float(t[b + 1] if b + 1 < len(t) else t[b]))
            for a, b in zip(starts, ends)]


def configure_style():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
                         "axes.titlesize": FONT_SIZE + 1, "legend.fontsize": FONT_SIZE - 1,
                         "xtick.labelsize": FONT_SIZE - 1, "ytick.labelsize": FONT_SIZE - 1,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def title(metric):
    return METRIC_LABELS.get(metric, metric)


def save(fig, folder, name):
    format_colorbars(fig)
    fig.savefig(folder / name, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def shade_recovery(ax, result):
    for start, end in recovery_intervals(result):
        ax.axvspan(start, end, color="red", alpha=RECOVERY_ALPHA, linewidth=0, zorder=0)


def decorate_time_axis(ax, result, ylabel):
    shade_recovery(ax, result)
    ax.set(xlabel="t [s]", ylabel=ylabel, xlim=(0, result["time"][-1]))
    ax.grid(True, alpha=0.18)


def body(ax, x, y, heading, length, width, color):
    patch = Rectangle((-length / 2, -width / 2), length, width,
                      facecolor=color, edgecolor="black", linewidth=0.7, zorder=5)
    patch.set_transform(transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData)
    ax.add_patch(patch)
    return patch


def spatial_limits(result):
    points = np.concatenate((result["states"][:, :2], result["states"][:, 5:7]))
    margin = max(EGO_LENGTH, HUMAN_LENGTH) * 0.7
    return ((points[:, 0].min() - margin, points[:, 0].max() + margin),
            (points[:, 1].min() - margin, points[:, 1].max() + margin))


def vehicle_handles():
    return [Patch(facecolor=EGO_COLOR, edgecolor="black", label="Ego"),
            Patch(facecolor=HUMAN_COLOR, edgecolor="black", label="Human")]


def plot_trajectory(result, metric, folder):
    t, states = result["time"], result["states"]
    fig = plt.figure(figsize=(9.2, 6.0), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 0.035), wspace=0.08)
    ax, cax = [fig.add_subplot(grid[0, i]) for i in range(2)]
    norm = colors.Normalize(0, t[-1])
    for offset in (0, 5):
        points = states[:, offset:offset + 2]
        segments = np.stack((points[:-1], points[1:]), axis=1)
        line = LineCollection(segments, cmap=TIME_COLORMAP, norm=norm, linewidth=2.2)
        line.set_array((t[:-1] + t[1:]) / 2)
        ax.add_collection(line)
    z = states[0]
    body(ax, z[0], z[1], z[2], EGO_LENGTH, EGO_WIDTH, EGO_COLOR)
    body(ax, z[5], z[6], z[7], HUMAN_LENGTH, HUMAN_WIDTH, HUMAN_COLOR)
    handles = vehicle_handles()
    negative = np.flatnonzero(np.isfinite(result["v0"]) & (result["v0"] < 0))
    if len(negative):
        z = states[negative[0]]
        for offset, length, width, color in (
            (0, EGO_LENGTH, EGO_WIDTH, EGO_COLOR),
            (5, HUMAN_LENGTH, HUMAN_WIDTH, HUMAN_COLOR),
        ):
            silhouette = body(ax, z[offset], z[offset + 1], z[offset + 2],
                              length, width, color)
            silhouette.set_alpha(FIRST_NEGATIVE_BODY_ALPHA)
            silhouette.set_zorder(6)
        ax.plot([z[0], z[5]], [z[1], z[6]], linestyle="none", marker="*",
                color="red", markersize=12, markeredgecolor="darkred", zorder=8)
        handles.append(Line2D([], [], marker="*", linestyle="none", color="red",
                              markersize=10, label=r"First $V_0<0$"))
    ax.set(xlabel="x [m]", ylabel="y [m]", title=title(metric))
    ax.set_xlim(spatial_limits(result)[0]); ax.set_ylim(spatial_limits(result)[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.legend(handles=handles, loc="best")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=TIME_COLORMAP), cax=cax)
    cax.set_title("t", pad=10)
    bar.set_label("[s]")
    save(fig, folder, "trajectory.pdf")


def plot_values(result, metric, folder):
    for filename, fields in (("brt.pdf", ("brt",)), ("v0.pdf", ("v0",)),
                             ("values.pdf", ("brt", "v0"))):
        fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
        for field in fields:
            ax.plot(result["time"], result[field], color=EGO_COLOR if field == "brt" else HUMAN_COLOR,
                    label="BRT" if field == "brt" else r"$V_0$", linewidth=1.5)
        ax.axhline(0, color="0.3", linewidth=0.8, linestyle=":")
        decorate_time_axis(ax, result, "Value")
        ax.set_title(title(metric))
        ax.legend()
        save(fig, folder, filename)


def plot_hamiltonian(result, metric, folder):
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    ax.plot(result["time"], result["hamiltonian"], color=EGO_COLOR,
            label=r"$H^*=\nabla V^\top f_R$", linewidth=1.5)
    ax.plot(result["time"], result["dv_dt"], color=HUMAN_COLOR,
            label=r"$\mathrm{d}V/\mathrm{d}t$",
            linestyle="--", linewidth=1.5)
    ax.axhline(0, color="0.3", linestyle=":", linewidth=0.8)
    decorate_time_axis(ax, result, "H")
    ax.set_title(title(metric))
    ax.legend()
    save(fig, folder, "hamiltonian.pdf")


def plot_commands(result, metric, folder, model):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout="constrained")
    labels = (r"$\dot{\delta}_E$ [rad/s]", r"$a_E$ [m/s$^2$]",
              r"$\omega_H$ [rad/s]", r"$a_H$ [m/s$^2$]")
    lows = np.r_[model["u_lo"], model["d_lo"]]
    highs = np.r_[model["u_hi"], model["d_hi"]]
    for j, ax in enumerate(axes.flat):
        color = EGO_COLOR if j < 2 else HUMAN_COLOR
        ax.step(result["time"], result["commands"][:, j], where="post", color=color, linewidth=1.2)
        decorate_time_axis(ax, result, labels[j])
        for bound in (lows[j], highs[j]):
            ax.axhline(bound, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_ylim(lows[j] - 0.08 * (highs[j] - lows[j]),
                    highs[j] + 0.08 * (highs[j] - lows[j]))
    fig.suptitle(title(metric))
    save(fig, folder, "commands.pdf")


def frame_times():
    times = np.arange(int(np.floor(SIMULATION_TIME * ANIMATION_FREQUENCY)) + 1) / ANIMATION_FREQUENCY
    if not np.isclose(times[-1], SIMULATION_TIME, atol=1e-12, rtol=0):
        times = np.r_[times, SIMULATION_TIME]
    else:
        times[-1] = SIMULATION_TIME
    return times


def export_frames(result, metric, folder, model):
    """Interpolate states to requested frame times; evaluate values at those states."""
    limits = spatial_limits(result)
    requested = frame_times()
    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    with PdfPages(folder / "frame.pdf") as pdf:
        pdf.infodict()["Title"] = title(metric)
        for time in requested:
            ax.clear()
            z = np.array([np.interp(time, result["time"], result["states"][:, j]) for j in range(9)])
            state = relative_state(z, model)
            inside = inside_grid(state, model)
            if inside:
                value, terminal, _ = interpolate(model, state)
                info = f"t={time:.2f} s    V0={terminal:.4g}    brt={value:.4g}"
            else:
                info = f"t={time:.2f} s    V0=N/A    brt=N/A    recovery"
            for offset, color in ((0, EGO_COLOR), (5, HUMAN_COLOR)):
                ax.plot(result["states"][:, offset], result["states"][:, offset + 1],
                        color="0.85", linewidth=0.8, zorder=1)
                before = result["time"] < time
                ax.plot(np.r_[result["states"][before, offset], z[offset]],
                        np.r_[result["states"][before, offset + 1], z[offset + 1]],
                        color=color, linewidth=1.3, zorder=2)
            body(ax, z[0], z[1], z[2], EGO_LENGTH, EGO_WIDTH, EGO_COLOR)
            body(ax, z[5], z[6], z[7], HUMAN_LENGTH, HUMAN_WIDTH, HUMAN_COLOR)
            ax.set(xlabel="x [m]", ylabel="y [m]", xlim=limits[0], ylim=limits[1])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.15)
            ax.legend(handles=vehicle_handles(), loc="best")
            ax.set_title(title(metric) + "\n" + info)
            pdf.savefig(fig, dpi=DPI)
    plt.close(fig)
    print(f"frame.pdf: {len(requested)} pages at {ANIMATION_FREQUENCY:g} frames/s")


def run_metric(metric):
    path = resolve(DATA_DIRECTORY) / f"{metric}.npz"
    folder = resolve(OUTPUT_ROOT) / f"pursuit_{metric}"
    model = load_model(path)
    print(f"metric={metric}; saved BRT horizon={model['target_time']:g} s")
    result = simulate(model)
    folder.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_trajectory(result, metric, folder)
    plot_values(result, metric, folder)
    plot_hamiltonian(result, metric, folder)
    plot_commands(result, metric, folder, model)
    export_frames(result, metric, folder, model)
    configuration = dict(input_file=str(path.resolve()), metric=metric,
                         simulation_time=SIMULATION_TIME,
                         simulation_frequency=SIMULATION_FREQUENCY,
                         animation_frequency=ANIMATION_FREQUENCY,
                         font_size=FONT_SIZE, saved_target_time=model["target_time"],
                         ego_initial_state=np.asarray(EGO_INITIAL_STATE).tolist(),
                         human_initial_state=np.asarray(HUMAN_INITIAL_STATE).tolist(),
                         recovery_position_gain=RECOVERY_POSITION_GAIN,
                         recovery_acceleration_weight=RECOVERY_ACCELERATION_WEIGHT,
                         recovery_yaw_weight=RECOVERY_YAW_WEIGHT,
                         recovery_low_speed=RECOVERY_LOW_SPEED,
                         recovery_heading_gain=RECOVERY_HEADING_GAIN,
                         first_negative_definition="first sampled finite V0 < 0; not a geometric collision test",
                         hamiltonian_definition="gradient dot relative dynamics with applied clipped inputs")
    np.savez_compressed(folder / "simulation.npz", **result,
                        configuration_json=np.asarray(json.dumps(configuration)),
                        source_metadata_json=np.asarray(json.dumps(model["metadata"])))
    negative = np.flatnonzero(np.isfinite(result["v0"]) & (result["v0"] < 0))
    print(f"First V0 < 0: t={result['time'][negative[0]]:.6g} s" if len(negative)
          else "No V0 < 0 detected during HJ intervals.")
    print(f"Recovery intervals: {recovery_intervals(result)}")
    print(f"Files saved in: {folder.resolve()}")
    return folder


def replot_saved(folder):
    folder = resolve(folder)
    with np.load(folder / "simulation.npz", allow_pickle=False) as data:
        result = {name: data[name] for name in (
            "time", "states", "v0", "brt", "recovery",
        )}
        config = json.loads(str(data["configuration_json"].item()))
    for name, key in (("EGO_LENGTH", "ego_length"), ("EGO_WIDTH", "ego_width"),
                      ("HUMAN_LENGTH", "human_length"), ("HUMAN_WIDTH", "human_width")):
        if key in config and not np.isclose(config[key], globals()[name]):
            raise ValueError(f"Geometria salvata differente: {key}")
    configure_style()
    metric = config["metric"]
    plot_trajectory(result, metric, folder)
    plot_values(result, metric, folder)
    print(f"Rigenerati trajectory.pdf, brt.pdf, v0.pdf, values.pdf in {folder}")
    return folder


def main():
    if REPLOT_ONLY:
        return [replot_saved(folder) for folder in REPLOT_FOLDERS]
    if isinstance(METRICS, str) or not METRICS:
        raise ValueError('METRICS must be a nonempty sequence, e.g. ("euclidean", "dce").')
    if any(not isinstance(metric, str) or not metric
           or Path(metric).name != metric or metric in (".", "..")
           or metric.endswith(".npz") or "\\" in metric
           for metric in METRICS):
        raise ValueError("Each metric must be an NPZ filename without extension or directories.")
    if len(set(METRICS)) != len(METRICS):
        raise ValueError("METRICS contains duplicate names.")
    missing = [resolve(DATA_DIRECTORY) / f"{metric}.npz" for metric in METRICS
               if not (resolve(DATA_DIRECTORY) / f"{metric}.npz").is_file()]
    if missing:
        raise FileNotFoundError("Missing NPZ files:\n" + "\n".join(map(str, missing)))
    folders = []
    for number, metric in enumerate(METRICS, start=1):
        print(f"\n[{number}/{len(METRICS)}] {metric}")
        folders.append(run_metric(metric))
        # Release the six-dimensional fields before loading the next metric.
        gc.collect()
    print(f"\nCompleted {len(folders)} simulations.")
    return folders


if __name__ == "__main__":
    main()