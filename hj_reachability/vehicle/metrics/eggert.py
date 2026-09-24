from dataclasses import dataclass

import numpy as np

from ..geometry.rectangles import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    VehicleGeometry,
)
from ..geometry.collision_sat import collision_margin_sat


@dataclass
class EggertMetricResult:
    probability: np.ndarray
    terminal_values: np.ndarray
    first_contact_time: np.ndarray
    parameters: dict


def rectangle_distance(x, y, theta, ego, human):
    """Geometric distance between rectangles, vectorized over poses."""

    x, y, theta = np.broadcast_arrays(x, y, theta)
    centres = np.stack((x, y), axis=-1)

    cosine = np.cos(theta)
    sine = np.sin(theta)

    rotation = np.empty(x.shape + (2, 2))
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 1] = -sine
    rotation[..., 1, 0] = sine
    rotation[..., 1, 1] = cosine

    # Human vertices in the ego frame.
    human_vertices = (
        np.einsum("...ij,kj->...ki", rotation, human.vertices())
        + centres[..., None, :]
    )

    ego_half_size = np.array([ego.half_length, ego.half_width])
    offsets = np.maximum(np.abs(human_vertices) - ego_half_size, 0.0)
    human_to_ego = np.linalg.norm(offsets, axis=-1).min(axis=-1)

    # Ego vertices in the human frame.
    relative_vertices = ego.vertices() - centres[..., None, :]
    ego_vertices = np.einsum(
        "...ji,...kj->...ki", rotation, relative_vertices
    )

    human_half_size = np.array([human.half_length, human.half_width])
    offsets = np.maximum(np.abs(ego_vertices) - human_half_size, 0.0)
    ego_to_human = np.linalg.norm(offsets, axis=-1).min(axis=-1)

    distance = np.minimum(human_to_ego, ego_to_human)

    # Required for intersecting edges without enclosed vertices.
    margin = collision_margin_sat(x, y, theta, ego=ego, human=human)
    return np.where(margin <= 0.0, 0.0, distance)


def _reflection_indices(axis, *, angular=False, tolerance=1e-6):
    """Return the index map for axis -> -axis, or None if unavailable."""

    axis = np.asarray(axis, dtype=float)

    # Row i contains distances between -axis[i] and each grid node.
    difference = axis[:, None] + axis[None, :]
    if angular:
        difference = np.arctan2(
            np.sin(difference), np.cos(difference)
        )

    indices = np.argmin(np.abs(difference), axis=1)
    errors = np.abs(difference[np.arange(axis.size), indices])

    # Require a one-to-one, involutive reflection.
    if (
        np.any(errors > tolerance)
        or np.unique(indices).size != axis.size
        or not np.array_equal(indices[indices], np.arange(axis.size))
    ):
        return None

    return indices


def _evaluate_motion_group(
    x0,
    y0,
    theta0,
    v_h,
    delta_e,
    v_e,
    *,
    lf,
    lr,
    ego,
    human,
    horizon,
    dt,
    beta_d,
    tau_d0,
    escape_rate,
    collision_tolerance,
    contact_time_tolerance,
):
    """Evaluate XY positions sharing theta0, v_h, delta_e and v_e."""

    size = x0.size
    probability = np.zeros(size)
    survival = np.ones(size)
    contact_time = np.full(size, np.inf)

    initial_margin = collision_margin_sat(
        x0, y0, theta0, ego=ego, human=human
    )
    initial_contact = initial_margin <= collision_tolerance

    probability[initial_contact] = 1.0
    survival[initial_contact] = 0.0
    contact_time[initial_contact] = 0.0

    active = np.flatnonzero(~initial_contact)
    if active.size == 0:
        return probability, contact_time

    # These quantities are shared by the entire group.
    wheelbase = lf + lr
    beta = np.arctan((lr / wheelbase) * np.tan(delta_e))
    omega = v_e * np.cos(beta) * np.tan(delta_e) / wheelbase

    cos_theta0 = np.cos(theta0)
    sin_theta0 = np.sin(theta0)

    half_length = ego.half_length + human.half_length
    half_width = ego.half_width + human.half_width

    # Exact specialization for theta = 0 mod 2*pi and delta = 0.
    aligned = bool(
        delta_e == 0.0
        and abs(np.arctan2(sin_theta0, cos_theta0)) < 1e-12
    )

    def pose(indices, time):
        """Relative pose; scalar time shares trigonometry across XY."""

        time = np.asarray(time)
        angle = omega * time

        # Stable analytical integration, also valid for omega = 0.
        travel = v_e * time * np.sinc(angle / (2.0 * np.pi))
        ego_x = travel * np.cos(beta + 0.5 * angle)
        ego_y = travel * np.sin(beta + 0.5 * angle)

        dx = x0[indices] + v_h * time * cos_theta0 - ego_x
        dy = y0[indices] + v_h * time * sin_theta0 - ego_y

        cosine = np.cos(angle)
        sine = np.sin(angle)

        return (
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            theta0 - angle,
        )

    # -------------------------------------------------------------
    # Analytic first-contact time for the aligned straight case
    # -------------------------------------------------------------

    scheduled_contact = np.full(size, np.inf)

    if aligned:
        relative_speed = v_h - v_e

        if relative_speed != 0.0:
            # SAT <= tolerance for axis-aligned rectangles.
            longitudinal_limit = half_length + collision_tolerance
            lateral_limit = half_width + collision_tolerance

            crossing_a = (-longitudinal_limit - x0) / relative_speed
            crossing_b = (longitudinal_limit - x0) / relative_speed

            entry = np.maximum(np.minimum(crossing_a, crossing_b), 0.0)
            exit_time = np.maximum(crossing_a, crossing_b)

            will_contact = (
                ~initial_contact
                & (np.abs(y0) <= lateral_limit)
                & (exit_time >= entry)
                & (entry <= horizon)
            )
            scheduled_contact[will_contact] = entry[will_contact]

    number_of_steps = int(np.ceil(horizon / dt))
    bisection_steps = max(
        0, int(np.ceil(np.log2(dt / contact_time_tolerance)))
    )

    # -------------------------------------------------------------
    # Time integration on active states only
    # -------------------------------------------------------------

    for step in range(number_of_steps):
        if active.size == 0:
            break

        start = step * dt
        end = min((step + 1) * dt, horizon)
        middle = 0.5 * (start + end)

        if aligned:
            # No SAT, vertex transforms or bisections are needed.
            hit = scheduled_contact[active] <= end
            stop = np.minimum(scheduled_contact[active], end)
            interval = np.maximum(stop - start, 0.0)
            evaluation_time = start + 0.5 * interval

            x = x0[active] + (v_h - v_e) * evaluation_time

            gap_x = np.maximum(np.abs(x) - half_length, 0.0)
            gap_y = np.maximum(np.abs(y0[active]) - half_width, 0.0)
            distance = np.hypot(gap_x, gap_y)

        else:
            middle_pose = pose(active, middle)
            middle_contact = (
                collision_margin_sat(
                    *middle_pose, ego=ego, human=human
                ) <= collision_tolerance
            )

            # Endpoint checks are unnecessary if midpoint contact exists.
            end_contact = np.zeros(active.size, dtype=bool)
            remaining = np.flatnonzero(~middle_contact)

            if remaining.size:
                end_pose = pose(active[remaining], end)
                end_contact[remaining] = (
                    collision_margin_sat(
                        *end_pose, ego=ego, human=human
                    ) <= collision_tolerance
                )

            hit = middle_contact | end_contact
            hit_positions = np.flatnonzero(hit)
            stop = np.full(active.size, end)

            # Refine only the states with detected contact.
            if hit_positions.size:
                hit_indices = active[hit_positions]

                lower = np.where(
                    middle_contact[hit_positions], start, middle
                )
                upper = np.where(
                    middle_contact[hit_positions], middle, end
                )

                for _ in range(bisection_steps):
                    trial = 0.5 * (lower + upper)
                    trial_pose = pose(hit_indices, trial)
                    trial_contact = (
                        collision_margin_sat(
                            *trial_pose, ego=ego, human=human
                        ) <= collision_tolerance
                    )

                    upper = np.where(trial_contact, trial, upper)
                    lower = np.where(trial_contact, lower, trial)

                stop[hit_positions] = upper

            interval = np.maximum(stop - start, 0.0)

            # Reuse the midpoint pose for intervals without contact.
            x = np.array(middle_pose[0], copy=True)
            y = np.array(middle_pose[1], copy=True)
            theta = np.full(active.size, float(middle_pose[2]))

            # Only contacting states have a shortened interval.
            if hit_positions.size:
                evaluation_time = start + 0.5 * interval[hit_positions]
                shortened_pose = pose(
                    active[hit_positions], evaluation_time
                )
                x[hit_positions] = shortened_pose[0]
                y[hit_positions] = shortened_pose[1]
                theta[hit_positions] = shortened_pose[2]

            distance = rectangle_distance(
                x, y, theta, ego=ego, human=human
            )

        # Exact interval update for constant midpoint rates.
        critical_rate = np.exp(-beta_d * distance) / tau_d0
        total_rate = critical_rate + escape_rate

        event_fraction = -np.expm1(-total_rate * interval)
        critical_share = np.divide(
            critical_rate,
            total_rate,
            out=np.zeros_like(total_rate),
            where=total_rate > 0.0,
        )

        probability[active] += (
            survival[active] * critical_share * event_fraction
        )
        survival[active] *= np.exp(-total_rate * interval)

        # Absorb surviving probability at first contact.
        hit_indices = active[hit]
        probability[hit_indices] += survival[hit_indices]
        survival[hit_indices] = 0.0
        contact_time[hit_indices] = stop[hit]

        active = active[~hit]

    return probability, contact_time


def metricEggert(
    grid,
    dynamics,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    *,
    horizon: float = 1.0,
    dt: float = 0.01,
    beta_d: float = 1.0 / 1.5,
    tau_d0: float = 0.3,
    escape_rate: float = 0.1,
    p_crit: float = 0.5,
    collision_tolerance: float = 1e-9,
    contact_time_tolerance: float = 1e-5,
    batch_size: int = 10_000,
    use_symmetry: bool = True,
) -> EggertMetricResult:
    """Compute the metric on a 6D grid, returning NumPy arrays.

    Symmetry is used only when reflected grid nodes are available.
    Angles differing by 2*pi are treated as equivalent.

    General curved-motion contact detection remains sampled:
    very brief contacts between samples can be missed.
    """

    # -------------------------------------------------------------
    # 1. Validate inputs
    # -------------------------------------------------------------

    for name, value in {
        "horizon": horizon,
        "dt": dt,
        "beta_d": beta_d,
        "tau_d0": tau_d0,
        "contact_time_tolerance": contact_time_tolerance,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    for name, value in {
        "escape_rate": escape_rate,
        "collision_tolerance": collision_tolerance,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    if dt > horizon:
        raise ValueError("dt must not exceed horizon")

    if not np.isfinite(p_crit) or not 0.0 < p_crit < 1.0:
        raise ValueError("p_crit must lie strictly between 0 and 1")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    lf, lr = float(dynamics.lf), float(dynamics.lr)
    if not np.all(np.isfinite([lf, lr])) or min(lf, lr) <= 0.0:
        raise ValueError("lf and lr must be finite and positive")

    if grid.ndim != 6:
        raise ValueError("grid must be six-dimensional")

    axes = tuple(
        np.asarray(axis, dtype=float)
        for axis in grid.coordinate_vectors
    )
    if len(axes) != 6 or any(
        axis.ndim != 1
        or axis.size == 0
        or not np.all(np.isfinite(axis))
        for axis in axes
    ):
        raise ValueError("Expected six finite, non-empty 1D axes")

    shape = tuple(axis.size for axis in axes)
    x_axis, y_axis, theta_axis, vh_axis, delta_axis, ve_axis = axes

    probability = np.empty(shape)
    contact_time = np.empty(shape)

    # -------------------------------------------------------------
    # 2. Prepare reflection maps and the XY plane
    # -------------------------------------------------------------

    y_reflection = _reflection_indices(y_axis)
    theta_reflection = _reflection_indices(theta_axis, angular=True)
    delta_reflection = _reflection_indices(delta_axis)

    symmetry_used = bool(
        use_symmetry
        and y_reflection is not None
        and theta_reflection is not None
        and delta_reflection is not None
    )

    ix, iy = np.indices((x_axis.size, y_axis.size))
    ix, iy = ix.ravel(), iy.ravel()
    evaluated_states = 0

    # -------------------------------------------------------------
    # 3. Process groups with shared nominal kinematics
    # -------------------------------------------------------------

    group_shape = shape[2:]

    for group in np.ndindex(group_shape):
        i_theta, i_vh, i_delta, i_ve = group

        if symmetry_used:
            reflected_group = (
                int(theta_reflection[i_theta]),
                i_vh,
                int(delta_reflection[i_delta]),
                i_ve,
            )

            # The smaller tuple is responsible for both groups.
            if group > reflected_group:
                continue

            same_group = group == reflected_group
        else:
            reflected_group = group
            same_group = False

        selected = np.arange(ix.size)

        if symmetry_used and same_group:
            # Within a self-reflected group, compute only half of XY.
            selected = selected[iy <= y_reflection[iy]]

        for first in range(0, selected.size, batch_size):
            selection = selected[first:first + batch_size]
            bx, by = ix[selection], iy[selection]

            batch_probability, batch_contact = _evaluate_motion_group(
                x_axis[bx],
                y_axis[by],
                theta_axis[i_theta],
                vh_axis[i_vh],
                delta_axis[i_delta],
                ve_axis[i_ve],
                lf=lf,
                lr=lr,
                ego=ego,
                human=human,
                horizon=horizon,
                dt=dt,
                beta_d=beta_d,
                tau_d0=tau_d0,
                escape_rate=escape_rate,
                collision_tolerance=collision_tolerance,
                contact_time_tolerance=contact_time_tolerance,
            )

            target = (bx, by, *group)
            probability[target] = batch_probability
            contact_time[target] = batch_contact

            if symmetry_used:
                reflected_target = (
                    bx, y_reflection[by], *reflected_group
                )
                probability[reflected_target] = batch_probability
                contact_time[reflected_target] = batch_contact

            evaluated_states += selection.size

    # -------------------------------------------------------------
    # 4. Return probability and signed terminal values
    # -------------------------------------------------------------

    np.clip(probability, 0.0, 1.0, out=probability)

    return EggertMetricResult(
        probability=probability,
        terminal_values=p_crit - probability,
        first_contact_time=contact_time,
        parameters={
            "horizon": horizon,
            "dt": dt,
            "beta_d": beta_d,
            "tau_d0": tau_d0,
            "escape_rate": escape_rate,
            "p_crit": p_crit,
            "collision_tolerance": collision_tolerance,
            "contact_time_tolerance": contact_time_tolerance,
            "lf": lf,
            "lr": lr,
            "ego_length": ego.length,
            "ego_width": ego.width,
            "human_length": human.length,
            "human_width": human.width,
            "contact_policy": "absorb_survivors_at_first_contact",
            "symmetry_used": symmetry_used,
            "evaluated_states": evaluated_states,
            "total_states": int(np.prod(shape)),
        },
    )