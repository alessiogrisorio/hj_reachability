import jax.numpy as jnp

from hj_reachability.systems.relative_vehicle_6d import RelativeVehicle6D


LF = 1.2
LR = 1.5


def test_output_shapes():
    system = RelativeVehicle6D(
        lf=LF,
        lr=LR,
    )

    state = jnp.array([
        10.0,
        0.0,
        0.0,
        15.0,
        0.0,
        15.0,
    ])

    time = 0.0

    open_loop = system.open_loop_dynamics(state, time)
    control_matrix = system.control_jacobian(state, time)
    disturbance_matrix = system.disturbance_jacobian(state, time)

    assert open_loop.shape == (6,)
    assert control_matrix.shape == (6, 2)
    assert disturbance_matrix.shape == (6, 2)


def test_equal_parallel_velocities():
    system = RelativeVehicle6D(
        lf=LF,
        lr=LR,
    )

    state = jnp.array([
        10.0,
        0.0,
        0.0,
        15.0,
        0.0,
        15.0,
    ])

    derivative = system.open_loop_dynamics(state, time=0.0)

    expected = jnp.zeros(6)

    assert jnp.allclose(derivative, expected)