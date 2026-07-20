import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class RelativeVehicle6D(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(
            self,
            lf,
            lr,
            ego_min_acceleration=-7.0,
            ego_max_acceleration=2.5,
            ego_max_steering_rate=0.087,
            human_min_acceleration=-7.0,
            human_max_acceleration=2.5,
            human_max_yaw_rate=1.0,
            control_mode="max",
            disturbance_mode="min",
            control_space=None,
            disturbance_space=None,
        ):
        self.lf = lf
        self.lr = lr
        if control_space is None:
            control_space = sets.Box(
                l0 = jnp.array([
                    -ego_max_steering_rate,
                    ego_min_acceleration,
                ]),
                hi = jnp.array([
                    ego_max_steering_rate,
                    ego_max_acceleration,
                ]),
            )
        if disturbance_space is None:
            disturbance_space = sets.Box(
                jnp.array([
                    -human_max_yaw_rate,
                    human_min_acceleration,
                ]),
                jnp.array([
                    human_max_yaw_rate,
                    human_max_acceleration,
                ]),
            )
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        del time
        x_rel, y_rel, theta_rel, v_h, delta_e, v_e = state

        beta_e = jnp.arctan(
            self.lr / (self.lr + self.lf) * jnp.tan(delta_e)
        )

        omega_e = (
            v_e
            * jnp.cos(beta_e)
            / (self.lr + self.lf)
            * jnp.tan(delta_e)
        )

        return jnp.array([
            v_h * jnp.cos(theta_rel) - v_e * jnp.cos(beta_e) + y_rel * omega_e,
            v_h * jnp.sin(theta_rel) - v_e * jnp.sin(beta_e) - x_rel * omega_e,
            - omega_e,
            0.0,
            0.0,
            0.0,
        ])

    def control_jacobian(self, state, time):
        del state, time
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])

    def disturbance_jacobian(self, state, time):
        del state, time
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])

