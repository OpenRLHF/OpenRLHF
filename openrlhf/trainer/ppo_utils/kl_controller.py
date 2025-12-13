import numpy as np


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def build_kl_controller(init_kl_coef, kl_target=None, kl_horizon=10000):
    """Factory to build KL controller from explicit params."""
    if kl_target:
        return AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
    return FixedKLController(init_kl_coef)
