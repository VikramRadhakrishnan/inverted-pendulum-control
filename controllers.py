"""
Control algorithms for the Gymnasium ``InvertedPendulum-v5`` environment.

This file is consumed by ``simulation.py``. It is also the export target of
``notebooks/controllers_solution.ipynb`` (and, once you've completed the
exercises, ``notebooks/controllers_exercise.ipynb``) -- running the export
cell in either notebook regenerates this file from the notebook's code
cells.

Environment recap
------------------
- Observation: ``[x, theta, x_dot, theta_dot]``
    - ``x``        : cart position (m)
    - ``theta``    : pole angle from vertical (rad)
    - ``x_dot``    : cart velocity (m/s)
    - ``theta_dot``: pole angular velocity (rad/s)
- Action: a single force applied to the cart, ``Box(-3.0, 3.0, (1,))``.
"""

import os

import numpy as np


# ---------------------------------------------------------------------------
# 1. Manual control
# ---------------------------------------------------------------------------
def get_manual_action(action_space, pressed_keys, magnitude=None):
    """Convert currently-held keyboard keys into an action.

    Parameters
    ----------
    action_space : gymnasium.spaces.Box
        The environment's action space.
    pressed_keys : dict
        Maps key names (``"left"``, ``"right"``) to booleans indicating
        whether the corresponding arrow key is currently held down.
    magnitude : float, optional
        Magnitude of the force to apply. Defaults to the maximum force
        allowed by ``action_space``.

    Returns
    -------
    numpy.ndarray
        The action to send to ``env.step``, clipped to ``action_space``.
    """
    if magnitude is None:
        magnitude = float(action_space.high[0])

    force = 0.0
    if pressed_keys.get("left"):
        force -= magnitude
    if pressed_keys.get("right"):
        force += magnitude

    action = np.array([force], dtype=action_space.dtype)
    return np.clip(action, action_space.low, action_space.high)


# ---------------------------------------------------------------------------
# 2. PID control
# ---------------------------------------------------------------------------
class PIDController:
    """A simple PID controller that balances the pole at ``setpoint``.

    The controller drives the pole angle ``theta`` (``observation[1]``) to
    ``setpoint`` (0 rad = perfectly upright) by applying a force to the cart:

        u(t) = Kp * e(t) + Ki * integral(e) + Kd * d(e)/dt

    where ``e(t) = theta(t) - setpoint``.
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.reset()

    def reset(self):
        """Clear the integral and derivative memory (call between episodes)."""
        self._integral = 0.0
        self._prev_error = 0.0

    def set_gains(self, kp, ki, kd):
        """Update the P, I and D gains (e.g. from GUI sliders)."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute_action(self, observation, action_space, dt):
        """Compute the control action for the current observation.

        Parameters
        ----------
        observation : array-like
            ``[x, theta, x_dot, theta_dot]`` from the environment.
        action_space : gymnasium.spaces.Box
            The environment's action space, used to clip the output.
        dt : float
            Time step between calls (``env.unwrapped.dt``), used for the
            integral and derivative terms.

        Returns
        -------
        numpy.ndarray
            The action to send to ``env.step``, clipped to ``action_space``.
        """
        theta = observation[1]
        error = theta - self.setpoint

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        action = np.array([output], dtype=action_space.dtype)
        return np.clip(action, action_space.low, action_space.high)


# ---------------------------------------------------------------------------
# 3. Reinforcement learning control (Stable-Baselines3 PPO)
# ---------------------------------------------------------------------------
def train_rl_agent(env, hyperparams, total_timesteps, save_path, callback=None):
    """Train a PPO agent on ``env`` and save it to ``save_path``.

    Parameters
    ----------
    env : gymnasium.Env
        The (non-rendered) training environment.
    hyperparams : dict
        PPO hyperparameters. Recognised keys: ``learning_rate``, ``n_steps``,
        ``batch_size``, ``gamma``, ``gae_lambda``, ``ent_coef``. Missing keys
        fall back to Stable-Baselines3 defaults.
    total_timesteps : int
        Number of environment steps to train for.
    save_path : str
        Path (without/with ``.zip``) to save the trained model checkpoint.
    callback : stable_baselines3.common.callbacks.BaseCallback, optional
        Callback passed through to ``model.learn`` (e.g. for progress
        reporting in the GUI).

    Returns
    -------
    stable_baselines3.PPO
        The trained model.
    """
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get("learning_rate", 3e-4),
        n_steps=hyperparams.get("n_steps", 1024),
        batch_size=hyperparams.get("batch_size", 64),
        gamma=hyperparams.get("gamma", 0.99),
        gae_lambda=hyperparams.get("gae_lambda", 0.95),
        ent_coef=hyperparams.get("ent_coef", 0.0),
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_path)

    return model


def load_rl_agent(path, env=None):
    """Load a previously-trained PPO model from ``path``.

    Parameters
    ----------
    path : str
        Path to the saved model (``.zip`` checkpoint).
    env : gymnasium.Env, optional
        Environment to attach to the loaded model.

    Returns
    -------
    stable_baselines3.PPO
        The loaded model.
    """
    from stable_baselines3 import PPO

    return PPO.load(path, env=env)


def get_rl_action(model, observation, deterministic=True):
    """Get the action chosen by a trained RL model for ``observation``.

    Parameters
    ----------
    model : stable_baselines3.PPO
        A trained (or loaded) model.
    observation : array-like
        ``[x, theta, x_dot, theta_dot]`` from the environment.
    deterministic : bool
        Whether to use the deterministic policy (recommended for evaluation).

    Returns
    -------
    numpy.ndarray
        The action to send to ``env.step``.
    """
    action, _ = model.predict(observation, deterministic=deterministic)
    return action
