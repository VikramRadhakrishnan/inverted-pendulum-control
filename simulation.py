"""
Inverted Pendulum control simulation.

Provides a small Tkinter front-end for the Gymnasium ``InvertedPendulum-v5``
(MuJoCo) environment with three control modes:

1. Manual control  -- drive the cart with the arrow keys.
2. PID control     -- a PID controller balances the pole; tune Kp/Ki/Kd
                       live with sliders.
3. RL control      -- train (or load) a Stable-Baselines3 PPO agent and
                       watch it balance the pole.

The actual control algorithms live in ``controllers.py`` so they can be
implemented/edited independently (see ``notebooks/``).

Run with::

    python simulation.py
"""

import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import gymnasium as gym

from controllers import (
    PIDController,
    get_manual_action,
    get_rl_action,
    load_rl_agent,
    train_rl_agent,
)

ENV_ID = "InvertedPendulum-v5"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "ppo_inverted_pendulum.zip")

# Delay (ms) between simulation steps, driven by Tkinter's event loop.
STEP_DELAY_MS = 20

# Force applied per key press in manual mode. The action space allows up to
# 3.0, but pushing that hard makes the cart overshoot and the pole nearly
# impossible to balance by hand.
MANUAL_FORCE_MAGNITUDE = 1.0


def _warm_up_torch_dynamo():
    """Import ``torch._dynamo`` once, before any GLFW window is created.

    Stable-Baselines3 lazily imports ``torch._dynamo`` (and its CUDA
    bindings) the first time an optimizer is constructed, which happens on
    the background training thread. If a GLFW render window has already
    been opened and closed on the main thread by then, that first import
    segfaults. Doing it here, eagerly and on the main thread, avoids the
    crash.
    """
    try:
        import torch._dynamo  # noqa: F401
    except Exception:
        pass


class PendulumSimulatorApp:
    """Tkinter application that ties the GUI to the Gymnasium environment."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Inverted Pendulum Control Simulator")
        self.root.geometry("480x420")
        self.root.resizable(False, False)

        self.env = None
        self.obs = None
        self.episode_reward = 0.0
        self.pressed_keys = {}
        self.after_id = None
        self.episode_end_frame = None
        self._torch_dynamo_warmed = False

        self.pid = PIDController()
        self.rl_model = None

        self.show_main_menu()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()

    def _clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.episode_end_frame = None

    def _stop_loop(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _close_env(self):
        self._stop_loop()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _make_env(self, render_mode=None):
        if render_mode is not None and not self._torch_dynamo_warmed:
            _warm_up_torch_dynamo()
            self._torch_dynamo_warmed = True
        return gym.make(ENV_ID, render_mode=render_mode)

    def _render_window_closed(self):
        """Return True if the user closed the MuJoCo viewer window."""
        renderer = getattr(self.env.unwrapped, "mujoco_renderer", None)
        viewer = getattr(renderer, "viewer", None)
        return viewer is not None and getattr(viewer, "window", None) is None

    def _bind_glfw_keys(self):
        """Hook arrow/reset/quit keys into the MuJoCo render window."""
        try:
            import glfw

            viewer = self.env.unwrapped.mujoco_renderer.viewer
            window = viewer.window
            if window is None:
                return

            # The default MuJoCo overlay (Pan/Zoom, Switch camera, FPS, ...)
            # covers most of the small render window. Hide it by default;
            # the user can press H to bring it back.
            viewer._hide_menu = True

            def key_callback(_window, key, _scancode, action, _mods):
                if key == glfw.KEY_LEFT:
                    self.pressed_keys["left"] = action != glfw.RELEASE
                elif key == glfw.KEY_RIGHT:
                    self.pressed_keys["right"] = action != glfw.RELEASE
                elif key == glfw.KEY_R and action == glfw.PRESS:
                    self.pressed_keys["reset"] = True
                elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                    self.pressed_keys["quit"] = True
                elif key == glfw.KEY_H and action == glfw.PRESS:
                    viewer._hide_menu = not viewer._hide_menu

            glfw.set_key_callback(window, key_callback)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Warning: could not bind keyboard controls to render window ({exc})")

    def _show_episode_end_prompt(self, on_restart):
        """Pause the step loop and ask the user whether to start a new episode."""
        self.episode_end_frame = tk.Frame(self.root)
        self.episode_end_frame.pack(pady=10)

        tk.Label(
            self.episode_end_frame,
            text=f"Episode finished! Final reward: {self.episode_reward:.2f}",
            font=("Helvetica", 11, "bold"),
        ).pack(pady=(0, 5))

        tk.Button(
            self.episode_end_frame,
            text="Start New Episode",
            width=20,
            command=on_restart,
        ).pack()

    def _dismiss_episode_end_prompt(self):
        if self.episode_end_frame is not None:
            self.episode_end_frame.destroy()
            self.episode_end_frame = None

    def _set_pressed_key(self, key, value):
        """Return an event handler that sets ``pressed_keys[key] = value``.

        Used to make on-screen buttons act as proxies for held-down keys.
        """

        def handler(_event):
            self.pressed_keys[key] = value

        return handler

    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------
    def show_main_menu(self):
        self._close_env()
        self._clear()

        tk.Label(
            self.root, text="Inverted Pendulum Simulator", font=("Helvetica", 18, "bold")
        ).pack(pady=(30, 5))
        tk.Label(self.root, text="Choose a control method:", font=("Helvetica", 12)).pack(
            pady=(0, 20)
        )

        tk.Button(
            self.root, text="Manual Control", width=30, height=2, command=self.start_manual
        ).pack(pady=8)
        tk.Button(
            self.root, text="PID Control", width=30, height=2, command=self.start_pid
        ).pack(pady=8)
        tk.Button(
            self.root,
            text="Reinforcement Learning Control",
            width=30,
            height=2,
            command=self.start_rl_menu,
        ).pack(pady=8)

        tk.Button(self.root, text="Quit", width=30, command=self.root.destroy).pack(pady=(30, 10))

    # ------------------------------------------------------------------
    # Manual control
    # ------------------------------------------------------------------
    def start_manual(self):
        self._close_env()
        self._clear()

        self.env = self._make_env(render_mode="human")
        self.obs, _ = self.env.reset(seed=0)
        self.episode_reward = 0.0
        self.pressed_keys = {"left": False, "right": False, "reset": False, "quit": False}

        tk.Label(self.root, text="Manual Control", font=("Helvetica", 16, "bold")).pack(pady=10)
        tk.Label(
            self.root,
            justify="left",
            text=(
                "Click on the simulation window, then use:\n\n"
                "  ← / →   push the cart left / right\n"
                "  R       reset the episode\n"
                "  Esc     return to the menu\n\n"
                "Or use the on-screen buttons below."
            ),
        ).pack(pady=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        left_button = tk.Button(button_frame, text="◀ Left", width=10)
        left_button.pack(side="left", padx=10)
        left_button.bind("<ButtonPress-1>", self._set_pressed_key("left", True))
        left_button.bind("<ButtonRelease-1>", self._set_pressed_key("left", False))

        right_button = tk.Button(button_frame, text="Right ▶", width=10)
        right_button.pack(side="left", padx=10)
        right_button.bind("<ButtonPress-1>", self._set_pressed_key("right", True))
        right_button.bind("<ButtonRelease-1>", self._set_pressed_key("right", False))

        self.status_var = tk.StringVar(value="Episode reward: 0.00")
        tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 11)).pack(pady=10)

        tk.Button(self.root, text="Back to Menu", width=20, command=self.show_main_menu).pack(
            pady=20
        )

        self.env.render()
        self._bind_glfw_keys()
        self._manual_step()

    def _manual_step(self):
        if self.env is None:
            return

        if self._render_window_closed() or self.pressed_keys.get("quit"):
            self.show_main_menu()
            return

        if self.pressed_keys.get("reset"):
            self.obs, _ = self.env.reset()
            self.episode_reward = 0.0
            self.pressed_keys["reset"] = False

        action = get_manual_action(
            self.env.action_space, self.pressed_keys, magnitude=MANUAL_FORCE_MAGNITUDE
        )
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.episode_reward += reward
        self.env.render()

        self.status_var.set(f"Episode reward: {self.episode_reward:.2f}")

        if terminated or truncated:
            self._show_episode_end_prompt(self._restart_manual_episode)
            return

        self.after_id = self.root.after(STEP_DELAY_MS, self._manual_step)

    def _restart_manual_episode(self):
        self._dismiss_episode_end_prompt()
        self.obs, _ = self.env.reset()
        self.episode_reward = 0.0
        self._manual_step()

    # ------------------------------------------------------------------
    # PID control
    # ------------------------------------------------------------------
    def start_pid(self):
        self._close_env()
        self._clear()

        self.env = self._make_env(render_mode="human")
        self.obs, _ = self.env.reset(seed=0)
        self.episode_reward = 0.0
        self.pressed_keys = {"reset": False, "quit": False}
        self.pid.reset()

        tk.Label(self.root, text="PID Control", font=("Helvetica", 16, "bold")).pack(pady=10)
        tk.Label(
            self.root,
            text="Tune the gains while the controller runs. (R = reset, Esc = menu)",
        ).pack(pady=(0, 10))

        self.kp_var = tk.DoubleVar(value=20.0)
        self.ki_var = tk.DoubleVar(value=0.0)
        self.kd_var = tk.DoubleVar(value=2.0)

        slider_config = [
            ("Kp", self.kp_var, 0.0, 100.0),
            ("Ki", self.ki_var, 0.0, 5.0),
            ("Kd", self.kd_var, 0.0, 20.0),
        ]
        for label, var, lo, hi in slider_config:
            row = tk.Frame(self.root)
            row.pack(fill="x", padx=30, pady=4)
            tk.Label(row, text=label, width=4, anchor="w").pack(side="left")
            tk.Scale(
                row,
                from_=lo,
                to=hi,
                resolution=0.05,
                orient="horizontal",
                variable=var,
                length=280,
            ).pack(side="left")

        self.status_var = tk.StringVar(value="Episode reward: 0.00")
        tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 11)).pack(pady=10)

        tk.Button(self.root, text="Back to Menu", width=20, command=self.show_main_menu).pack(
            pady=10
        )

        self.env.render()
        self._bind_glfw_keys()
        self._pid_step()

    def _pid_step(self):
        if self.env is None:
            return

        if self._render_window_closed() or self.pressed_keys.get("quit"):
            self.show_main_menu()
            return

        if self.pressed_keys.get("reset"):
            self.obs, _ = self.env.reset()
            self.episode_reward = 0.0
            self.pid.reset()
            self.pressed_keys["reset"] = False

        self.pid.set_gains(self.kp_var.get(), self.ki_var.get(), self.kd_var.get())
        action = self.pid.compute_action(self.obs, self.env.action_space, self.env.unwrapped.dt)
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.episode_reward += reward
        self.env.render()

        self.status_var.set(
            f"Episode reward: {self.episode_reward:.2f}    "
            f"(Kp={self.kp_var.get():.2f}, Ki={self.ki_var.get():.2f}, Kd={self.kd_var.get():.2f})"
        )

        if terminated or truncated:
            self._show_episode_end_prompt(self._restart_pid_episode)
            return

        self.after_id = self.root.after(STEP_DELAY_MS, self._pid_step)

    def _restart_pid_episode(self):
        self._dismiss_episode_end_prompt()
        self.obs, _ = self.env.reset()
        self.episode_reward = 0.0
        self.pid.reset()
        self._pid_step()

    # ------------------------------------------------------------------
    # RL control
    # ------------------------------------------------------------------
    def start_rl_menu(self):
        self._close_env()
        self._clear()

        tk.Label(
            self.root, text="Reinforcement Learning Control", font=("Helvetica", 16, "bold")
        ).pack(pady=15)

        if os.path.exists(CHECKPOINT_PATH):
            tk.Label(
                self.root,
                justify="left",
                text=f"A trained checkpoint was found:\n{CHECKPOINT_PATH}",
            ).pack(pady=10)
            tk.Button(
                self.root,
                text="Use Existing Trained Model",
                width=30,
                height=2,
                command=self.run_rl_model,
            ).pack(pady=8)
            tk.Button(
                self.root,
                text="Train a New Model",
                width=30,
                height=2,
                command=self.show_rl_hyperparameters,
            ).pack(pady=8)
        else:
            tk.Label(
                self.root,
                text="No trained model found yet.\nConfigure hyperparameters and train a new agent.",
                justify="left",
            ).pack(pady=10)
            tk.Button(
                self.root,
                text="Configure & Train New Model",
                width=30,
                height=2,
                command=self.show_rl_hyperparameters,
            ).pack(pady=8)

        tk.Button(self.root, text="Back to Menu", width=30, command=self.show_main_menu).pack(
            pady=(30, 10)
        )

    def show_rl_hyperparameters(self):
        self._clear()

        tk.Label(
            self.root, text="RL Training Hyperparameters", font=("Helvetica", 16, "bold")
        ).pack(pady=15)

        defaults = {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "total_timesteps": 50000,
        }

        self.hp_vars = {}
        form = tk.Frame(self.root)
        form.pack(padx=30, pady=5)
        for i, (key, default) in enumerate(defaults.items()):
            tk.Label(form, text=key, anchor="w", width=16).grid(row=i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=str(default))
            tk.Entry(form, textvariable=var, width=12).grid(row=i, column=1, pady=3)
            self.hp_vars[key] = var

        self.train_status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.train_status_var).pack(pady=(10, 2))

        self.progress = ttk.Progressbar(self.root, length=300, mode="determinate")
        self.progress.pack(pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.train_button = tk.Button(btn_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side="left", padx=5)
        tk.Button(btn_frame, text="Back", command=self.start_rl_menu).pack(side="left", padx=5)

        self.run_button_frame = tk.Frame(self.root)
        self.run_button_frame.pack(pady=5)

    def start_training(self):
        try:
            hyperparams = {
                "learning_rate": float(self.hp_vars["learning_rate"].get()),
                "n_steps": int(self.hp_vars["n_steps"].get()),
                "batch_size": int(self.hp_vars["batch_size"].get()),
                "gamma": float(self.hp_vars["gamma"].get()),
                "gae_lambda": float(self.hp_vars["gae_lambda"].get()),
                "ent_coef": float(self.hp_vars["ent_coef"].get()),
            }
            total_timesteps = int(self.hp_vars["total_timesteps"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numbers for all hyperparameters.")
            return

        self.train_button.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = total_timesteps
        self.train_status_var.set("Training in progress... (see console for SB3 logs)")

        self._train_progress = {"steps": 0}
        self._training_done = False
        self._training_error = None

        def train_thread():
            try:
                from stable_baselines3.common.callbacks import BaseCallback

                progress = self._train_progress

                class ProgressCallback(BaseCallback):
                    def _on_step(self_inner):
                        progress["steps"] = self_inner.num_timesteps
                        return True

                train_env = self._make_env(render_mode=None)
                train_rl_agent(
                    train_env,
                    hyperparams,
                    total_timesteps,
                    CHECKPOINT_PATH,
                    callback=ProgressCallback(),
                )
                train_env.close()
            except Exception as exc:  # noqa: BLE001
                self._training_error = exc
            finally:
                self._training_done = True

        threading.Thread(target=train_thread, daemon=True).start()
        self._poll_training()

    def _poll_training(self):
        self.progress["value"] = min(self._train_progress["steps"], self.progress["maximum"])

        if not self._training_done:
            # Track this via self.after_id so _close_env() (called when the
            # user navigates away) can cancel it and avoid updating widgets
            # that have since been destroyed.
            self.after_id = self.root.after(200, self._poll_training)
            return

        self.after_id = None

        if self._training_error is not None:
            messagebox.showerror("Training failed", str(self._training_error))
            self.train_button.config(state="normal")
            return

        self.train_status_var.set("Training complete! Model saved to checkpoints/.")
        tk.Button(
            self.run_button_frame,
            text="Run Trained Model",
            width=24,
            height=2,
            command=self.run_rl_model,
        ).pack(pady=5)

    def run_rl_model(self):
        self._close_env()
        self._clear()

        self.env = self._make_env(render_mode="human")
        self.obs, _ = self.env.reset(seed=0)
        self.episode_reward = 0.0
        self.pressed_keys = {"reset": False, "quit": False}
        self.rl_model = load_rl_agent(CHECKPOINT_PATH)

        tk.Label(
            self.root, text="RL Control (trained model)", font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        tk.Label(self.root, text="(R = reset episode, Esc = back to menu)").pack(pady=(0, 10))

        self.status_var = tk.StringVar(value="Episode reward: 0.00")
        tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 11)).pack(pady=10)

        tk.Button(self.root, text="Back to Menu", width=20, command=self.show_main_menu).pack(
            pady=20
        )

        self.env.render()
        self._bind_glfw_keys()
        self._rl_step()

    def _rl_step(self):
        if self.env is None:
            return

        if self._render_window_closed() or self.pressed_keys.get("quit"):
            self.show_main_menu()
            return

        if self.pressed_keys.get("reset"):
            self.obs, _ = self.env.reset()
            self.episode_reward = 0.0
            self.pressed_keys["reset"] = False

        action = get_rl_action(self.rl_model, self.obs)
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.episode_reward += reward
        self.env.render()

        self.status_var.set(f"Episode reward: {self.episode_reward:.2f}")

        if terminated or truncated:
            self._show_episode_end_prompt(self._restart_rl_episode)
            return

        self.after_id = self.root.after(STEP_DELAY_MS, self._rl_step)

    def _restart_rl_episode(self):
        self._dismiss_episode_end_prompt()
        self.obs, _ = self.env.reset()
        self.episode_reward = 0.0
        self._rl_step()


if __name__ == "__main__":
    app = PendulumSimulatorApp()
    app.run()
