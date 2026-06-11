# Inverted Pendulum Control Simulator

A small Tkinter desktop app that drives Gymnasium's
[`InvertedPendulum-v5`](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)
(MuJoCo) environment with three control modes:

1. **Manual control** -- balance the pole yourself with the arrow keys (or
   on-screen buttons).
2. **PID control** -- a PID controller balances the pole; tune Kp/Ki/Kd live
   with sliders.
3. **Reinforcement learning control** -- train (or load) a Stable-Baselines3
   PPO agent and watch it balance the pole.

A separate MuJoCo viewer window renders the cart-and-pole, while a Tkinter
control panel provides the menus, buttons, sliders and status readouts.

## How it works

The environment's observation is `[x, theta, x_dot, theta_dot]` (cart
position, pole angle from vertical, cart velocity, pole angular velocity)
and the action is a single force in `[-3, 3]` applied to the cart. An
episode ends when the pole falls past a threshold angle or after 1000 steps.

- **`controllers.py`** contains the three control algorithms, each in its
  own function/class so they can be studied or re-implemented independently
  of the GUI:
  - `get_manual_action(action_space, pressed_keys)` -- maps held arrow
    keys/buttons to a force.
  - `PIDController` -- a PID controller (`compute_action`) with live-tunable
    `kp`/`ki`/`kd` gains.
  - `train_rl_agent`, `load_rl_agent`, `get_rl_action` -- train/load a
    Stable-Baselines3 PPO agent and query it for actions.
- **`simulation.py`** is the Tkinter application: the main menu, the three
  mode screens, the simulation step loop (driven by `root.after`), the
  MuJoCo viewer window setup, and RL training/checkpoint management.
- **`notebooks/`** contains a Jupyter exercise notebook (the same functions
  left unimplemented, with explanations) and a solution notebook, each with
  an export cell that regenerates `controllers.py`.

## Project layout

```
controllers.py               # the three control algorithms (used by the simulator)
simulation.py                 # the Tkinter app / simulation loop
requirements.txt              # Python dependencies
checkpoints/                   # trained RL models are saved here
notebooks/
  controllers_exercise.ipynb  # same functions, left blank, with explanations
  controllers_solution.ipynb  # fully worked solutions + export script
```

## Setup

Requires Python 3.10+ and a display (MuJoCo's renderer needs one; on a
headless machine set up a virtual display such as `Xvfb` first).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the simulator

```bash
python simulation.py
```

This opens a window with three buttons: **Manual Control**, **PID Control**,
and **Reinforcement Learning Control**.

A MuJoCo viewer window opens alongside the control panel. Its on-screen
overlay (Pan/Zoom, Switch camera, FPS, ...) is hidden by default so it
doesn't block the view of the pendulum -- press `H` in the viewer window to
toggle it back on.

When an episode ends (the pole falls or the step limit is reached), the
simulation pauses and the control panel shows the final episode reward with
a **Start New Episode** button. Click it, or press `R` in the viewer window,
to reset and continue -- or use **Back to Menu** / `Esc` to exit.

### Manual control

Click on the viewer window, then use:

- `←` / `→` -- push the cart left / right
- `R` -- reset the episode immediately
- `Esc` -- return to the main menu

The control panel also has **◀ Left** / **Right ▶** buttons that work the
same as the arrow keys, for trackpad/touch use.

### PID control

Three sliders (Kp, Ki, Kd) let you tune the controller while it runs. The
controller drives the pole angle to 0 (upright). Try `Kp=20, Ki=0, Kd=2` as a
starting point, then experiment. `R` resets the episode immediately, `Esc`
returns to the menu.

### Reinforcement learning control

- If no checkpoint exists yet (`checkpoints/ppo_inverted_pendulum.zip`), you
  configure PPO hyperparameters (learning rate, n_steps, batch_size, gamma,
  gae_lambda, ent_coef, total_timesteps) and train a new agent. A progress
  bar tracks training, which runs in the background so the UI stays
  responsive.
- Once a checkpoint exists, you can choose to **use the existing trained
  model** or **train a new one** (overwriting the checkpoint).
- After training/loading, the trained agent runs in the MuJoCo viewer.

## Notebooks: implement the algorithms yourself

`controllers.py` ships with working implementations so the simulator runs
out of the box. To learn how each algorithm works (or re-implement them
yourself):

1. Open `notebooks/controllers_exercise.ipynb`. Each control algorithm has a
   markdown cell explaining how it works and step-by-step implementation
   instructions, followed by a code cell with the function/class signature
   but the body left as `# YOUR CODE HERE` / `raise NotImplementedError`.
2. Implement `get_manual_action`, `PIDController`, `train_rl_agent`,
   `load_rl_agent`, and `get_rl_action`.
3. Run the **export cell** at the end of the notebook -- it writes your
   implementations to `../controllers.py`.
4. Run `python simulation.py` from the project root to test your
   implementations.

If you get stuck, `notebooks/controllers_solution.ipynb` contains the same
explanations with fully worked implementations, plus the same export cell.
