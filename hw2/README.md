# Homework 2: Policy Gradients

## Setup

For general setup and Modal instructions, see Homework 1's README.

## Examples

Here are some example commands. Run them in the `hw2` directory.

* To run on a local machine:
  ```bash
  uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```

* If you want Box2D environments (e.g., `LunarLander-v2`), install the optional extra:
  ```bash
  uv sync --extra box2d
  ```
  or run with the extra just for one command:
  ```bash
  uv run --extra box2d src/scripts/run.py --env_name LunarLander-v2 -n 100 -b 1000 --exp_name lunarlander
  ```

* If you want MuJoCo environments (e.g., `HalfCheetah-v4`), install the optional extra:
  ```bash
  uv sync --extra mujoco
  ```
  or run with the extra just for one command:
  ```bash
  uv run --extra mujoco src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah
  ```

* To run on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```
  * Note that Modal is likely not necessary for this assignment.
In testing, training was much faster on a local laptop CPU than on Modal.
However, you may still use Modal if you wish.
  * You may request a different GPU type, CPU count, and memory size by changing variables in `src/scripts/modal_run.py`
  * Use `modal run --detach` to keep your job running in the background.

## Troubleshooting

* If you see an error about `swig` when installing `box2d-py` (only needed for `--extra box2d`), install `swig` and `cmake`:
  * Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y swig cmake`
  * If you don't have `sudo`: `conda install -c conda-forge swig cmake`
  * macOS (Homebrew): `brew install swig cmake`
  * Modal: should already be installed
* If you see `ModuleNotFoundError: No module named 'glfw'` (or `'mujoco'`) with MuJoCo envs, install the MuJoCo extra and rerun with `--extra mujoco`.
