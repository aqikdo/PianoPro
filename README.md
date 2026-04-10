# PianoPro

PianoPro is a research codebase for robotic piano playing built on top of a vendored copy of `robopianist`. The repository combines:

- a RoboPianist-style MuJoCo environment in [`robopianist/`](./robopianist)
- single-task PPO training in [`single_task/`](./single_task)
- multi-task / hierarchical diffusion-policy training in [`multi_task/`](./multi_task)
- a goal autoencoder in [`goal_auto_encoder/`](./goal_auto_encoder)
- bundled Zarr datasets in [`dataset_hl.zarr/`](./dataset_hl.zarr) and [`dataset_ll.zarr/`](./dataset_ll.zarr)
- a preprocessing/demo tutorial in [`tutorial/`](./tutorial)

The paper PDF is included as [`PianoPro.pdf`](./PianoPro.pdf).

## Repository status

This repository is not packaged as a polished library. It is closer to a research snapshot:

- some scripts still reference an older `pianomime/` directory layout
- several training/evaluation paths expect extra assets such as note trajectories, demonstrations, or checkpoints that are not all present in this repo
- Linux/macOS helper scripts are included, but Windows setup is not documented in-code

The README below focuses on what is actually present and how to navigate it safely.

## What is in the repo

### 1. RoboPianist environment

[`robopianist/`](./robopianist) contains a forked RoboPianist environment and utilities:

- task definitions under `robopianist/suite/tasks/`
- piano and hand models under `robopianist/models/`
- MIDI and audio helpers under `robopianist/music/`
- wrappers and viewer utilities under `robopianist/wrappers/` and `robopianist/viewer/`

The vendored package reports version `1.0.9`.

### 2. Single-task RL

[`single_task/`](./single_task) trains a residual PPO policy for one piece/task at a time. Core files:

- `train_ppo.py`: PPO training entry point
- `eval_ppo.py`: rollout/evaluation script
- `utils.py`: environment construction, wrappers, demonstrations, residual action logic
- `piano_with_shadow_hands_res.py`: task definition used by training

The PPO pipeline uses `stable_baselines3`, vectorized environments, optional residual actions, and DeepMimic-style demonstration guidance.

### 3. Multi-task / hierarchical policies

[`multi_task/`](./multi_task) contains diffusion-based policy training and evaluation:

- `train_high_level.py`: high-level trajectory model
- `train_low_level.py`: low-level action model
- `train_low_level_mlp.py` / `train_low_level_bet.py`: alternative low-level training variants
- `train_single_stage.py`: single-stage diffusion model variant
- `eval_high_level.py`: generate high-level fingertip trajectories
- `eval_low_level.py`: roll out the low-level controller in the environment
- `dataset.py`: Zarr dataset loading, normalization, sequence windowing
- `network.py`: diffusion model building blocks

The code uses Hugging Face `diffusers` for DDPM scheduling and EMA-based checkpointing.

### 4. Goal autoencoder

[`goal_auto_encoder/`](./goal_auto_encoder) trains an autoencoder over goal/state representations:

- `train_sdf_ae.py`: training loop
- `test_sdf_ae.py`: evaluation/testing script
- `dataset.py`, `network.py`, `loss.py`: data loading, model, and loss definitions

This encoder is later loaded by parts of the hierarchical evaluation pipeline.

### 5. Tutorial assets

[`tutorial/`](./tutorial) includes a notebook and media for a preprocessing workflow:

- `data_preprocessing.ipynb`
- sample MIDI/video/assets
- hand landmark model and example outputs

## Dependencies

The Python dependencies tracked in [`requirements.txt`](./requirements.txt) include:

- PyTorch ecosystem
- MuJoCo 3.0
- `stable-baselines3`
- `diffusers`
- `zarr`
- `gymnasium`
- audio packages such as `PyAudio` and `pyFluidSynth`

Install Python packages with:

```bash
pip install -r requirements.txt
```

For Linux/macOS system packages, there is also:

```bash
bash scripts/install_deps.sh
```

That script installs `fluidsynth`, `portaudio`, `ffmpeg`, and copies soundfonts / Shadow Hand assets into the expected RoboPianist locations.

## Quick start

### Check the RoboPianist fork

```bash
python robopianist/cli.py --version
python robopianist/cli.py --check-pig-exists
```

You can also inspect the available RoboPianist tasks from Python:

```python
from robopianist import suite
print(suite.ALL)
```

### Train a single-task PPO policy

The main entry point is:

```bash
python single_task/train_ppo.py --help
```

There are example shell scripts in [`scripts/run_ppo.sh`](./scripts/run_ppo.sh) and [`single_task/run_ppo.sh`](./single_task/run_ppo.sh), but they still reference the old `pianomime/` path layout. Treat them as parameter examples rather than copy-paste commands.

### Train diffusion models

High-level diffusion model:

```bash
python multi_task/train_high_level.py dataset_hl.zarr
```

Low-level diffusion model:

```bash
python multi_task/train_low_level.py dataset_ll.zarr
```

### Train the goal autoencoder

```bash
python goal_auto_encoder/train_sdf_ae.py dataset_hl.zarr
```

## Data

Two Zarr datasets are bundled at the repository root:

- [`dataset_hl.zarr/`](./dataset_hl.zarr): high-level dataset
- [`dataset_ll.zarr/`](./dataset_ll.zarr): low-level dataset

The diffusion loaders expect the Zarr layout:

- `data/action`
- `data/state`
- `meta/episode_ends`

Additional paths referenced by some scripts, such as `dataset/high_level_trajectories/`, `dataset/notes/`, `trained_songs/`, or checkpoint files like `checkpoint_ae.ckpt`, are not fully bundled here and may need to be generated or restored from the original experiment workspace.

## Outputs and checkpoints

By convention, scripts write outputs to paths such as:

- `robopianist_rl/ckpts/`
- `robopianist_rl/tensorboard/`
- `trained_songs/`
- `diffusion/ckpts/`
- `vae/ckpts/`

Create those directories if your run script assumes they already exist.

## Project structure

```text
PianoPro/
├── robopianist/         # Environment, tasks, models, wrappers, viewer
├── single_task/         # PPO-based single-song training/evaluation
├── multi_task/          # Diffusion-based hierarchical training/evaluation
├── goal_auto_encoder/   # Goal representation autoencoder
├── tutorial/            # Notebook and preprocessing/demo assets
├── scripts/             # Example shell scripts and dependency helper
├── dataset_hl.zarr/     # High-level dataset
├── dataset_ll.zarr/     # Low-level dataset
└── PianoPro.pdf         # Paper / project PDF
```

## Known caveats

- Several scripts hardcode `torch.device('cuda')`; CPU-only execution will require edits.
- Some helper scripts assume a Unix shell and EGL-based MuJoCo rendering.
- Path references are inconsistent: current repo root is flat, while some scripts still expect `pianomime/...`.
- Evaluation scripts expect pretrained checkpoints to exist in the working directory.
- The repository contains vendored third-party assets and a modified RoboPianist codebase, so upstream RoboPianist documentation may not match behavior exactly.

## Suggested workflow for new users

1. Install Python and system dependencies.
2. Verify the RoboPianist fork imports and the soundfont file exists.
3. Inspect the training script you plan to run and fix any old `pianomime/` paths first.
4. Start with the bundled Zarr datasets before attempting full environment rollouts.
5. Treat the shell scripts as experiment notes, not as guaranteed one-command entry points.

## Citation

If you use this repository, cite the corresponding paper in [`PianoPro.pdf`](./PianoPro.pdf) and the original RoboPianist project.
