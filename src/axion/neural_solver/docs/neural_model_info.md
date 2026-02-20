## Neural model overview (Axion Neural Solver)

This codebase trains a **neural dynamics model** that predicts the **time evolution of system states** from simulation trajectories. In the current “states-only” setup, the dataset contains **system states** and **next states** (optionally also `gravity_dir` / `root_body_q`), and does **not** contain contacts or actuation signals.

The key components are:

- **Model**: `axion.neural_solver.models.models.ModelMixedInput`
- **Transformer backbone** (continuous-input GPT): `axion.neural_solver.models.model_transformer.GPT`
- **Training loop / loss**: `axion.neural_solver.algorithms.vanilla_trainer.VanillaTrainer` (and `SequenceModelTrainer` for sequence datasets)
- **Target definition and angle wrapping**: `axion.neural_solver.integrators.newton_based_integrator_neural.NewtonBasedNeuralIntegrator`
- **Env adapter used by training**: `axion.neural_solver.envs.axionToTrajectorySampler.AxionEnvToTrajectorySamplerAdapter`

---

## What the model learns (supervision signal)

Training is supervised using dataset pairs \((x_t, x_{t+1})\), where:

- `states` is \(x_t\)
- `next_states` is \(x_{t+1}\)

The trainer constructs a regression target called `target` from `(states, next_states)` using:

- **Absolute prediction**: \(y_t = x_{t+1}\)
- **Relative prediction** (default in configs): \(y_t = x_{t+1} - x_t\)

Angular coordinates (joint angles) are wrapped so that deltas represent the **shortest signed angle difference** (avoid jumps at \(\pm \pi\)).

Implementation:
- Target construction happens in `VanillaTrainer.preprocess_data_batch()` via
  `neural_integrator.convert_next_states_to_prediction(states, next_states, dt)`.
- The relative/absolute behavior is controlled by `env.neural_integrator_cfg.prediction_type`.

---

## Learning algorithm and loss

### Trainer classes

`src/axion/neural_solver/train/train.py` selects the algorithm:

- **`VanillaTrainer`**: uses `BatchTransitionDataset` (transitions batched from an HDF5 file).
- **`SequenceModelTrainer`**: subclass of `VanillaTrainer` that switches to `TrajectoryDataset` and trains on fixed-length sequences.

For transformer training, configs typically use:

- `algorithm.name: SequenceModelTrainer`
- `network.transformer: ...` (enables transformer backbone)
- `algorithm.sample_sequence_length` aligned with `env.neural_integrator_cfg.num_states_history`

### Loss

The training objective is **mean squared error (MSE)** in the **prediction space**:

\[
\mathcal{L} = \mathrm{MSE}\left(\hat{y}_t, y_t\right)
\]

Where:
- \(\hat{y}_t = f_\theta(\text{inputs at time }t)\)
- \(y_t\) is the target computed from `(states, next_states)`

Optional output reweighting:
- If `network.normalize_output: True`, the loss is effectively weighted by inverse output std, using the dataset running statistics (`RunningMeanStd`).

Extra reported metrics:
- The trainer also converts \(\hat{y}_t\) back into predicted \(\hat{x}_{t+1}\) and reports per-state MSE and norms in **state space** for interpretability.

---

## Model architecture

### High-level structure

`ModelMixedInput` is a “feature extractor + backbone + head” model:

1. **Input encoders**: currently only `low_dim` is used. The configured low-dim inputs are concatenated and passed through an MLP encoder (which can be identity if `layer_sizes: []`).
2. **Transformer backbone** (optional): if `network.transformer` exists, the model applies a GPT-style causal transformer to the per-timestep features.
3. **Prediction head**: an MLP (`MLPDeterministic`) maps features to the output prediction dimension.

### What goes into `low_dim`

Inputs are chosen by config `inputs.low_dim`. For example, the Pendulum transformer config uses:

- `states_embedding` (typically identical to `states` for Newton-based integrator)
- `gravity_dir` (direction of gravity; often constant, but included for frame-awareness)

Note:
- `root_body_q` is often stored in datasets, but it is **not used** unless explicitly listed in `inputs.low_dim`.

### Transformer: how it is adapted for learning physics from continuous states

The transformer backbone is implemented in `models/model_transformer.py` and is based on nanoGPT/GPT-2 structure (LayerNorm, causal self-attention, MLP blocks), but is modified to accept **continuous-valued vectors per timestep**, instead of discrete token IDs.

Key edits:

- **Token “embedding” becomes a Linear layer**:
  - Standard GPT uses `nn.Embedding(vocab_size, n_embd)` with integer token IDs.
  - This code uses `nn.Linear(input_dim, n_embd)` where `input_dim` is the encoded feature dimension.
  - This makes each timestep’s feature vector act like a “token”.

- **Positional embedding remains**:
  - `wpe = nn.Embedding(block_size, n_embd)` adds learnable positional embeddings up to `block_size`.
  - This enables the transformer to represent time order and learn time-dependent dynamics.

- **Causal attention**:
  - The transformer is causal (`is_causal=True` for flash attention, or a causal mask otherwise).
  - This enforces that timestep \(t\) can only attend to \(\le t\), consistent with forward dynamics modeling.

- **Output projection returns features**, not logits over a vocabulary:
  - `lm_head` maps `n_embd -> n_embd` and the model returns `output_features` (no cross-entropy).
  - Final state-prediction is produced by a separate MLP head in `ModelMixedInput`.

In short: the transformer is used as a **sequence model over continuous state features**. It learns temporal dependencies that can capture inertial/latent dynamics (history dependence) when the system is partially observable from a single state snapshot or when the chosen prediction target benefits from context.

---

## Sequence learning: how time context is presented to the transformer

### Dataset-side sequences

When using `SequenceModelTrainer`, the training dataset is `TrajectoryDataset`, which returns fixed-length windows:

- `states`: shape `(B, T, state_dim)` after collation
- `next_states`: shape `(B, T, state_dim)`
- plus any additional keys present in the HDF5 file (e.g., `gravity_dir`, `root_body_q`)

### Integrator-side input processing

The Newton-based integrator (`newton_based_integrator_neural.py`) ensures:

- inputs are in `(B, T, D)` form (`_ensure_bt`)
- `states_embedding` exists (often identical to `states`)
- angle wrapping is applied consistently to `states` and `next_states`

The trainer calls `neural_integrator.process_neural_model_inputs(data)` before running the model forward pass.

---

## Dataset expectations (HDF5)

The training pipeline expects an HDF5 file with group `data` and attribute `mode`:

- `mode == "trajectory"` for transformer/RNN sequence training (`TrajectoryDataset`)
- `mode == "transition"` or `"trajectory"` for `BatchTransitionDataset` (but transformer training typically uses `"trajectory"`)

For “states-only” training, the minimal required keys are:

- `states`
- `next_states`

Optional keys that can be used if configured as inputs:

- `gravity_dir` (used if listed in `inputs.low_dim`)
- `root_body_q` (only used if listed in `inputs.low_dim`)

Contacts and actuation are currently not part of the training signal unless explicitly added to the dataset and to `inputs.low_dim`.

---

## Practical notes / gotchas (current codebase state)

- **Hidden actuation risk in “states-only” datasets**:
  - The Pendulum trajectory sampler can step the simulator using randomly sampled `joint_acts`.
  - If you generate “states-only” datasets without `passive=True`, the dynamics depend on actuation but the dataset does not store it — making the mapping `states -> next_states` ambiguous.

- **`root_body_q` is commonly saved but not used**:
  - Unless `root_body_q` is added to `inputs.low_dim`, it does not influence training.

- **Config fields from the Warp pipeline may be misleading in the Axion/Newton pipeline**:
  - Fields like `states_frame` / `anchor_frame_step` exist in configs but are not used by the Newton-based integrator (no frame conversion there).

- **Rollout evaluation may not be evaluating the neural dynamics**:
  - `NeuralSimEvaluator` uses `neural_env.step(..., env_mode="neural")`.
  - The current Axion env adapter always steps the ground-truth Axion engine, so “neural rollouts” may not differ from ground-truth in evaluation unless the adapter is extended to actually step with the learned model.

---

## Where to look in code (entrypoints)

- **Train entrypoint**: `src/axion/neural_solver/train/train.py`
- **Transformer-enabled config example**: `src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml`
- **Model definition**: `src/axion/neural_solver/models/models.py` and `src/axion/neural_solver/models/model_transformer.py`
- **Loss + optimization**: `src/axion/neural_solver/algorithms/vanilla_trainer.py`
- **Sequence dataset**: `src/axion/neural_solver/utils/datasets.py` (`TrajectoryDataset`)
- **Target construction + wrapping**: `src/axion/neural_solver/integrators/newton_based_integrator_neural.py`

