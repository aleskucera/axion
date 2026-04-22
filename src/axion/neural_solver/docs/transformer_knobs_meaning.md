# Transformer Knobs — What They Mean

These parameters live under `network.transformer` in the training config and control the
capacity, speed, and regularisation of the GPT-style transformer used as the sequence backbone.

---

## `block_size: 16`

**What it is:** The maximum number of tokens (time-steps) the transformer can attend to at once —
i.e. its context window.  In this project a "token" is one (state, action, contact-info, …) tuple
at a single simulation step.

**Relationship to other knobs:**
`block_size` must be ≥ `num_states_history` (env) and ≥ `sample_sequence_length` (algorithm).
Currently both of those are 10, so 16 gives a small margin.

| Direction | Effect |
|-----------|--------|
| **Increase** | The model can look further back in time. Useful if the system has long-range dependencies (e.g. a pendulum that was perturbed many steps ago still affects the current contact force). Increases memory and compute quadratically with sequence length. |
| **Decrease** | Faster training and lower memory footprint; forces the model to make predictions from a shorter history. May hurt accuracy when the relevant past is longer than the window. |

**Example use-cases:**
- *Increase to 32–64* when training on longer rollout sequences or on systems with slow dynamics (e.g. a heavy pendulum where the transient response lasts 30+ steps).
- *Decrease to 8* for rapid prototyping or when the dataset only contains very short episodes and you want to avoid padding waste.

---

## `n_layer: 6`

**What it is:** The number of stacked transformer blocks (depth of the network).  Each block
contains one self-attention layer + one feed-forward MLP.  More layers → the model can learn
more abstract, hierarchical features from the sequence.

| Direction | Effect |
|-----------|--------|
| **Increase** | Higher model capacity and better representation power. Recommended when the dynamics are complex (multi-body contacts, non-linear coupling). Training time scales roughly linearly. Risk of overfitting on small datasets. |
| **Decrease** | Faster training, smaller model, less risk of overfitting. May underfit on complex dynamics. |

**Example use-cases:**
- *Increase to 8–12* when moving from a simple pendulum to a multi-link articulated robot with many simultaneous contacts.
- *Decrease to 2–3* when the dataset is small (< 50 k samples) or when you need real-time inference on embedded hardware and latency is critical.

---

## `n_head: 12`

**What it is:** The number of parallel attention heads inside each transformer block.  The
embedding `n_embd` is split equally across heads, so each head operates on a
`n_embd / n_head = 192 / 12 = 16`-dimensional subspace.  Different heads can independently
learn to attend to different aspects of the sequence (e.g. one head tracks velocity, another
tracks contact normals).

**Constraint:** `n_embd` must be exactly divisible by `n_head`.

| Direction | Effect |
|-----------|--------|
| **Increase** | More specialised attention patterns per layer; can capture richer relational structure. Must increase `n_embd` proportionally (or decrease head-dim). More compute per layer. |
| **Decrease** | Each head has a wider view of the embedding; fewer specialised patterns. Reduces compute. Head-dim grows, which can also help when the per-head representation needs to be richer. |

**Example use-cases:**
- *Increase to 16 (with n_embd 256)* when you suspect the model needs to simultaneously track many independent physical channels (e.g. 6-DoF contact forces + body velocity + gravity direction).
- *Decrease to 4 or 6* when `n_embd` is small (e.g. 64) to keep head-dim reasonable (≥ 16 is a common rule of thumb).

---

## `n_embd: 192`

**What it is:** The size of the token embedding — the width of every residual stream passing
through the transformer.  All internal projections (Q, K, V, feed-forward) derive their
dimensions from this value.  This is the single biggest lever for overall model capacity.

| Direction | Effect |
|-----------|--------|
| **Increase** | Larger representational capacity; the model can encode more information per time-step. Parameter count grows quadratically. Recommended before increasing `n_layer`. |
| **Decrease** | Smaller, faster model. May create an information bottleneck if the input feature dimensionality is already large relative to `n_embd`. |

**Example use-cases:**
- *Increase to 256 or 384* when adding more input channels (e.g. enabling `enable_lambda_head` and feeding richer contact geometry) and validation loss plateaus.
- *Decrease to 64 or 128* for a lightweight model intended to run inside a real-time controller loop where inference budget is < 1 ms.

---

## `dropout: 0.0`

**What it is:** Fraction of neuron activations randomly zeroed during each forward pass of
training.  Acts as a stochastic regulariser that prevents co-adaptation of neurons.

The comment in the config gives the canonical advice: **0.0 for pre-training (learning from
scratch), 0.1+ for fine-tuning** (adapting a pre-trained model to a new, smaller dataset).

| Direction | Effect |
|-----------|--------|
| **Increase (e.g. 0.1 – 0.3)** | Stronger regularisation; useful when the training dataset is small relative to model size, or when fine-tuning on domain-specific data. Slows convergence. |
| **Decrease / keep at 0.0** | Fastest convergence, appropriate when the dataset is large and the model is not obviously overfitting. |

**Example use-cases:**
- *Set to 0.1* when fine-tuning a pre-trained Pendulum model on a smaller dataset from a
  different pendulum variant (different mass/length) to prevent the model from forgetting its
  general physics knowledge.
- *Keep at 0.0* during the initial large-scale pre-training run where the dataset is 1 M+ samples.

---

## `bias: False`

**What it is:** Whether to include a learnable bias term in the `Linear` projection layers and
`LayerNorm` layers of the transformer.  Disabling bias slightly reduces the parameter count
and, in many modern LLM architectures, has little or no effect on final accuracy.

| Direction | Effect |
|-----------|--------|
| **True** | Adds a small number of extra parameters. Can marginally help when data is limited or the output distributions are strongly offset from zero. |
| **False** (current) | Slightly fewer parameters; more symmetric weight initialisation; matches modern best-practices (GPT-2 style). Typically the right default. |

**Example use-cases:**
- *Switch to True* if you observe that the model's outputs are systematically biased (e.g. consistently over-predicting contact forces) and you have ruled out normalisation issues — the extra bias terms give the network an explicit mechanism to shift its outputs.
- *Keep False* in the default pre-training setup; the `normalize_output: True` flag in the network config already handles output centering.
