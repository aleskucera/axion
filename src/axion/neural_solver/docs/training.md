*Note: This markdown file was written by hand, old-fashioned style. Well, most of it at least.*

# Training of the transformer Torch module
This file describes each necessary file that is used during the training process.

## Entry point: `train/train.py`
This script:
- reads the training config in [transformer.yaml](../train/cfg/Pendulum/transformer.yaml)
- parses `--cfg` and `--logdir` CLI arguments
- initates training

Training shall be initiated from CLI, for example:
```
python src/axion/neural_solver/train/train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/ 
```

The script:
- creates `NnTrainingInterface()` neural_env
- creates `algo = SequenceModelTrainer()` and hands the Axion adapter to it
- finally, runs:
```
algo.train()
```

## Trainer file: `algorithms/sequence_model_trainer.py`
Purpose: 

The `SequenceModelTrainer` class has the following methods:
- **Constructor** <br> 
Saves the `neural_env` instance and its `utils_provider`. Creates the torch network module (the transformer) by constructing an instance of the `ModelMixedInput` class, but only if `model_checkpoint_path` is not None. Initializes various training-related attributes, such as `num_epochs`, `num_iters_per_epoch`, `batch_size` (most of them were read form [transformer.yaml](../train/cfg/Pendulum/transformer.yaml))

- **`get_datasets(...)`**<br>
Creates the `TrajectoryDataset` class instances for training and possibly validation datasets. On init, the `TrajectoryDataset` reads and processes the hdf5 dataset file via `h5py`. Training can use a single HDF5 path or multiple: set `train_dataset_path` in the config to a list of paths (e.g. `[path/len500.hdf5, path/len2000.hdf5]`) to train on several datasets without merging them; they are combined with PyTorch's `ConcatDataset`.

- **`compute_dataset_statistics(...)`**<br>
Computes the mean and std of the input and output of the dataset.

- **`get_scheduled_learning_rate(...)`**<br>

- **`preprocess_data_batch`**<br>
Ensures the model inputs have correct shape (B,T,dim) and that the angles get wrapped, via `utils_provider`'s `process_neural_model_inputs(...)` method. Also converts next_states to desired model predictions (= state differences).

- **`compute_rollout_loss(...)`**<br>
Differentiable autoregressive rollout loss. Unrolls the model for `rollout_loss_horizon` steps from the first GT state. At each step the model receives its own predicted state (plus GT auxiliary inputs like contacts and gravity), so gradients flow through the full chain via BPTT. Returns MSE between the predicted rollout and GT next_states. Enabled with `use_rollout_loss: true` in config; controlled by `rollout_loss_weight` and `rollout_loss_horizon`.

- **`compute_loss(...)`**<br>

- **`train(...)`**<br>

- **`eval(...)`**<br>

- **`one_epoch(...)`**<br>
Calls either `train(...)` or `eval(...)`.

- **`test(...)`**<br>

- **`save_model(...)`**<br>


## ModelMixedInput definition: `models/models.py`
This script contains the definition of the `ModelMixedInput` torch module. It is a **sequence-to-output** model that encodes mixed inputs (= low dimensional states) with per-input MLP encoders into a single feature vector per timestep.

custom MLP encoders ---> GPT ---> MLP head

Top-down architecture overview:
```
class ModelMixedInput
 - input encoders (MLPBase object from models/base_models.py)
 - transformer (GPT object from models/model_transformer.py)
 - MLPDeterministic 

class MLPDeterministic:
..
```

## Transformer model definition: `models/model_transformer.py`

This file includes the architecture of a transformer model, adapted from GPT-2/nanoGPT. It has ~2.7 M trainable parameters.

Top-down architecture overview:
```
class GPT:
 - Linear layer
 - Embedding layer
 - Dropout layer
 - list of 'Block' modules
 - LayerNorm (class)
 - Linear layer - head

class Block:
 - LayerNorm
 - CasualSelfAttention (class)
 - LayerNorm 
 - MLP (class)

class LayerNorm:
...

class MLP:
...

class CasualSelfAttention:
...

```