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
- creates `AxionEnvToTrajectorySamplerAdapter()` neural_env
- creates `SequenceModelTrainer()` and hands the Axion adapter to it

## Transformer model definition: `models/model_transformer.py`

TO-DO

## Trainer file: `algorithms/sequence_model_trainer.py`
Purpose: 

The `SequenceModelTrainer` class has the following methods:
- **Constructor** <br> 
Saves the `neural_env` instance and its integrator_neural. Creates the torch network module (the transformer) by constructing an instance of the `ModelMixedInput` class, but only if `model_checkpoint_path` is not None. Initializes various training-related attributes, such as `num_epochs`, `num_iters_per_epoch`, `batch_size` (most of them were read form [transformer.yaml](../train/cfg/Pendulum/transformer.yaml))

- **`get_datasets(...)`**<br>
Creates the `TrajectoryDataset` class instances for training and possibly validation datasets. On init, the `TrajectoryDataset`reads and processes the hdf5 dataset file via `h5py`.

- **`compute_dataset_statistics(...)`**<br>
Computes the mean and std of the input and output of the dataset.

- **`get_scheduled_learning_rate(...)`**<br>

- **`preprocess_data_batch`**<br>
Ensures the model inputs have correct shape (B,T,dim) and that the anlges get wrapped, via `neural_integrator`'s `proccess_neural_model_inputs(...)` method. Also convertes next_states to desired model predictions (= state differences).

- **`compute_loss(...)`**<br>

- **`train(...)`**<br>

- **`eval(...)`**<br>

- **`one_epoch(...)`**<br>
Calls either `train(...)` or `eval(...)`.

- **`test(...)`**<br>

- **`save_model(...)`**<br>

