## Create a Dataset 

### States only
```
python src/axion/neural_solver/generate/simple_generate_dataset_pendulum.py --env-name Pendulum --num-transitions 10000 --dataset-name pendulumDatasetName.hdf5 --trajectory-length 100 --num-envs 2 --seed 0 --passive --device cuda:1
```
### States + contacts
```
python src/axion/neural_solver/generate/generate_dataset_pendulum.py --env-name Pendulum --num-transitions 10000 --dataset-name pendulumDatasetName.hdf5 --trajectory-length 100 --num-envs 2 --seed 0 --passive --device cuda:1
```
### States + contacts fields, but **no tilted contact plane** (plane_coefficients are zeros)
```
python src/axion/neural_solver/generate/generate_dataset_pendulum.py --env-name Pendulum --num-transitions 10000 --dataset-name pendulumDatasetName.hdf5 --trajectory-length 100 --num-envs 2 --seed 0 --passive --device cuda:1 --without-contacts
```
Generated HDF5 includes:
- converted NN contact fields (`contact_normals`, `contact_points_0/1`, `contact_depths`, `contact_thicknesses`)
- pre-conversion batched Axion contact arrays under `data/axion_contacts/*` for reconstruction/debug.

## Training

### Begin training (example)
```
python src/axion/neural_solver/train/train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/
```
optionally: `--device cuda:1`, `--checkpoint /path/to/checkpoint.pt`, `--no-time-stamp`

### Begin lambda classifier training (example)
```
python src/axion/neural_solver/train/train_lambda_network.py --cfg src/axion/neural_solver/train/cfg/Pendulum/lambdaNetwork.yaml --logdir src/axion/neural_solver/train/trained_models/lambda_classifier
```

### Begin velocity+lambda residual training (example)
```
python src/axion/neural_solver/train/train_vel_and_lambda_network.py --cfg src/axion/neural_solver/train/cfg/Pendulum/velAndLambdaNetwork.yaml --logdir src/axion/neural_solver/train/trained_models/vel_and_lambda_residual
```

### Begin MTL training (regression + lambda classification, example)
```
python src/axion/neural_solver/train/train_mtl.py --cfg src/axion/neural_solver/train/cfg/Pendulum/mtlNetwork.yaml --logdir src/axion/neural_solver/train/trained_models/mtl
```

### Visualize training (wandb)
View runs at [wandb.ai](https://wandb.ai) in your project (e.g. `neural-solver-transformer`). 
Ensure `wandb.login()` runs before training (see `train.py`).

### W&B sweeps
Create a new sweep and start an agent:
```
python src/axion/neural_solver/train/sweep_train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/sweep1 --project neural-solver-transformer
```

Join an existing sweep (replace `<SWEEP_ID>`):
```
python src/axion/neural_solver/train/sweep_train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/ --project neural-solver-transformer --sweep_id <SWEEP_ID>
```

Limit how many runs this agent executes:
```
python src/axion/neural_solver/train/sweep_train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/ --project neural-solver-transformer --sweep_id <SWEEP_ID> --count 10
```

## Testing
Test a neural module that has already undergone training:
```
python src/axion/neural_solver/train/train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --test --checkpoint src/axion/neural_solver/train/trained_models/03-01-2026-10-46-58/nn/final_model.pt --no-time-stamp --logdir ./eval_logs
```

Run a cli testing script that loads the model.pt into NeuralPredictor class and steps it:
```
python src/axion/neural_solver/standalone/test_trained_model_cli.py --model-path src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/nn/best_eval_model.pt --cfg-path src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/cfg.yaml 
```

## Dataset postprocessing
### Add lambda activity labels
```
python src/axion/neural_solver/utils/add_lambda_activity_labels.py --input src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1.hdf5 --output src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1_with_activity.hdf5
```

Optional multiclass labels (`0/1/2`) based on `abs(next_lambdas - lambdas)` with cutpoints `0.1` and `1000`:
```
python src/axion/neural_solver/utils/add_lambda_activity_labels.py --input <...>.hdf5 --output <...>_with_activity.hdf5 --multiclass
```

## Misc

### Copy from remote to local (called on local)
```
scp mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/datasets/Pendulum/pendulumTrainStatesOnly100kenvs100Seed0.hdf5 /home/maros/axion/src/axion/neural_solver/datasets/Pendulum
```

```
scp -r mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/train/trained_models/02-25-2026-13-07-56 /home/maros/axion/src/axion/neural_solver/train/trained_models
```
### Copy from to local to remote (called on local)
```
scp /home/maros/axion/src/axion/neural_solver/datasets/Pendulum/pendulumValidMtlNoContacts2000len500envs2seed1LabelsTh05.hdf5 mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/datasets/Pendulum/pendulumValidMtlNoContacts2000len500envs2seed1LabelsTh05.hdf5
```

### Monitoring GPUs/CPU
one time:
```
nvidia-smi
```

continuous monitoring
```
watch -n 1 nvidia-smi
```

```
nvtop?
```

cpu continuous monitoring
```
top
```