## Create a Dataset (example)
```
python src/axion/neural_solver/generate/simple_generate_dataset_pendulum.py --env-name Pendulum --num-transitions 10000 --dataset-name pendulumDatasetName.hdf5 --trajectory-length 100 --num-envs 2 --seed 0 --passive
```

## Training

### Begin training (example)
```
python src/axion/neural_solver/train/train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/ 
```

### Visualize training (wandb)
View runs at [wandb.ai](https://wandb.ai) in your project (e.g. `neural-solver-transformer`). 
Ensure `wandb.login()` runs before training (see `train.py`).

## Testing
Test a neural module that has already undergone training:
```
python src/axion/neural_solver/train/train.py --cfg train/cfg/Pendulum/transformer.yaml --test --checkpoint src/axion/neural_solver/train/trained_models/02-22-2026-16-45-54/nn/best_eval_model.pt --no-time-stamp --logdir ./eval_logs
```

Run a cli testing script that loads the model.pt into NeuralPredictor class and steps it:
```
python src/axion/neural_solver/standalone/test_trained_model_cli.py --model-path src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/nn/best_eval_model.pt --cfg-path src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/cfg.yaml 
```


## Misc

### Copy from remote to local
```
scp mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/datasets/Pendulum/pendulumTrainStatesOnly100kenvs100Seed0.hdf5 ~/Downloads/
```

```
scp -r mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/train/trained_models/02-25-2026-13-07-56 /home/maros/axion/src/axion/neural_solver/train/trained_models
```