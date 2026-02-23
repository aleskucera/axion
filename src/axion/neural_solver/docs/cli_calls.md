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

## Misc

### Copy from remote to loca
```
scp mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/datasets/Pendulum/pendulumTrainStatesOnly100kenvs100Seed0.hdf5 ~/Downloads/
```