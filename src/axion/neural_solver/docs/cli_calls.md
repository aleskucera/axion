## Create a Dataset (example)
```
python src/axion/neural_solver/generate/simple_generate_dataset_pendulum.py --env-name Pendulum --num-transitions 10000 --dataset-name pendulumDatasetName.hdf5 --trajectory-length 100 --num-envs 2 --seed 0 --passive
```

## Training

### Begin training (example)
```
python src/axion/neural_solver/train/train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/ 
```

### Visualize in tensorboard (example)
```
tensorboard --logdir src/axion/neural_solver/train/trained_models/02-18-2026-15-25-04
```
