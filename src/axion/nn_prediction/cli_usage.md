python example_usage.py \
    --model-path <path-to-model.pt> \
    --cfg-path <path-to-cfg.yaml> \
    --device cuda:0 \          # or 'cpu'
    --num-steps 10             # number of prediction steps

python3 src/axion/nn_prediction/example_usage.py --model-path src/axion/nn_prediction/trained_models/NeRD_pretrained/pendulum/model.pt --cfg-path src/axion/nn_prediction/trained_models/NeRD_pretrained/pendulum/cfg.yaml