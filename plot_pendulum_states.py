from axion.nn_prediction.utils.analysis_utils import plot_states_from_csv, plot_model_input_from_csv

#csv_filepath = "src/axion/nn_prediction/pendulum_states_axion_example_usage.csv"
#csv_filepath = "src/axion/core/pendulum_states_NerdEngine.csv"
csv_filepath = "src/axion/nn_prediction/pendulum_model_inputs.csv"

#plot_states_from_csv(csv_filepath)
plot_model_input_from_csv(csv_filepath, "root_body_q", 250)