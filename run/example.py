#author:LiangZhang@NREL
import pandas as pd
import fsbe.base as fsbe

# Example Run
inputs_output_example = pd.read_csv('../data/example_data.csv')
inputs_example = inputs_output_example.iloc[:, 0:-1]
output_example = inputs_output_example.iloc[:, -1]

selected_inputs_dataframe = fsbe.hybrid_feature_selection_with_filter_and_wrapper(
    inputs = inputs_example,
    output = output_example,
    exemption_variable_list = ['feature_9_exemption'],
    time_lag_variable_list = ['feature_10_time_lag_test_1','feature_11_time_lag_test_2'],
    time_lag_steps = 2,
    filter_method_irrelevancy_threshold = 0.1,
    filter_method_redundancy_threshold = 0.1,
    only_return_inputs_names = False,
    print_procedures = True)

print(selected_inputs_dataframe)
