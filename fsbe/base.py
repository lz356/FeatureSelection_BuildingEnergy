#author:LiangZhang@NREL
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor

def hybrid_feature_selection_with_filter_and_wrapper(inputs,
                                                     output,
                                                     exemption_variable_list = [],
                                                     time_lag_variable_list = [],
                                                     time_lag_steps = 0,
                                                     filter_method_irrelevancy_threshold = 0.1,
                                                     filter_method_redundancy_threshold = 0.1,
                                                     only_return_inputs_names = False,
                                                     print_procedures = True):
    """
    The function conducts (1) feature pre-processing (not including domain-knowledge-based feature extraction),
    (2) filter method, and (3) wrapper method. To be more specific, Step 1 creates additional time-lag features, Step 2
    uses filter method, Minimum redundancy maximum relevance (MRMR), to coarsely select feature, and Step 3 uses forward
    wrapper method based on random forest algorithm. Please cite the following paper if used for publication purpose:

    Zhang, L., & Wen, J. (2019). A systematic feature selection procedure for short-term data-driven building energy
    forecasting model development. Energy and Buildings, 183, 428-442.

    Before using this function, pre-processing inputs/output data is necessary. Pre-processing is not included in this
    module so the data need cleaning, imputation, and normalization if necessary. The input is a m*n DataFrame, whose
    headers are inputs/sensors names. Output is an m*1 Series (building energy consumption). This module does not
    include domain-knowledge-based feature extraction, such as complex weather variables (e.g. Climate Z) and virtual
    cooling rate(e.g. chilled water flow rate * (chilled water outlet temperature - chilled water inlet temperature)).
    If domain-knowledge-based feature extraction is needed, calculate the extracted inputs and include them into the
    inputs before using this module.

    :param inputs: DataFrame; input DataFrame with input/sensor names as column headers
    :param output: Series; output Series (normally building energy) with the same row numbers as the inputs
    :param exemption_variable_list: list of strings; default value: []; a list of input names that are exempted
            from the selection and have to be selected in the final feature set
    :param time_lag_variable_list: list of strings; default value: []; a list of input names that needs lag processing,
            based on which time-lag feature are generated in Step 1
    :param time_lag_steps: int; default value: 0; integer that indicates how many time lag steps to generate the
            time-lag variables, based on which time-lag feature are generated in Step 1
    :param filter_method_irrelevancy_threshold: float; default value: 0.1; proportion of inputs that will be
            removed because of high irrelevancy
    :param filter_method_redundancy_threshold: float; default value: 0.1; proportion of inputs that will be
            removed because of high redundancy
    :param only_return_inputs_names: bool; default value: False; indicating whether the list of final input names or
            the DataFrame of final inputs is the return value of this function
    :param print_procedures: bool; default value: True; indicating whether the text of processing procedures are printed

    :return: final selected feature name list or DataFrame consists of final selected features
    """

    if print_procedures:
        print('Step 0. Prepare inputs and output data in DataFrame. Pre-processing is not included in this module \
              so the data need cleaning, imputation, and normalization if necessary. The input is a m*n DataFrame, \
              whose headers are inputs/sensors names. Output is an m*1 Series (building energy consumption). This \
              module does not include domain-knowledge-based feature extraction, such as complex weather variables \
              (e.g. Climate Z) and virtual cooling rate \
              (e.g. chilled water flow rate * (chilled water outlet temperature - chilled water inlet temperature)).\
              If domain-knowledge-based feature extraction is needed, calculate the extracted inputs and include \
              them into the inputs before using this module.')

    if print_procedures:
        print('Step 1. Pre-processing using domain knowledge: creating time-lag variables')

    if time_lag_variable_list != []:
        time_lag_variable_df = inputs[time_lag_variable_list]

        for i in range(1,1 + time_lag_steps):
            df_temp = time_lag_variable_df.shift(periods=i, freq=None, axis=0).bfill(axis=0)
            inputs = inputs.join(df_temp, how='left', rsuffix=f'_t-{i}')

    if print_procedures:
        print('Step 2. Filter method with Minimum Redundancy Maximum Relevance (MRMR)')

    # remove irrelevancy
    number_of_features_removed_by_irrelevancy = round(filter_method_irrelevancy_threshold * len(inputs.columns))
    inputs_output = pd.concat([inputs, output], axis = 1)
    cor = inputs_output.corr()
    cor_target = abs(cor[inputs_output.columns[-1]])[0:-1]
    relevant_features = cor_target.sort_values(ascending=False)[
                        0:len(inputs) - number_of_features_removed_by_irrelevancy -1]
    inputs = inputs_output[relevant_features.index.tolist()]

    # remove redundancy
    number_of_features_removed_by_redundancy = round(filter_method_redundancy_threshold * len(inputs.columns))
    for i in range(0,number_of_features_removed_by_redundancy):
        temp_corr = inputs.corr().abs()

        s = pd.Series(data=[1] * len(inputs.columns), index=inputs.columns)
        df_diagonal = pd.DataFrame(0, index=s.index, columns=s.index, dtype=s.dtype)
        np.fill_diagonal(df_diagonal.values, s)

        temp_corr = (temp_corr - df_diagonal)

        var_1 = temp_corr.max().sort_values(ascending=False).index[0]
        var_2 = temp_corr.max().sort_values(ascending=False).index[1]

        if inputs[var_1].corr(output) < inputs[var_2].corr(output):
            inputs = inputs.drop(columns=[var_1])
        else:
            inputs = inputs.drop(columns=[var_2])



    if print_procedures:
        print('Step 3. Forward wrapper method based on random forest')
    # The random forest here is without hyper-parameter tuning
    random_forest = RandomForestRegressor()
    # Check "SequentialFeatureSelector" on http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/
    sfs = SFS(random_forest,
              k_features = (len(exemption_variable_list)+1, len(inputs.columns)), # range of final selected features
              forward = True,
              floating = False,
              scoring = 'neg_root_mean_squared_error',  # error metrics
              cv = 4,  # four-fold cross-validation
              n_jobs = -1, # use all CPUs to compute
              fixed_features = exemption_variable_list)  # features to be exempted from wrapper method
    sfs = sfs.fit(inputs, output)
    selection_procedure = sfs.subsets_

    if print_procedures:
        print(selection_procedure)

    final_feature_names = sfs.k_feature_names_

    if only_return_inputs_names:
        return final_feature_names
    else:
        #return pd.concat([inputs[list(final_feature_names)], output], axis = 1)
        return inputs[list(final_feature_names)]
