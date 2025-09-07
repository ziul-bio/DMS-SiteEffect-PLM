#!/usr/bin/env python3 -u

###################### To run this script ##################################

# python scripts/OLS.py -m data/nonviral/metadata/BLAT_ECOLX_Ranganathan2015.csv -o experiments/OLS/nonviral/pool_split/BLAT_ECOLX_Ranganathan2015.csv   
    

################ imports #####################
import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from statsmodels.formula.api import ols



###################### Define Functions #######################

def split_mut_sit(mutation):
    parts = re.match(r'([A-Za-z])(\d+)(\S)', mutation) # \S for any non- white space
    if parts:
        return list(parts.groups())
    else:
        return [None, None, None]
    

def data_prep(path_meta_data):
    scaler = StandardScaler()
    data = pd.read_csv(path_meta_data)
    sample_size = data.shape[0] 
    protein_length = len(data['sequence'][0])
    data['target'] = scaler.fit_transform(data['target'].to_frame()).squeeze()
    data[['wt', 'site', 'mut']] = data['mutant'].apply(lambda x: pd.Series(split_mut_sit(x)))

    # this will garantee we have at least two of each class
    common_sites = data['site'].value_counts()[data['site'].value_counts() > 4].index
    data = data[data['site'].isin(common_sites)]
   
    return data, sample_size, protein_length



def run_regression(data, sample_size, protein_length):
    '''this version computes y_pred for train and test sets'''
    # Initialize lists for storing results
    folds = []
    r2s_train, maes_train, rmses_train = [], [], []
    r2s_test, maes_test, rmses_test = [], [], []
    rhos_train, rhos_test = [], []

    for fold, rep in enumerate([1, 2, 3], start=1):
        # seed is equal to sample size + protein length * rep
        seed = (sample_size + protein_length) * rep
        
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['site'])

        # Fit the OLS model using formula interface
        model = ols('target ~ site', data=train_data).fit()

        # Make predictions
        y_train = train_data['target']
        y_test = test_data['target']
        y_pred_train = pd.DataFrame(model.predict(train_data))
        y_pred_test = pd.DataFrame(model.predict(test_data))

        # Evaluate the model
        r2_train = metrics.r2_score(y_train, y_pred_train)
        mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
        mse_train = metrics.mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        rho_train, p_value_train = spearmanr(y_train, y_pred_train)

        r2_test = metrics.r2_score(y_test, y_pred_test)
        mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
        mse_test = metrics.mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        rho_test, p_value_test = spearmanr(y_test, y_pred_test)

        # Append results
        r2s_train.append(r2_train)
        maes_train.append(mae_train)
        rmses_train.append(rmse_train)

        r2s_test.append(r2_test)
        maes_test.append(mae_test)
        rmses_test.append(rmse_test)

        rhos_train.append(rho_train)
        rhos_test.append(rho_test)

        folds.append(fold)

        # Return the collected results
        print(f"    Results:  fold {fold}, Seed: {seed}, r2_train: {r2_train:.3f}, r2_test: {r2_test:.3f}")
    print(f"Results:  r2_train: {np.mean(r2s_train):.2f}, r2_test: {np.mean(r2s_test):.2f}")
    return r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds


def save_results(r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds):
    # Create dictionary for results
    res_dict = {
        "Model": ['Lasso'] * 3,
        "Fold": folds,
        "R2_score_train": r2s_train,
        "MAE_score_train": maes_train,
        "RMSE_score_train": rmses_train,
        "R2_score_test": r2s_test,
        "MAE_score_test": maes_test,
        "RMSE_score_test": rmses_test,
        "rho_score_train": rhos_train,
        "rho_score_test": rhos_test,
    }

    # Convert results to DataFrame
    results = pd.DataFrame(res_dict).reset_index(drop=True)
    return results






############################# Run Predictions #############################

def main():
    parser = argparse.ArgumentParser(description="Run regression for different target datasets and layers")
    parser.add_argument("-m", "--metadata", type=str, help="Target name in the metadata")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file")
    args = parser.parse_args()
    
    # Define the target name and output file
    path_meta_data = args.metadata
    output = args.output

    output_dir = os.path.dirname(output)  # Get the directory path
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    # prepare data
    data, sample_size, protein_length = data_prep(path_meta_data)

    # run regression
    r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds = run_regression(data, sample_size, protein_length)
    results = save_results(r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds)
    results.to_csv(output)
    print(f'Process Finished!')

   
if __name__ == "__main__":
    main()