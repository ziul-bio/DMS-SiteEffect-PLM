#!/usr/bin/env python3 -u

###################### To run this script ##################################

# python scripts/reg_LassoCV_sitesplit.py -i embeddings/ViCAM/CRVDBv29_maxLen2046_Full_lr1e6/PA_FLU_Sun2015_embeddings.pt -m data/DMS_mut_metadata/PA_FLU_Sun2015_metadata.csv -o results/lassoCV_sitesplit/vicam_300M/PA_FLU_Sun2015.csv    
    

################ imports #####################
import os
import scipy
import torch
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import random

# to ignore the convergence warnings and Rho computeation warnings. 
# Use with caution. Only use when you are sure that the model is working fine.
import warnings
warnings.filterwarnings('ignore') 
from sklearn.exceptions import ConvergenceWarning


###################### Define Functions #######################

def split_data(df, seed, train_pct=0.8, test_pct=0.2):
    """
    This function randomly splits a dataframe into train, test, and validation data given a seed by mutation site.
    
    Parameters:
     - df (DataFrame): dataframe containing information about mutants. Mutants should be in the order wt amino acid, site of mutation, mutant amino acid. ex "M1F"
     - seed (int): the seed to be used when shuffling sites randomly.
     - train_pct (float): the percentage of data that will be split into the train dataset. Default is 0.8
     - test_pct (float): the percentage of data that will be split into the test dataset. Default is 0.2

    Returns:
     - train_df (DataFrame): the DataFrame containing randomly selected data by site to be used as the train dataset.
     - test_df (DataFrame): the DataFrame containing randomly selected data by site to be used as the test dataset.
     - val_df (DataFrame): the DataFrame containing randomly selected data by site to be used as the val dataset.
    """
    # find sites of mutation and order randomly
    df["site"] = [int(s[1:-1]) for s in df["mutant"]]
    sites = df["site"].unique()
    random.seed(seed)
    random.shuffle(sites)

    if train_pct + test_pct != 1:
        print("Split percentages must sum to 1")
        return

    df_size = df.shape[0]
    df_testpct = df_size*test_pct
    test_sites, train_sites = [], []

    # determine sites for test, then train
    for site in sites:
        if len(test_sites) <= df_testpct:
            test_sites.extend([mut_site for mut_site in df["site"] if mut_site == site])
        else:
            train_sites.extend([mut_site for mut_site in df["site"] if mut_site == site])

    # subset df for train, test data
    train_df = df[df["site"].isin(set(train_sites))]
    test_df = df[df["site"].isin(set(test_sites))]

    return train_df, test_df

def features_scaler(features):
    '''Scale the features by min-max scaler, to ensure that the features selected by Lasso are not biased by the scale of the features'''
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features)



def run_regression(X_train, X_test, y_train, y_test):
    '''this version computes y_pred for train and test sets'''
 
    model = LassoCV(max_iter=1000, tol=1e-2, n_jobs=-1)
    model.fit(X_train, y_train)

    # get the number of non-zero coefficients
    coeficients = model.coef_
    num_nonzero_coef = np.sum(coeficients != 0)

    # Make predictions
    y_pred_train = pd.DataFrame(model.predict(X_train))
    y_pred_test = pd.DataFrame(model.predict(X_test))

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

    return r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test, rho_train, rho_test, num_nonzero_coef


def save_results(folds, r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, num_nonzero_coefs):
    # Create dictionary for results
    res_dict = {
        "Fold": folds,
        "R2_score_train": r2s_train,
        "MAE_score_train": maes_train,
        "RMSE_score_train": rmses_train,
        "R2_score_test": r2s_test,
        "MAE_score_test": maes_test,
        "RMSE_score_test": rmses_test,
        "rho_score_train": rhos_train,
        "rho_score_test": rhos_test,
        "num_zero_coefs": num_nonzero_coefs
    }

    # Convert results to DataFrame
    results = pd.DataFrame(res_dict).reset_index(drop=True)
    return results



def run_regression_on_compressed_files(path_compressed_embed_file, path_meta_data):
    '''Run regression on compressed embeddings'''
    
    meta_data = pd.read_csv(path_meta_data)
    results = pd.DataFrame()

    # load and merge the data with features
    embed = torch.load(path_compressed_embed_file, weights_only=True)
    embed_df = pd.DataFrame.from_dict(embed).T.reset_index()
    embed_df.rename(columns={'index': 'ID'}, inplace=True)

    # Initialize lists for storing results
    folds, num_nonzero_coefs = [], []
    r2s_train, maes_train, rmses_train = [], [], []
    r2s_test, maes_test, rmses_test = [], [], []
    rhos_train, rhos_test = [], []

    for fold, seed in enumerate([374, 98, 20, 8477, 1234], start=1):
        train, test = split_data(meta_data, seed)
        train_data = train.merge(embed_df, how='inner', left_on='ID', right_on='ID')
        test_data = test.merge(embed_df, how='inner', left_on='ID', right_on='ID')
        y_train = train_data['target']
        y_test = test_data['target']

        X_train = features_scaler(train_data.iloc[:, train.shape[1]:])
        X_test = features_scaler(test_data.iloc[:, test.shape[1]:])
    
        # run regression
        r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test, rho_train, rho_test, num_nonzero_coef = run_regression(X_train, X_test, y_train, y_test)

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
        num_nonzero_coefs.append(num_nonzero_coef)

        print(f"Results:  fold {fold}, r2_train: {r2_train:.3f}, r2_test: {r2_test:.3f}, Num coefs: {num_nonzero_coef}")
    
    res = save_results(folds, r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, num_nonzero_coefs)
    results = pd.concat([results, res], axis=0)

    return results


############################# Run Predictions #############################

def main():
    parser = argparse.ArgumentParser(description="Run regression for different target datasets and layers")
    parser.add_argument("-i", "--input", type=str, help="Path to the input file")
    parser.add_argument("-m", "--metadata", type=str, help="Target name in the metadata")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file")
    args = parser.parse_args()
    
    # Define the target name and output file
    path_compressed_embed_file = args.input
    path_meta_data = args.metadata
    output = args.output

    output_dir = os.path.dirname(output)  # Get the directory path
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    print("Starting regression...")
    results = run_regression_on_compressed_files(path_compressed_embed_file, path_meta_data)
    results.to_csv(output)
    print(f'Process Finished!')

   
if __name__ == "__main__":
    main()