#!/usr/bin/env python3 -u

###################### To run this script ##################################

# python scripts/AVG_SS.py -m data/cellular/metadata/BLAT_ECOLX_Ranganathan2015.csv -o experiments/AVG/cellular/site_split/BLAT_ECOLX_Ranganathan2015.csv    
    

################ imports #####################
import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
#from scipy.stats import spearmanr


###################### Define Functions #######################

def split_data(meta_data, seed, train_pct=0.8, test_pct=0.2):
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
    meta_data["site"] = [int(s[1:-1]) for s in meta_data["mutant"]]
    sites = meta_data["site"].unique()
    random.seed(seed)
    random.shuffle(sites)

    if train_pct + test_pct != 1:
        print("Split percentages must sum to 1")
        return

    df_size = meta_data.shape[0]
    df_test_size = df_size*test_pct
    test_sites, train_sites = [], []

    # determine sites for test, then train
    for site in sites:
        if len(test_sites) <= df_test_size:
            test_sites.extend([mut_site for mut_site in meta_data["site"] if mut_site == site])
        else:
            train_sites.extend([mut_site for mut_site in meta_data["site"] if mut_site == site])

    # subset df for train, test data
    train_df = meta_data[meta_data["site"].isin(set(train_sites))]
    test_df = meta_data[meta_data["site"].isin(set(test_sites))]

    return train_df, test_df




def data_prep(path_meta_data):
    scaler = StandardScaler()
    meta_data = pd.read_csv(path_meta_data)
    meta_data['target'] = scaler.fit_transform(meta_data['target'].to_frame()).squeeze()
    meta_data = meta_data.query("mutant != 'WT'")
    return meta_data



def run_regression(meta_data):
    '''this version computes y_pred for train and test sets'''
    # Initialize lists for storing results
    folds = []
    r2s_train, maes_train, rmses_train = [], [], []
    r2s_test, maes_test, rmses_test = [], [], []

    for fold, rep in enumerate([1, 2, 3], start=1):
        # seed is equal to sample size + protein length * rep
        seed = (meta_data.shape[0] + len(meta_data['sequence'][0])) * rep
        
        train_data, test_data = split_data(meta_data, seed)

        # Fit the Average model, this will compute the average effect by site, and if the site does not exist, it applies the overall average.
        model = train_data.groupby(['site'])['target'].mean().to_dict()
        model['0'] = train_data['target'].mean().item()

        # Make predictions
        y_train = train_data['target']
        y_test = test_data['target']
       
        y_pred_train = pd.DataFrame([model.get(str(site), model['0']) for site in train_data['site']])
        y_pred_test = pd.DataFrame([model.get(str(site), model['0']) for site in test_data['site']])

        # Evaluate the model
        r2_train = metrics.r2_score(y_train, y_pred_train)
        mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
        mse_train = metrics.mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        #rho_train, p_value_train = spearmanr(y_train, y_pred_train)

        r2_test = metrics.r2_score(y_test, y_pred_test)
        mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
        mse_test = metrics.mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        #rho_test, p_value_test = spearmanr(y_test, y_pred_test)

        # Append results
        r2s_train.append(r2_train)
        maes_train.append(mae_train)
        rmses_train.append(rmse_train)

        r2s_test.append(r2_test)
        maes_test.append(mae_test)
        rmses_test.append(rmse_test)

        # rhos_train.append(rho_train)
        # rhos_test.append(rho_test)

        folds.append(fold)

        # Return the collected results
        print(f"    Results:  fold {fold}, Seed: {seed}, r2_train: {r2_train:.3f}, r2_test: {r2_test:.3f}")
    print(f"Results:  r2_train: {np.mean(r2s_train):.2f}, r2_test: {np.mean(r2s_test):.2f}")
    return r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, folds


def save_results(r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, folds):
    # Create dictionary for results
    res_dict = {
        "Fold": folds,
        "R2_score_train": r2s_train,
        "MAE_score_train": maes_train,
        "RMSE_score_train": rmses_train,
        "R2_score_test": r2s_test,
        "MAE_score_test": maes_test,
        "RMSE_score_test": rmses_test,
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
    meta_data = data_prep(path_meta_data)

    # run regression
    r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, folds = run_regression(meta_data)
    results = save_results(r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, folds)
    results.to_csv(output)
    print(f'Process Finished!')

   
if __name__ == "__main__":
    main()