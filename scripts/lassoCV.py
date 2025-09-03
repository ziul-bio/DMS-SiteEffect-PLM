#!/usr/bin/env python3 -u

###################### To run this script ##################################

# python scripts/lassoCV.py -e embeddings/esm2_650m/nonviral/BLAT_ECOLX_Ranganathan2015.pt -m data/nonviral/metadata/BLAT_ECOLX_Ranganathan2015.csv -o experiments/lassoCV/esm2_650m/nonviral/BLAT_ECOLX_Ranganathan2015.csv    
    

################ imports #####################
import os
import scipy
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import spearmanr

# to ignore the convergence warnings and Rho computeation warnings. 
# Use with caution. Only use when you are sure that the model is working fine.
import warnings
warnings.filterwarnings('ignore') 
from sklearn.exceptions import ConvergenceWarning


###################### Define Functions #######################
def data_prep(path_compressed_embed_file, path_meta_data):
    '''Run regression on compressed embeddings'''
    
    scaler = StandardScaler()
    meta_data = pd.read_csv(path_meta_data)
    sample_size = meta_data.shape[0] 
    protein_length = len(meta_data['sequence'][0])
    meta_data['target'] = scaler.fit_transform(meta_data['target'].to_frame()).squeeze()
    
    # load and merge the data with features
    embed = torch.load(path_compressed_embed_file, weights_only=True)
    embed_df = pd.DataFrame.from_dict(embed).T.reset_index()
    embed_df.rename(columns={'index': 'ID'}, inplace=True)

    data = meta_data.merge(embed_df, how='inner', left_on='ID', right_on='ID')
    
    target = data['target']
    features = data.iloc[:, meta_data.shape[1]:]

    
    return features, target, sample_size, protein_length



def run_regression(features, target, sample_size, protein_length):
    '''this version computes y_pred for train and test sets'''
    # Initialize lists for storing results
    folds, num_nonzero_coefs = [], []
    r2s_train, maes_train, rmses_train = [], [], []
    r2s_test, maes_test, rmses_test = [], [], []
    rhos_train, rhos_test = [], []

    #for fold, seed in enumerate([98, 374, 1234], start=1):
    for fold, rep in enumerate([1, 2, 3], start=1):
        # seed is equal to sample size + protein length * rep
        seed = (sample_size + protein_length) * rep
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=seed)

        # Define and train the regression model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = LassoCV(max_iter=10000, tol=1e-4, n_jobs=-1)
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

            # Return the collected results
            print(f"Results:  fold {fold}, Seed: {seed}, r2_train: {r2_train:.3f}, r2_test: {r2_test:.3f}, Num coefs: {num_nonzero_coef}")
    print(f"Results:  r2_train: {np.mean(r2s_train):.2f}, r2_test: {np.mean(r2s_test):.2f}, Num coefs: {np.mean(num_nonzero_coefs):.2f}")
    return r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds, num_nonzero_coefs


def save_results(r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds, num_nonzero_coefs):
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
        "num_nonzero_coefs": num_nonzero_coefs
    }

    # Convert results to DataFrame
    results = pd.DataFrame(res_dict).reset_index(drop=True)
    return results






############################# Run Predictions #############################

def main():
    parser = argparse.ArgumentParser(description="Run regression for different target datasets and layers")
    parser.add_argument("-e", "--embed", type=str, help="Path to the input file")
    parser.add_argument("-m", "--metadata", type=str, help="Target name in the metadata")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file")
    args = parser.parse_args()
    
    # Define the target name and output file
    path_compressed_embed_file = args.embed
    path_meta_data = args.metadata
    output = args.output

    output_dir = os.path.dirname(output)  # Get the directory path
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    # prepare data
    features, target, sample_size, protein_length = data_prep(path_compressed_embed_file, path_meta_data)

    # run regression
    r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds, num_nonzero_coefs = run_regression(features, target, sample_size, protein_length)
    results = save_results(r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds, num_nonzero_coefs)
    results.to_csv(output)
    print(f'Process Finished!')

   
if __name__ == "__main__":
    main()