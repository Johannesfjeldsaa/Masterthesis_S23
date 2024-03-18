import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

### Summarizing Classification Results ###

# Helper function for summary functions
def summarize_classification_and_add_dfrow(y_pred, y_test, indx, seed, df_summaryallseeds):
    """
    Function to summarize a singel classification results and add a row to the summary dataframe.

    Parameters:
    - y_pred (np.array): The predicted labels
    - y_test (np.array): The true labels
    - indx (int): The index of the row to add to the summary dataframe
    - seed (int): The random seed
    - df_summaryallseeds (pd.DataFrame): The summary dataframe to add the new row to

    Returns:
    - pd.DataFrame: The summary dataframe with the new row added
    """

    summary_dict = {'seed': seed}
    summary_dict['accuracy'] = accuracy_score(y_true=y_test, y_pred=y_pred)
    summary_dict['error'] = 1 - summary_dict['accuracy']
    summary_dict['precision'], summary_dict['recall'], summary_dict['f1-score'], _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average='binary', zero_division=np.nan)
    summary_dict['support'] = len(y_test)
    
    for key, value in summary_dict.items():
        if key not in  ['support', 'seed']:
            summary_dict[key] = round(value, 2)
    
    new_row = summary_dict.values()
    df_summaryallseeds.loc[indx] = list(new_row)
    
    return df_summaryallseeds

def summarize_across_seeds(target_summaries_per_year_and_feature_comb):
    """
    Function to summarize classification results across multiple seeds but for a single feature combination and year.

    Parameters:
    - target_summaries_per_year_and_feature_comb (pd.DataFrame): The classification results for each seed in the form of a 
                                                                 dataframe with columns for seed, y_true (vector) and y_pred (vector)

    Returns:
    - pd.DataFrame: The classification summary for the feature combination and year
    """
    seeds = target_summaries_per_year_and_feature_comb['seed'].to_list()
    df_summaryallseeds = pd.DataFrame(columns=['seed', 'accuracy', 'error', 'precision', 'recall', 'f1-score', 'support'], 
                                        index=range(len(seeds)))
    
    for indx, seed in tqdm(enumerate(seeds), desc="Processing seeds", colour='green', leave=False):
        y_pred = target_summaries_per_year_and_feature_comb['y_pred'].loc[indx]
        y_test = target_summaries_per_year_and_feature_comb['y_true'].loc[indx]
        
        df_summaryallseeds = summarize_classification_and_add_dfrow(y_pred, y_test, indx, seed, df_summaryallseeds)
    
    return df_summaryallseeds

def summarize_with_df(target_summaries):
    """
    Function to summarize classification results across multiple seeds, feature combinations and years.

    Parameters:
    - target_summaries (dict): The classification results in nested dictionaries. First layer is feature combination keys, second layer is years. 
                               In the second layer the classification results are stored as dataframes with columns for seed, y_true (vector) and y_pred (vector)
    
    Returns:
    - dict: The classification summaries for each feature combination and year
    """

    classification_summaries = {}
    for feature_comb_key, feature_comb in target_summaries.items():
        classification_summaries[feature_comb_key] = {}
        for year, target_summaries_per_year_and_feature_comb in feature_comb.items():
            classification_summaries[feature_comb_key][year] = summarize_across_seeds(target_summaries_per_year_and_feature_comb)
    
    return classification_summaries

