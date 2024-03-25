import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

### Summarizing Classification Results ###

# Helper function for summary functions
def summarize_classification_and_add_dfrow(y_train_true, y_train_pred, 
                                           y_test_true, y_test_pred, 
                                           indx, seed, df_summaryallseeds):
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
    summary_dict['accuracy'] = accuracy_score(y_true=y_test_true, y_pred=y_test_pred)
    summary_dict['error'] = 1 - summary_dict['accuracy']
    summary_dict['precision'], summary_dict['recall'], summary_dict['f1-score'], _ = precision_recall_fscore_support(y_true=y_test_true, y_pred=y_test_pred, average='binary', zero_division=np.nan)
    summary_dict['support'] = len(y_test_true)

    summary_dict['training_accuracy'] = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
    summary_dict['training_error'] = 1 - summary_dict['training_accuracy']
    summary_dict['training_precision'], summary_dict['training_recall'], summary_dict['training_f1-score'], _ = precision_recall_fscore_support(y_true=y_train_true, y_pred=y_train_pred, average='binary', zero_division=np.nan)
    summary_dict['training_support'] = len(y_train_true)
    
    for key, value in summary_dict.items():
        if key not in  ['support', 'training_support', 'seed']:
            summary_dict[key] = round(value, 2)
    
    new_row = summary_dict.values()
    df_summaryallseeds.loc[indx] = list(new_row)
    
    return df_summaryallseeds

def summarize_across_seeds(target_summaries_per_year_and_feature_comb):
    """
    Function to summarize classification results across multiple seeds but for a single feature combination and year.

    Parameters:
    - target_summaries_per_year_and_feature_comb (pd.DataFrame): The classification results for each seed in the form of a 
                                                                 dataframe with columns for 
                                                                 seed, y_train_true (vector), y_train_pred (vector), 
                                                                 y_test_true (vector) and y_test_pred (vector)

    Returns:
    - pd.DataFrame: The classification summary for the feature combination and year
    """
    seeds = target_summaries_per_year_and_feature_comb['seed'].to_list()
    df_summaryallseeds = pd.DataFrame(columns=['seed', 
                                               'accuracy', 'error', 'precision', 'recall', 'f1-score', 'support',
                                               'training_accuracy', 'training_error', 'training_precision', 'training_recall', 'training_f1-score', 'training_support'], 
                                      index=range(len(seeds)))
    
    for indx, seed in tqdm(enumerate(seeds), desc="Processing seeds", colour='green', leave=False):
        y_train_true = target_summaries_per_year_and_feature_comb['y_train_true'].loc[indx]
        y_train_pred = target_summaries_per_year_and_feature_comb['y_train_pred'].loc[indx]
        y_test_true = target_summaries_per_year_and_feature_comb['y_test_true'].loc[indx]
        y_test_pred = target_summaries_per_year_and_feature_comb['y_test_pred'].loc[indx]
        
        
        
        df_summaryallseeds = summarize_classification_and_add_dfrow(y_train_true, y_train_pred, 
                                                                    y_test_true, y_test_pred, 
                                                                    indx, seed, df_summaryallseeds)
    
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

def create_plotdata_from_summary(classification_summaries, metric, years, include_train=False):
    """
    Function to create a plot dataframe from the classification summaries.

    Parameters:
    - classification_summaries (dict): The classification summaries for each feature combination and year

    Returns:
    - pd.DataFrame: The plot dataframe
    """
    plot_data = pd.DataFrame(columns=['Year', metric, 'feature_comb_key'])

    for feature_comb_key, classification_summary in classification_summaries.items():
        metric_in_feature_comb = []
        metric_in_feature_comb_training = []
        year_list = []
        for year in years:
            if year in classification_summary.keys():
                metric_in_feature_comb = metric_in_feature_comb + classification_summary[year][metric].tolist()
                year_list = year_list + [year for _ in range(len(classification_summary[year][metric].tolist()))]
                if include_train:
                    metric_in_feature_comb_training = metric_in_feature_comb_training + classification_summary[year][f'training_{metric}'].tolist()
        
        feature_comp_plot_data = pd.DataFrame({'Year': year_list,
                                               metric: metric_in_feature_comb, 
                                               'feature_comb_key': [feature_comb_key for _ in range(len(metric_in_feature_comb))]})
        if include_train:
            feature_comp_plot_data[f'training_{metric}'] = metric_in_feature_comb_training

        plot_data = pd.concat([plot_data, feature_comp_plot_data], ignore_index=True)
    
    return plot_data
