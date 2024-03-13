# This file contains helper functions for running classification experiments and feature selection using Boruta Shap.
# author: Johannes Fjeldså
# email: johannes.larsen.fjeldsa@nmbu.no

# Importing packages
from turtle import title
from unittest import skip
import pandas as pd
import numpy as np
from sympy import plot
from torch import seed
from tqdm import tqdm

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import statistics

from BorutaShap import BorutaShap

import matplotlib.pyplot as plt


# Helper function for summary functions
def summarize_classification_and_add_dfrow(y_pred, y_test, indx, seed, df_summaryallseeds):
    """
    Function to summarize the classification results and add a row to the summary dataframe.

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
    summary_dict['precision'], summary_dict['recall'], summary_dict['f1-score'], _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average='binary')
    summary_dict['support'] = len(y_test)
    
    for key, value in summary_dict.items():
        if key not in  ['support', 'seed']:
            summary_dict[key] = round(value, 2)
    
    new_row = summary_dict.values()
    df_summaryallseeds.loc[indx] = list(new_row)
    
    return df_summaryallseeds

def open_cross_sections(scenario_indx_key=None, years=None, scaled=False):
    scenario_indx_key = scenario_indx_key if scenario_indx_key is not None else {'ssp126': 0, 'ssp585': 1}
    years = years if years is not None else list(range(2015, 2101))

    if scaled:
        cross_sections = {year: pd.read_csv('/'.join(['/nird/home/johannef/Masterthesis_S23 DataFiles/tables/cross_sections/standardscaled', f'cross_section_{year}_standscaled.csv'])) for year in years}
    else:
        cross_sections = {year: pd.read_csv('/'.join(['/nird/home/johannef/Masterthesis_S23 DataFiles/tables/cross_sections/unscaled', f'cross_section_{year}.csv'])) for year in years}

    for year in years:
        cross_section = cross_sections[year]
        cross_section['scenario_indx'] = cross_section['scenario'].map(scenario_indx_key)
        cross_section['scenario_indx'] = cross_section['scenario_indx'].astype('Int64')
        cross_sections[year] = cross_section[cross_section['scenario'].isin(list(scenario_indx_key.keys()))]

    return cross_sections
### RF ###
from sklearn.ensemble import RandomForestClassifier

def tune_RF(X_train, y_train, seed, param_grid=None, rand_search_kwgs=None, return_model=False):
    """
    Function to tune Random Forest hyperparameters using RandomizedSearchCV
    
    Parameters:
    - X_train (pd.DataFrame): The training data
    - y_train (pd.Series): The training labels
    - seed (int) The random seed
    - param_grid (dict): The hyperparameter grid to search over
    - rand_search_kwgs (dict): The RandomizedSearchCV keyword arguments
    - return_model (bool): Wether or not to return the best model or the best hyperparameters

    Returns:
    - RandomForestClassifier: The best model if return_model is True or the best hyperparameters if return_model is False
    """
    
    param_grid = param_grid if param_grid is not None else {
        'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
        'max_depth': [None] + [int(x) for x in np.linspace(10, 110, num = 11)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    rand_search_kwgs = rand_search_kwgs if rand_search_kwgs is not None else {
        'n_iter': 100, # Covers more feature combinations (here 100 out of 4320)
        'cv': 5, # higher num decreases chance of overfitting 
        'verbose': 0,
        'random_state': seed,
        'n_jobs': -1
    }

    rf = RandomForestClassifier(random_state=seed)
    random_search = RandomizedSearchCV(estimator=rf, 
                                       param_distributions=param_grid, 
                                       **rand_search_kwgs)
    
    random_search.fit(X_train, y_train)

    if return_model:
        return RandomForestClassifier(**random_search.best_params_)
    else:
        return random_search.best_params_

def run_RF_across_seeds(X, y, seeds, param_grid, rand_search_kwgs):
    """
    Function to run Random Forest across multiple seeds and summarize the results. 

    Parameters:
    - X (pd.DataFrame): The training data
    - y (pd.Series): The training labels
    - seeds (list): The random seeds to use
    - param_grid (dict): The hyperparameter grid to search over
    - rand_search_kwgs (dict): The RandomizedSearchCV keyword arguments

    Returns:
    - pd.DataFrame: The summary of the classification results (df with one row per seed) and the confusion matrices (list of arrays) for each seed.
    """

    df_summaryallseeds = pd.DataFrame(columns=['seed', 'accuracy', 'error', 'precision', 'recall', 'f1-score', 'support'], 
                                        index=range(len(seeds)))
    cm_all_seeds = []     
    for indx, seed in tqdm(enumerate(seeds), desc="Processing seeds", colour='green', leave=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        
        best_rf_config = tune_RF(X_train, y_train, seed, param_grid, rand_search_kwgs)

        best_rf = RandomForestClassifier(**best_rf_config)
        best_rf.fit(X_train, y_train)
        
        y_pred = best_rf.predict(X_test)
        cm_all_seeds.append(confusion_matrix(y_true=y_test, y_pred=y_pred))
        df_summaryallseeds = summarize_classification_and_add_dfrow(y_pred, y_test, indx, seed, df_summaryallseeds)

    return df_summaryallseeds, cm_all_seeds

### SVM ###
from sklearn.svm import SVC

def tune_SVM(X_train, y_train, seed, param_grid=None, rand_search_kwgs=None, return_model=False):
    pass

def run_SVM_across_seeds(X, y, seeds, param_grid, rand_search_kwgs):
    df_summaryallseeds = pd.DataFrame(columns=['seed', 'accuracy', 'error', 'precision', 'recall', 'f1-score', 'support'], 
                                        index=range(len(seeds)))
    cm_all_seeds = []     
    for indx, seed in tqdm(enumerate(seeds), desc="Processing seeds", colour='green', leave=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        
        best_svm_config = tune_SVM(X_train, y_train, seed, param_grid, rand_search_kwgs)

        best_svm = SVC(**best_svm_config)
        best_svm.fit(X_train, y_train)
        
        y_pred = best_svm.predict(X_test)
        cm_all_seeds.append(confusion_matrix(y_true=y_test, y_pred=y_pred))
        df_summaryallseeds = summarize_classification_and_add_dfrow(y_pred, y_test, indx, seed, df_summaryallseeds)

    return df_summaryallseeds, cm_all_seeds

### XGB ###
from xgboost import XGBClassifier

def tune_XGB(X_train, y_train, seed):
    pass

### Boruta Shap ###

def run_boruta_shap(cross_sections, model_name, years=None, param_grid=None, rand_search_kwgs=None, random_state=0, 
                    save_dfs=True, plot=False, display_plot=False, save_plot=False, plot_fq=10, plot_kwgs=None):
    """
    Function to run Boruta Shap feature selection across multiple years. 

    Parameters:
    - cross_sections (dict): The cross-sectional data as a dictionary of dataframes. Use years as keys and the cross-sectional data as values.
    - model_name (str): The name of the model to use for feature selection. Options are 'RF', 'SVM', 'LR', 'XGB'
    - years (list): The years to process, if None all years in cross_sections will be used.
    - param_grid (dict): The hyperparameter grid to search over
    - rand_search_kwgs (dict): The RandomizedSearchCV keyword arguments
    - random_state (int): The seed to use for random processes
    - plot (bool): Wether or not to plot the feature importance
    - display (bool): Wether or not to display the plot
    - save_plot (bool): Wether or not to save the plot
    - plot_fq (int): The frequency of years to plot
    - plot_kwgs (dict): The plot keyword arguments (https://github.com/Ekeany/Boruta-Shap/blob/master/src/BorutaShap.py)

    Saves:
    - pd.DataFrame: The Boruta Shap median feature importance scores for all years 
    - pd.DataFrame: The Boruta Shap decisions for all years
    """

    years = years if years is not None else list(cross_sections.keys())                                        

    if model_name == 'RF':
        tuner = tune_RF
    elif model_name == 'SVM':
        tuner = tune_SVM
    elif model_name == 'LR':
        tuner = tune_LR
    elif model_name == 'XGB':
        tuner = tune_XGB
    else:
        raise ValueError('Model name not recognized')
    
    features = [col for col in cross_sections[2015].columns if col not in ['scenario', 'scenario_indx']]
    Boruta_scores_df = pd.DataFrame(columns=['year']+features, index=range(len(years)))
    Boruta_descisions_df = pd.DataFrame(columns=['year']+features, index=range(len(years)))

    for indx, year in tqdm(enumerate(years), 
                           desc="Processing Years", 
                           leave=False, 
                           initial=years[0], 
                           total=len(years)):
        
        cross_section_df = cross_sections[year]
        X = cross_section_df[features]
        y = cross_section_df['scenario_indx']
        
        model = tuner(X, y, seed=random_state, param_grid=param_grid, rand_search_kwgs=rand_search_kwgs, return_model=True)
        
        Feature_Selector = BorutaShap(
            model=model,
            importance_measure='shap',
            classification=True,
            percentile=80
        )

        Feature_Selector.fit(
            X=X, y=y, 
            n_trials=100, 
            sample=True, # Speeds up alg.
            normalize=True, # Give z scores for shap values
            verbose=False, 
            random_state=random_state
        )

        if plot and year % plot_fq == 0:
            print(f'plot {year}')
            title = f'Boruta Shap Feature Importance using {model_name} classifier ({year})'
            plot_kwgs = plot_kwgs if plot_kwgs is not None else {
                'which_features': 'all',
                'X_rotation': 90, 
                'X_size': 8, 
                'figsize': (12,8),
                'y_scale': 'log',
                'title': title, 
                'display': display_plot, 
                'return_fig': True
            }        
            plot_kwgs['title'] = title # update title with year
            fig, ax = Feature_Selector.plot(**plot_kwgs)

            if save_plot:
                save_path = f'/nird/home/johannef/Masterthesis_S23 Results/FigureFiles/Feature selection/Borutashap importance/{model_name}/'
                fig.savefig(f'{save_path}Boruta_shap_{model_name}_importance_{year}.png')
            
        results_df = pd.DataFrame(
            data={
                'Features':Feature_Selector.history_x.iloc[1:].columns.values,
                'Median Feature Importance':Feature_Selector.history_x.iloc[1:].median(axis=0).values,
                'Standard Deviation Importance':Feature_Selector.history_x.iloc[1:].std(axis=0).values
            }
        )

        decision_mapper = Feature_Selector.create_mapping_of_features_to_attribute(maps=['Tentative','Rejected','Accepted', 'Shadow'])
        results_df['Decision'] = results_df['Features'].map(decision_mapper)
        results_df = results_df[results_df['Features'].str.contains('Shadow') == False].sort_values('Features')
        
        importance = [year] + results_df['Median Feature Importance'].tolist()
        Boruta_scores_df.loc[indx] = importance

        decisions = [year] + results_df['Decision'].tolist()
        Boruta_descisions_df.loc[indx] = decisions

    if save_dfs:
        Boruta_scores_df.to_csv(f'/nird/home/johannef/Masterthesis_S23 DataFiles/tables/feature_selection/Boruta_shap_{model_name}_scores_ssp126ssp585.csv', index=False)
        Boruta_descisions_df.to_csv(f'/nird/home/johannef/Masterthesis_S23 DataFiles/tables/feature_selection/Boruta_shap_{model_name}_decisions_ssp126ssp585.csv', index=False)

### Classification Experiment ###
    
def run_classification_experiment(cross_sections, model_name, feature_combinations, years=None, seeds=None, param_grid=None, rand_search_kwgs=None):
    """
    Function to run a classification experiment across multiple years, seeds and feature combinations.

    Parameters:
    - cross_sections (dict): The cross-sectional data as a dictionary of dataframes. Use years as keys and the cross-sectional data as values.
    - model_name (str): The name of the model to use for feature selection. Options are 'RF', 'SVM', 'LR', 'XGB' and 'GNB'
    - feature_combinations (dict): The feature combinations to use for the experiment. Use a dictionary with feature combination keys and feature lists as values.
    - years (list): The years to process, if None all years in cross_sections will be used.
    - seeds (list): The random seeds to use, if None all seeds from 0 to 9 will be used.
    - param_grid (dict): The hyperparameter grid to search over
    - rand_search_kwgs (dict): The RandomizedSearchCV keyword arguments

    Returns:
    - dict: The classification summaries for each feature combination and year
    - dict: The confusion matrices for each feature combination and year
    """
    
    years = years if years is not None else list(cross_sections.keys())
    seeds = seeds if seeds is not None else list(range(10))

    if model_name == 'RF':
        run_across_seeds = run_RF_across_seeds
    elif model_name == 'SVM':
        run_across_seeds = run_SVM_across_seeds
    elif model_name == 'LR':
        run_across_seeds = run_LR_across_seeds
    elif model_name == 'XGB':
        run_across_seeds = run_XGB_across_seeds
    elif model_name == 'GNB':
        run_across_seeds = run_GNB_across_seeds
    else:
        raise ValueError('Model name not recognized')
    
    classification_summaries = {feature_comb_key: {} for feature_comb_key in feature_combinations.keys()}
    confusion_matrices = {feature_comb_key: {} for feature_comb_key in feature_combinations.keys()}                              
        
    for year in tqdm(years, 
                     desc="Processing Years", 
                     colour='green', 
                     leave=False, 
                     initial=years[0], 
                     total=len(years)):
        cross_section_df = cross_sections[year]

        for feature_comb_key, feature_comb in feature_combinations.items():
            if isinstance(feature_comb, dict): # all dynamic features for each year
                feature_comb = feature_comb[year]
                if len(feature_comb) == 0:
                    continue
                
            X = cross_section_df[feature_comb]
            y = cross_section_df['scenario_indx']

            df_summaryallseeds, cm_all_seeds = run_across_seeds(X, y, seeds, param_grid, rand_search_kwgs)

            classification_summaries[feature_comb_key][year] = df_summaryallseeds
            average_cm = (sum(cm_all_seeds) / (len(cm_all_seeds)*24)*100).astype(float).round(2)
            confusion_matrices[feature_comb_key][year] = average_cm

    return classification_summaries, confusion_matrices