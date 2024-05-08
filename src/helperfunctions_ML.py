# This file contains helper functions for running classification experiments 
# author: Johannes Fjeldså
# email: johannes.larsen.fjeldsa@nmbu.no

# Importing packages
import pandas as pd
import numpy as np
from tqdm import tqdm

import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve



from sklearnex import patch_sklearn
# https://intel.github.io/scikit-learn-intelex/latest/samples/random_forest_yolanda.html
patch_sklearn()

import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


### Data Handling ###

def open_cross_sections(scenario_indx_key=None, years=None, scaled=False, directory=None):
    """
    Function to open cross-sectional data for multiple years.
    - If scaled is True the function will open the standardscaled data, otherwise the unscaled data will be opened.
    - The function will return a dictionary with years as keys and cross-sectional data as values.

    Parameters:
    - scenario_indx_key (dict): The scenario index key to use for mapping scenarios to integers.
    - years (list): The years to open data for.
    - scaled (bool): Wether or not to open the standardscaled data.
    - directory (str): The directory to open the data from. If None the default directory will be used.

    Returns:
    - dict: The cross-sectional data for each year. 
    """
    directory = directory if directory is not None else 'D:/Programmering/msc/MSc_DataFiles_040424/tables/cross_sections'
    scenario_indx_key = scenario_indx_key if scenario_indx_key is not None else {'ssp126': 0, 'ssp585': 1}
    years = years if years is not None else list(range(2015, 2101))

    if scaled:
        cross_sections = {year: pd.read_csv('/'.join([directory, 'standardscaled', f'cross_section_{year}_standscaled.csv'])) for year in years}
    else:
        cross_sections = {year: pd.read_csv('/'.join([directory, 'unscaled', f'cross_section_{year}.csv'])) for year in years}

    for year in years:
        cross_section = cross_sections[year]
        cross_section['scenario_indx'] = cross_section['scenario'].map(scenario_indx_key)
        cross_section['scenario_indx'] = cross_section['scenario_indx'].astype('Int64')
        cross_sections[year] = cross_section[cross_section['scenario'].isin(list(scenario_indx_key.keys()))]

    return cross_sections

### tune functions ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


def tune(X_train, y_train, model_name, seed, search_alg, param_grid, scoring,  search_kwgs=None, return_model=False, roc_analysis=False):
    """
    Function to tune model using cross validation. 
    resourses:
    - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    - https://www.analyticsvidhya.com/blog/2021/01/gaussian-naive-bayes-with-hyperpameter-tuning/ 
    - https://xgboost.readthedocs.io/en/latest/parameter.html
    - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


    Parameters:
    - X_train (pd.DataFrame): The training data
    - y_train (pd.Series): The training labels
    - model_name (str): The name of the model to tune. Options are 'RF', 'SVM', 'LR', 'GNB', 'XGB', 'KNN'
    - seed (int): The random seed
    - search_alg (str): The search algorithm to use for hyperparameter tuning. Options are 'random' and 'grid'
    - param_grid (dict): The hyperparameter grid to search over
    - search_kwgs (dict): The RandomizedSearchCV or GridSearchCV keyword arguments
    - return_model (bool): Wether or not to return the best model or the best hyperparameters
    - roc_analysis (bool): Wether or not to include ROC analysis. Parameter is only used for SVC as this model dont have the predict_proba method built in.

    Returns:
    - The best model if return_model is True or the best hyperparameters if return_model is False
    """

    if model_name == 'RF':
        model = RandomForestClassifier(random_state=seed)
        param_grid = param_grid if param_grid is not None else {
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_depth': [None] + [int(x) for x in np.linspace(10, 110, num = 11)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4, 6],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        }

    elif model_name == 'DT':
        model = DecisionTreeClassifier(random_state=seed)
        param_grid = param_grid if param_grid is not None else {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [2, 5, 10, 15, 20],
            'criterion': ["gini"]
            }

    elif model_name == 'SVM' or model_name == 'SVC':
        model = SVC(random_state=seed)
        param_grid = param_grid if param_grid is not None else {
            'C': np.logspace(-3, 2, 6),
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 6, 10], 
            }
        if roc_analysis:
            param_grid['probability'] = [True]

    elif model_name == 'LR':
        model = LogisticRegression(random_state=seed)
        param_grid = param_grid if param_grid is not None else {
            'solver': ['newton-cg', 'liblinear'],
            'penalty': ['l2'],
            'C': [100, 10, 1.0, 0.1, 0.01]
            }
    elif model_name == 'GNB':
        model = GaussianNB()
        param_grid = param_grid if param_grid is not None else {
            'var_smoothing': np.logspace(0,-9, num=100)
            }
    elif model_name == 'XGB':
        model = XGBClassifier
        param_grid = param_grid if param_grid is not None else {
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_depth':range(3,10,2),
            'min_child_weight': [1, 2, 3, 4, 5],
            'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'alpha': [1e-5, 1e-2, 0.1, 1, 100],
            'lambda': [1e-5, 1e-2, 0.1, 1, 100],
        }
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
        param_grid = param_grid if param_grid is not None else {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
            }
    else:
        raise ValueError('Model name not recognized')
    

    estimator = model
    if search_alg == 'random':
        search_kwgs = search_kwgs if search_kwgs is not None else {
            'n_iter': 100, # Covers more feature combinations 
            'cv': 10, # higher num decreases chance of overfitting 
            'verbose': 0,
            'random_state': seed,
            'n_jobs': -1,
            'scoring': scoring,
            }
        search = RandomizedSearchCV(estimator=estimator, 
                                    param_distributions=param_grid, 
                                    **search_kwgs)
    elif search_alg == 'grid':
        search_kwgs = search_kwgs if search_kwgs is not None else {
            'cv': 10, # higher num decreases chance of overfitting 
            'verbose': 0,
            'n_jobs': -1,
            'scoring': scoring
            }
        search = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              **search_kwgs)
    else:
        raise ValueError('Search algorithm not recognized')

    search.fit(X_train, y_train)

    if return_model:
        return model(**search.best_params_)
    else:
        return search.best_params_
    

### Classification Experiment ###

def run_across_seeds(X, y, seeds, model_name, search_alg, param_grid, scoring, search_kwgs, skip_tuning=False, include_ROC_analysis=False):
    """
    Function to run model across multiple seeds and summarize the results. 
    ROC code from (https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates)

    Parameters:
    - X (pd.DataFrame): The training data
    - y (pd.Series): The training labels
    - seeds (list): The random seeds to use
    - model_name (str): The name of the model to use for feature selection. Options are 'RF', 'SVM', 'LR'
    - param_grid (dict): The hyperparameter grid to search over
    - rand_search_kwgs (dict): The RandomizedSearchCV keyword arguments
    - skip_tuning (bool): Wether or not to skip the tuning process
    - include_ROC_analysis (bool): Wether or not to include ROC analysis. If included the function will return the ROC data in addition to target_summaries

    Returns:
    - target_summaries (pd.DataFrame): The classification results for each seed in the form of a dataframe with columns for seed, y_true (vector) and y_pred (vector)
    - roc_information (pd.DataFrame): The ROC data for each seed in the form of a dict with the 'actual_data' 
                                      as a df with columns for seed, fpr (vector) and tpr (vector). 
                                      And interped data to enable plotting the mean roc curve across seeds.
    """
    if model_name == 'RF':
        model = RandomForestClassifier
    elif model_name == 'SVM' or model_name == 'SVC':
        model = SVC
    elif model_name == 'LR':
        model = LogisticRegression
    elif model_name == 'GNB':
        model = GaussianNB
    elif model_name == 'XGB':
        model = XGBClassifier
    elif model_name == 'KNN':
        model = KNeighborsClassifier
    elif model_name == 'DT':
        model = DecisionTreeClassifier
    else:
        raise ValueError('Model name not recognized')
    
    target_summaries_per_year_and_feature_comb = pd.DataFrame(columns=['seed', 'y_train_true', 'y_train_pred', 
                                                                       'y_test_true', 'y_test_pred',  'model'], 
                                                              index=range(len(seeds)))
    if include_ROC_analysis:
        roc_summaries_per_year_and_feature_comb = pd.DataFrame(columns=['seed', 'fpr', 'tpr'], 
                                                               index=range(len(seeds)))
        tprs = []
        base_fpr = np.linspace(0, 1, 11)

    
    for indx, seed in tqdm(enumerate(seeds), 
                           desc="Processing seeds", 
                           bar_format='{l_bar}{bar:20}{r_bar}', 
                           leave=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        X_train, X_test = X_train.values, X_test.values
        y_train, y_test = y_train.values, y_test.values

        if skip_tuning:
            best_model = model(random_state=seed)
        else:
            best_config = tune(X_train, y_train, model_name, seed, search_alg, param_grid, scoring, search_kwgs, roc_analysis=include_ROC_analysis)
            best_model = model(**best_config)
        
        best_model.fit(X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        target_summaries_per_year_and_feature_comb.loc[indx] = [seed, 
                                                                y_train.to_numpy(dtype=int), y_train_pred.astype(np.dtype(int)),
                                                                y_test.to_numpy(dtype=int), y_test_pred.astype(np.dtype(int)), 
                                                                best_model.get_params()]
        if include_ROC_analysis:
            
            probabilities = best_model.predict_proba(X_test)
            probabilities_posclass = probabilities[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probabilities_posclass)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

            roc_summaries_per_year_and_feature_comb.loc[indx] = [seed, fpr, tpr]

    if include_ROC_analysis:
        tprs = np.array(tprs)
        mean_tprs = np.mean(tprs, axis=0)
        std = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        roc_information = {
            'actual_data': roc_summaries_per_year_and_feature_comb, 
            'plot_data': {
                'mean_tprs': mean_tprs, 'tprs_upper': tprs_upper, 'tprs_lower': tprs_lower, 'base_fpr': base_fpr
            }
        }

        return target_summaries_per_year_and_feature_comb, roc_information
    else:
        return target_summaries_per_year_and_feature_comb



def run_classification_experiment(cross_sections, model_name, 
                                  feature_combinations, years=None, seeds=None, 
                                  search_alg=None, param_grid=None, scoring=None, search_kwgs=None, skip_tuning=False,
                                  include_ROC_analysis=False, roc_analysis_fq=10, 
                                  ):
    """
    Function to run a classification experiment across multiple years, seeds and feature combinations.

    Parameters:
    - cross_sections (dict): The cross-sectional data as a dictionary of dataframes. Use years as keys and the cross-sectional data as values.
    - model_name (str): The name of the model to use for feature selection. Options are 'RF', 'SVM', 'LR', 'XGB' and 'GNB'
    - feature_combinations (dict): The feature combinations to use for the experiment. Use a dictionary with feature combination keys and feature lists as values.
    - years (list): The years to process, if None all years in cross_sections will be used.
    - seeds (list): The random seeds to use, if None all seeds from 0 to 9 will be used.
    - search_alg (str): The search algorithm to use for hyperparameter tuning. Options are 'random' and 'grid'
    - param_grid (dict): The hyperparameter grid to search over
    - search_kwgs (dict): The RandomizedSearchCV or GridSearchCV keyword arguments
    - skip_tuning (bool): Wether or not to skip the tuning process
    - include_ROC_analysis (bool): Wether or not to include ROC analysis. If included the function will return the ROC data in addition to target_summaries

    Returns:
    - dict: The classification summaries for each feature combination and year
    - dict: The ROC data for each feature combination and year. Keys are feature_comb_keys in first layer and years in second. 
            Third layer contains dict with the actual data and the interped data to enable plotting the mean roc curbe across seeds. 
    """
    
    years = years if years is not None else list(cross_sections.keys())
    seeds = seeds if seeds is not None else list(range(10))
    search_alg = search_alg if search_alg is not None else 'random'
    scoring = scoring if scoring is not None else 'accuracy'

        
    target_summaries = {feature_comb_key: {} for feature_comb_key in feature_combinations.keys()}
    if include_ROC_analysis:
        roc_information = {feature_comb_key: {} for feature_comb_key in feature_combinations.keys()}

    for year in tqdm(years, 
                     bar_format='{l_bar}{bar:20}{r_bar}', 
                     desc ="Iterating years"):
        cross_section_df = cross_sections[year]

        for feature_comb_key, feature_comb in tqdm(feature_combinations.items(), 
                                                   bar_format='{l_bar}{bar:20}{r_bar}', 
                                                   leave=False, 
                                                   desc ="Iterating feature combinations"):
            
            X = cross_section_df[feature_comb]
            y = cross_section_df['scenario_indx']
            print(y)
            if include_ROC_analysis and year % roc_analysis_fq == 0:
                target_summaries[feature_comb_key][year], roc_information[feature_comb_key][year] = run_across_seeds(X, y, seeds, model_name, 
                                                                                                                     search_alg, param_grid, scoring, search_kwgs, 
                                                                                                                     skip_tuning,
                                                                                                                     include_ROC_analysis)
            else:
                target_summaries[feature_comb_key][year] = run_across_seeds(X, y, seeds, model_name,
                                                                            search_alg, param_grid, scoring, search_kwgs, 
                                                                            skip_tuning)

    if include_ROC_analysis:
        return target_summaries, roc_information
    else:
        return target_summaries
