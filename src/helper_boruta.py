from BorutaShap import BorutaShap
from helperfunctions_ML import *
import pandas as pd
from tqdm import tqdm

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
            percentile=100
        )

        Feature_Selector.fit(
            X=X, y=y, 
            n_trials=100, 
            sample=True, # Speeds up alg.
            #normalize=True, # Give z scores for shap values
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

