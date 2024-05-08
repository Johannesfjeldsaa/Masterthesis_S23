
import pickle
import os
import numpy as np

from src.helperfunctions_ML  import run_classification_experiment, open_cross_sections
from src.postproces_classificationresults import summarize_with_df

scaled_cross_sections = open_cross_sections(scaled=True, scenario_indx_key={'ssp126': 0, 'ssp245': 1, 'ssp370': 2, 'ssp585': 4})
model_name = 'LR' # Change this to the model you want to run: 'LR', 'RF', 'GNB', 'XGB', SVM/SVC, 'KNN'

nomask_features = ['fdETCCDI: nomask', 'gslETCCDI: nomask', 'pr: nomask', 'tas: nomask', 'txxETCCDI: nomask']
boruta_RF_features = ['pr: nomask', 'pr: sea_mask', 'tas: land_mask', 'tas: nomask', 'tas: sea_mask', 'txxETCCDI: land_mask', 'txxETCCDI: nomask', 'txxETCCDI: sea_mask']
mRMR_f_mut_features = ['fdETCCDI: nomask', 'gslETCCDI: nomask', 'tas: land_mask', 'tas: nomask', 'tas: sea_mask', 'txxETCCDI: land_mask', 'txxETCCDI: lat_mask_pm30deg', 'txxETCCDI: nomask', 'txxETCCDI: sea_mask', 'txxETCCDI: sea_mask_pm30deg']
johannes_supervised_features = ['fdETCCDI: nomask', 'pr: nomask', 'pr: sea_mask', 'rx5dayETCCDI: land_mask', 'tas: nomask', 'txxETCCDI: sea_mask']

feature_combinations = {
    'nomask_baseline': nomask_features, 
    'boruta_RF_features': boruta_RF_features,
    'mRMR_f_mut_features': mRMR_f_mut_features,
    'johannes_supervised_features': johannes_supervised_features
}


param_grid = {
    'multi_class': ['ovr'], 
    'solver': ['newton-cg', 'lbfgs'],
    'penalty': ['l2'],
    'C': [0.1, 1, 10, 100],
    }

start_year = 2015
end_year = 2050
years = list(range(start_year, end_year+1))
years = [2020, 2030, 2035, 2036, 2037, 2038, 2039, 2040, 2045, 2050]
target_summaries = run_classification_experiment(
    cross_sections=scaled_cross_sections, 
    model_name=model_name, 
    feature_combinations=feature_combinations, 
    years=years, 
    seeds=[int(i) for i in range(50)], 
    search_alg='grid', 
    param_grid=param_grid, #param_grid, # Use None for default hyperparameter grids
    scoring='roc_auc_ovo', 
    search_kwgs=None, # Use None for default search settings
    skip_tuning=False,
    include_ROC_analysis=False, 
    roc_analysis_fq=10
    )


save_path = f'D:/Programmering/msc/Masterthesis_S23-Results/dicts/{model_name}/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open('/'.join([save_path, f'target_summaries_{model_name}_allspps_{start_year}{end_year}_f1.pkl']), 'wb') as fp:
    pickle.dump(target_summaries, fp)

classification_summaries = summarize_with_df(target_summaries)
with open('/'.join([save_path, f'classification_summaries_{model_name}_allspps_{start_year}{end_year}_f1.pkl']), 'wb') as fp:
    pickle.dump(classification_summaries, fp)

with open('/'.join([save_path, f'roc_information_{model_name}_allspps_{start_year}{end_year}_f1.pkl']), 'wb') as fp:
    pickle.dump(roc_information, fp)


