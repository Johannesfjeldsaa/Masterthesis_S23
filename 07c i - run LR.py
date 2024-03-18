import pandas as pd
import numpy as np
import pickle

from src.helperfunctions_ML import run_classification_experiment, open_cross_sections
from src.postproces_classificationresults import summarize_with_df

scaled_cross_sections = open_cross_sections(scaled=True)


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

years = list(range(2015, 2051))
target_summaries, roc_information = run_classification_experiment(
    cross_sections=scaled_cross_sections, 
    model_name='LR', 
    feature_combinations=feature_combinations, 
    years=years, 
    seeds=[int(i) for i in range(50)], 
    include_ROC_analysis=True, roc_analysis_fq=10,
    )

with open('D:/Programmering/msc/Masterthesis_S23-Results/dicts/LR/target_summaries_LR.pkl', 'wb') as fp:
    pickle.dump(target_summaries, fp)

classification_summaries = summarize_with_df(target_summaries)
with open('D:/Programmering/msc/Masterthesis_S23-Results/dicts/LR/classification_summaries_LR.pkl', 'wb') as fp:
    pickle.dump(classification_summaries, fp)

with open('D:/Programmering/msc/Masterthesis_S23-Results/dicts/LR/roc_information_LR.pkl', 'wb') as fp:
    pickle.dump(roc_information, fp)