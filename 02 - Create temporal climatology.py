# author: Johannes L. Fjeldså
# email: johannes.larsen.fjeldså@nmbu.no

#%% import libraries
from src.preproces import *
import os
from tqdm import tqdm

#%% setup
# get test file# Define folder path
#data_path = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/'
extrem_data_path = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/ETCCDI'
work_dir = '/nird/home/johannef/Masterthesis_S23'
var = 'txxETCCDI'

file_handler = Handle_Files(work_dir)
preprocesser = Preprocess_Climate_Data(work_dir)

#tas_filenames = file_handler.get_all_filenames_in_dir(data_path, 
#                                                      condition=lambda filename: filename.endswith(".nc"),
#                                                      substrings=['ssp', 'tas', 'ACCESS-ESM1-5'])
#pr_filenames = file_handler.get_all_filenames_in_dir(data_path, 
#                                                     condition=lambda filename: filename.endswith(".nc"),
#                                                     substrings=['ssp', 'pr', 'ACCESS-ESM1-5'])
filenames = file_handler.get_all_filenames_in_dir(extrem_data_path, 
                                                  substrings=['.nc', 'ssp', 'txx', '_yr_', 'ACCESS-ESM1-5'])
#rx5day_filenames = file_handler.get_all_filenames_in_dir(extrem_data_path, 
#                                                         substrings=['.nc', 'ssp', 'rx5day', '_yr_', 'ACCESS-ESM1-5'])
#gsl_filenames = file_handler.get_all_filenames_in_dir(extrem_data_path, 
#                                                      substrings=['.nc', 'ssp', 'gslETCCDI', '_yr_', 'ACCESS-ESM1-5'])
#filenames = file_handler.get_all_filenames_in_dir(extrem_data_path, 
#                                                     substrings=['.nc', 'ssp', var, '_yr_', 'ACCESS-ESM1-5'])
sorted_files = {}

for file in filenames:
    scenario = file.split('_')[3]
    if scenario not in sorted_files.keys():
        sorted_files[scenario] = []
    sorted_files[scenario].append(file)


#%% Create yearly climatology
    
for scenario, file_names in sorted_files.items():
    save_path = work_dir + ' DataFiles/Annualclimatologies/nomask/' + var + '/' + scenario + '/'
    for file_name in tqdm(file_names):
        dataset = (
            file_handler.read_netcdf_to_xr(directory=extrem_data_path, file_name=file_name)
            #.drop('height')
        ) 

        # file_name = file_name.replace("_day_", "_yr_")
        if '2300' in file_name:
            dataset = dataset.sel(time=slice(None,'2100'))
            save_name = file_name.replace("2015-2300", "2015-2100")
        else:
            save_name = file_name

        #    save_name = file_name.replace("20150101-23001231", "2015-2100")
        dataset = preprocesser.create_temporal_climatology(dataset=dataset,
                                                           var_name=var,
                                                           climatology_type="yearly")
        
        #dataset[var] = (dataset[var].astype(np.float64) / (24*3600*10**9)).astype(np.int64)
        file_handler.save_dataset_to_netcdf(dataset, save_name, save_path)



# all files are now yearly observations         
#%% Check for files runned until 2300

save_dir = work_dir + ' DataFiles/Annualclimatologies/pr'

for scenario in sorted_files.keys():
    file_names = file_handler.get_all_filenames_in_dir('/'.join([save_dir, scenario]), 
                                                      substrings=['2300'])
    for file_name in file_names:
        dataset = (
            file_handler.read_netcdf_to_xr(directory='/'.join([save_dir, scenario]), file_name=file_name)
            .sel(year=slice(None,'2100'))
        )
        
        save_name = file_name.replace("20150101-23001231", "2015-2100")
        dataset.to_netcdf('/'.join([save_dir, scenario, save_name]))
        # Delete original file
        os.remove('/'.join([save_dir, scenario, file_name]))
