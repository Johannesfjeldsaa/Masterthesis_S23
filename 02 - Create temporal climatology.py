# author: Johannes L. Fjeldså
# email: johannes.larsen.fjeldså@nmbu.no

#%% import libraries
from src.preproces import *
import os

#%% setup
# get test file# Define folder path
#data_path = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/'
extrem_data_path = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/ETCCDI'
work_dir = '/nird/home/johannef/Masterthesis_S23'

file_handler = Handle_Files(work_dir)
preprocesser = Preprocess_Climate_Data(work_dir)

#tas_filenames = file_handler.get_all_filenames_in_dir(data_path, 
#                                                      condition=lambda filename: filename.endswith(".nc"),
#                                                      substrings=['ssp', 'tas', 'ACCESS-ESM1-5'])
#pr_filenames = file_handler.get_all_filenames_in_dir(data_path, 
#                                                     condition=lambda filename: filename.endswith(".nc"),
#                                                     substrings=['ssp', 'pr', 'ACCESS-ESM1-5'])
#txx_filenames = file_handler.get_all_filenames_in_dir(extrem_data_path, 
#                                                      substrings=['.nc', 'ssp', 'txx', '_yr_', 'ACCESS-ESM1-5'])
rx5day_filenames = file_handler.get_all_filenames_in_dir(extrem_data_path, 
                                                         substrings=['.nc', 'ssp', 'rx5day', '_yr_', 'ACCESS-ESM1-5'])

sorted_files = {}

for file in rx5day_filenames:
    scenario = file.split('_')[3]
    if scenario not in sorted_files.keys():
        sorted_files[scenario] = []
    sorted_files[scenario].append(file)

#%% Create yearly climatology
    
for scenario in sorted_files.keys():
    file_names = sorted_files[scenario]
    save_path = work_dir + ' DataFiles/Annualclimatologies/' + 'rx5dayETCCDI' + '/' + scenario + '/'
    for file_name in file_names:
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

        preprocesser.create_temporal_climatology(dataset=dataset,
                                                 var_name='rx5dayETCCDI',
                                                 climatology_type="yearly",
                                                 save_to_dataset=True,
                                                 file_name=save_name,
                                                 directory=save_path, 
                                                 is_original_name=False,
                                                 re_open=False)


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
