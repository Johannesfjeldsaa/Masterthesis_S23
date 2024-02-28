# author: Johannes L. Fjeldså
# email: johannes.larsen.fjeldså@nmbu.no

#%% Import libraries
from src.preproces import *

#%% Setup
work_dir = '/nird/home/johannef/Masterthesis_S23'
var = 'pr'

file_handler = Handle_Files(work_dir)
preprocesser = Preprocess_Climate_Data(work_dir)

data_path = '/nird/home/johannef/Masterthesis_S23 DataFiles/Annualclimatologies/nomask/' + var

sorted_files = {'ssp126': file_handler.get_all_filenames_in_dir('/'.join([data_path, 'ssp126'])),
                'ssp370': file_handler.get_all_filenames_in_dir('/'.join([data_path, 'ssp370'])),
                'ssp245': file_handler.get_all_filenames_in_dir('/'.join([data_path, 'ssp245'])),
                'ssp585': file_handler.get_all_filenames_in_dir('/'.join([data_path, 'ssp585']))}



#%% Edit mask
path_masks = '/nird/home/johannef/Masterthesis_S23 DataFiles/masks'

def condition(string):
    if string not in ['land_mask_ACCESS-ESM1-5.nc', 'sea_mask_ACCESS-ESM1-5.nc']:
        return string
    
mask_names = file_handler.get_all_filenames_in_dir(path_masks, condition=condition)
test_file = file_handler.read_netcdf_to_xr('/'.join([data_path, 
                                                     'ssp126', 
                                                     sorted_files['ssp126'][0]]))
masks = {}
for mask_name in mask_names:
    mask = file_handler.read_netcdf_to_xr(directory=path_masks, file_name=mask_name)
    mask_key = mask_name.replace('_ACCESS-ESM1-5.nc', '')
    if preprocesser.check_mask_compatability(test_file, 'pr', mask):
        masks[mask_key] = mask
    else:
        print('erroralarm')
    
#%% Apply mask 
for scenario in sorted_files.keys():
    file_names = sorted_files[scenario]

    for mask_key, mask in masks.items():
        save_path_mask = work_dir + ' DataFiles/Annualclimatologies/' + mask_key + '/' + var + '/' + scenario + '/'
    
        for file_name in file_names:
            ds = file_handler.read_netcdf_to_xr(directory='/'.join([data_path, scenario]), 
                                                file_name=file_name)
            
            save_name = file_name.replace('.nc', f'_{mask_key}.nc')
            ds_masked = ds.where(mask.mask)
            file_handler.save_dataset_to_netcdf(ds_masked, 
                                                save_name,
                                                save_path_mask)



# %%
