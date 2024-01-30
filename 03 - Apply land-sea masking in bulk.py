# author: Johannes L. Fjeldså
# email: johannes.larsen.fjeldså@nmbu.no

#%% Import libraries
from src.preproces import *

#%% Setup
work_dir = '/nird/home/johannef/Masterthesis_S23'

file_handler = Handle_Files(work_dir)
preprocesser = Preprocess_Climate_Data(work_dir)

data_path = '/nird/home/johannef/Masterthesis_S23 DataFiles/Annualclimatologies/rx5dayETCCDI'

sorted_files = {'ssp126': file_handler.get_all_filenames_in_dir('/'.join([data_path, 'ssp126'])),
                'ssp370': file_handler.get_all_filenames_in_dir('/'.join([data_path, 'ssp370'])),
                'ssp245': file_handler.get_all_filenames_in_dir('/'.join([data_path, 'ssp245'])),
                'ssp585': file_handler.get_all_filenames_in_dir('/'.join([data_path, 'ssp585']))}



#%% Edit mask
path_raw_masks = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/masks/'
land_sea_mask = file_handler.read_netcdf_to_xr(''.join([path_raw_masks, 'landfrac_ACCESS-ESM1-5.nc']))


test_file = file_handler.read_netcdf_to_xr('/'.join([data_path, 
                                                     'ssp126', 
                                                     sorted_files['ssp126'][0]]))

land_sea_mask['lon'] = test_file.rx5dayETCCDI.lon
land_sea_mask = land_sea_mask.rename({'time': 'year'})

land_sea_mask_w_ts = land_sea_mask.reindex_like(test_file)


land_sea_mask_w_ts = land_sea_mask_w_ts.assign(
    landfrac=(['lat', 'lon'], 
    np.array(land_sea_mask.isel(year=0)['landfrac'].values))
    )

land_mask = land_sea_mask_w_ts.where(land_sea_mask_w_ts >= 0.8).notnull()
sea_mask = land_sea_mask_w_ts.where(land_sea_mask_w_ts <= 0.2).notnull()


# rechek compatability, save if it is compatible
if preprocesser.check_mask_compatability(test_file, 'rx5dayETCCDI', land_mask) and preprocesser.check_mask_compatability(test_file, 'rx5dayETCCDI', sea_mask):
    print('Mask is compatible with files')

    
#%% Apply mask 
for scenario in sorted_files.keys():
    file_names = sorted_files[scenario]
    save_path_landmasked = work_dir + ' DataFiles/land_masked_annual_climatalogies/' + 'rx5dayETCCDI' + '/' + scenario + '/'
    save_path_seamasked = work_dir + ' DataFiles/sea_masked_annual_climatalogies/' + 'rx5dayETCCDI' + '/' + scenario + '/'
    for file_name in file_names:
        ds = file_handler.read_netcdf_to_xr(directory='/'.join([data_path, scenario]), 
                                            file_name=file_name)
        
        save_name_landmasked = file_name.replace('.nc', '_landmasked.nc')
        ds_landmasked = ds.where(land_mask['landfrac'])
        file_handler.save_dataset_to_netcdf(ds_landmasked, 
                                            save_name_landmasked,
                                            save_path_landmasked)

        save_name_seamasked = file_name.replace('.nc', '_seamasked.nc')
        ds_seamasked = ds.where(sea_mask['landfrac'])
        file_handler.save_dataset_to_netcdf(ds_seamasked, 
                                            save_name_seamasked,
                                            save_path_seamasked)

