# Import libraries
from lib2to3.pgen2.literals import test
from src.preproces import *
import xarray as xr 
import numpy as np

work_dir = '/nird/home/johannef/Masterthesis_S23'

file_handler = Handle_Files(work_dir)
preprocesser = Preprocess_Climate_Data(work_dir)

def open_dummy_file(var):
    dummy_file = file_handler.read_netcdf_to_xr('/nird/home/johannef/Masterthesis_S23 DataFiles/Annualclimatologies/nomask/' + var + '/ssp126/'+ var +'_yr_ACCESS-ESM1-5_ssp126_r1i1p1f1_gn_2015-2100.nc')
    return dummy_file

def initiate_dummymask_from_test_file(var, test_file=None):
    # Create ds as basis for mask creation
    dummy_file = test_file if test_file is not None else open_dummy_file(var)
    ds = dummy_file.copy()
    ds = xr.Dataset(coords={'lat': test_file['lat'], 'lon': test_file['lon'], 'year': test_file['year']})
    ds['mask'] = xr.DataArray(data=None, 
                            coords={'lat': test_file['lat'], 'lon': test_file['lon'], 'year': test_file['year']}, 
                            dims=['lat', 'lon', 'year'])
    ds['mask'].data = np.ones_like(ds['mask'], dtype=np.int32)
    mask = ds.copy()

    return mask

def create_mask_from_AR6_abriviations(region_abriviations, var, new_mask_name, test_file=None, dummymask=None):

    raw_masks_dir = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/masks/'
    test_file = test_file if test_file is not None else open_dummy_file(var)
    dummymask = dummymask if dummymask is not None else initiate_dummymask_from_test_file(var, test_file)

    mask_names = file_handler.get_all_filenames_in_dir(raw_masks_dir)
    region_mask_names = [mask_name for mask_name in mask_names if any(region in mask_name for region in region_abriviations)]

    mask_arrays = {}
    for mask_name in region_mask_names:
        mask_path = '/'.join([raw_masks_dir, mask_name])
        mask = file_handler.read_netcdf_to_xr(mask_path)
        mask['lon'] = test_file['lon']
        mask_arrays[mask_name] = mask['mask_array'].values

    combined_mask = np.sum(list(mask_arrays.values()), axis=0)
    mask = dummymask.assign(mask=(['lat', 'lon'], combined_mask))
    mask = mask.assign_attrs(mask_name=new_mask_name, 
                             description=f"Source is a combination of: {list(mask_arrays.keys())}")
    return mask

def create_latitude_mask(min_lat, max_lat, var, new_mask_name, test_file=None, dummymask=None):

    dummymask = dummymask if dummymask is not None else initiate_dummymask_from_test_file(var, test_file)
    mask = dummymask
    mask['mask'] = dummymask['mask'].where((dummymask['lat'] >= min_lat) & (dummymask['lat'] <= max_lat)).notnull()
    mask = mask.assign_attrs(mask_name=new_mask_name)
    return mask

def open_landmask(combine_further=False):
    if combine_further is False:
        land_mask = file_handler.read_netcdf_to_xr('/nird/home/johannef/Masterthesis_S23 DataFiles/masks/land_mask_ACCESS-ESM1-5.nc').notnull()
        land_mask = land_mask.assign_attrs(mask_name='land_mask')
    else:
        land_mask = file_handler.read_netcdf_to_xr('/nird/home/johannef/Masterthesis_S23 DataFiles/masks/land_mask_ACCESS-ESM1-5.nc')

    return land_mask

def open_seamask(combine_further=False):
    if combine_further is False:
        sea_mask = file_handler.read_netcdf_to_xr('/nird/home/johannef/Masterthesis_S23 DataFiles/masks/sea_mask_ACCESS-ESM1-5.nc').notnull()
        sea_mask = sea_mask.assign_attrs(mask_name='sea_mask')
    else:
        sea_mask = file_handler.read_netcdf_to_xr('/nird/home/johannef/Masterthesis_S23 DataFiles/masks/sea_mask_ACCESS-ESM1-5.nc')

    return sea_mask

def create_land_sea_latitude_mask(min_lat, max_lat, new_mask_name, land=False, sea=False):
    if land is True:
        base_mask = open_landmask(combine_further=True)
    elif sea is True:
        base_mask = open_seamask(combine_further=True)        
    else:
        raise ValueError('Gi om det er land eller sjømasken som skal lages.')
    mask = base_mask.where((base_mask['lat'] >= min_lat) & (base_mask['lat'] <= max_lat)).notnull()
    mask = mask.assign_attrs(mask_name=new_mask_name)
    return mask

def agregate_land_sea_mask(land=False, sea=False, test_file=None, var='tas'):

    test_file = test_file if test_file is not None else open_dummy_file(var)
    
    path_raw_masks = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/masks/'
    land_sea_mask = file_handler.read_netcdf_to_xr(''.join([path_raw_masks, 'landfrac_ACCESS-ESM1-5.nc']))

    land_sea_mask['lon'] = test_file[var].lon
    land_sea_mask = land_sea_mask.rename({'time': 'year'})
    land_sea_mask = land_sea_mask.rename({'landfrac': 'mask'})

    land_sea_mask_w_ts = land_sea_mask.reindex_like(test_file)

    land_sea_mask_w_ts = land_sea_mask_w_ts.assign(
        mask=(['lat', 'lon'], 
        np.array(land_sea_mask.isel(year=0)['mask'].values))
        )
    if land is True:
        land_mask = land_sea_mask_w_ts.where(land_sea_mask_w_ts >= 0.8)
        land_mask = land_mask.assign_attrs(mask_name='land_mask')
        return land_mask
    elif sea is True:
        sea_mask = land_sea_mask_w_ts.where(land_sea_mask_w_ts <= 0.2)
        sea_mask = sea_mask.assign_attrs(mask_name='sea_mask')
        return sea_mask

    
def run_global_climatology(mask, mask_name, weights, main_data_folder, main_save_folder_temporal, main_save_folder_spatial, variables, SSPs):

        main_data_folder = '/'.join([main_data_folder, 'nomask'])
        main_save_folder_temporal = '/'.join([main_save_folder_temporal, mask_name])
        main_save_folder_spatial = '/'.join([main_save_folder_spatial, mask_name])
        for var in variables:
            for scenario in SSPs:
                data_folder = '/'.join([main_data_folder, var, scenario])
                save_folder_temporal = '/'.join([main_save_folder_temporal, var, scenario])
                save_folder_spatial = '/'.join([main_save_folder_spatial, var, scenario])
                file_names = file_handler.get_all_filenames_in_dir(data_folder)
                for file_name in file_names:
                    print(file_name)
                    save_name = file_name.replace('.nc', f'_{mask_name}.nc')
                    ds = file_handler.read_netcdf_to_xr(directory=data_folder,
                                                        file_name=file_name)
                    ds_masked = ds.where(mask.mask)
                    file_handler.save_dataset_to_netcdf(ds_masked, 
                                                        save_name,
                                                        save_folder_temporal)
                    
                    save_name = save_name.replace('.nc', '_glob.nc')
                    preprocesser.create_spatial_climatology(ds_masked,
                                                            var_name=var,
                                                            weights=weights,
                                                            climatology_type='global',
                                                            save_to_dataset=True, 
                                                            file_name=save_name,
                                                            directory=save_folder_spatial,
                                                            is_original_name=False)


def create_masked_climatologies(mask_name, var, excisting_mask=False, region_abriviations=None, max_lat=None, min_lat=None, model_name='ACCESS-ESM1-5', test_file=None):
    
    # Technical setup
    SSPs = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    main_data_dir = '/nird/home/johannef/Masterthesis_S23 DataFiles/Annualclimatologies'
    main_save_dir_temporal = '/nird/home/johannef/Masterthesis_S23 DataFiles/Annualclimatologies'
    main_save_dir_spatial = '/nird/home/johannef/Masterthesis_S23 DataFiles/AnnualGlobalClimatologies'
    raw_masks_dir = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/masks/'
    new_masks_dir = '/nird/home/johannef/Masterthesis_S23 DataFiles/masks/'
    test_file = test_file if test_file is not None else open_dummy_file(var)
    savename_new_mask_name = mask_name + '_' + model_name + '.nc'
    # Create mask 
    if excisting_mask is False:
        if region_abriviations is not None:
            mask = create_mask_from_AR6_abriviations(region_abriviations, var, mask_name, test_file)
        elif min_lat and max_lat is not None:
            mask = create_latitude_mask(min_lat, max_lat, var, mask_name, test_file)
        else:    
            raise ValueError('Gi regionforkortelser for regionmask, gi min eller max lat for latitude.')
        # save mask 
        savename_new_mask_name = mask_name + '_' + model_name + '.nc'
        file_handler.save_dataset_to_netcdf(mask, savename_new_mask_name, new_masks_dir)


    # Run global climatologies 
    weights = np.cos(np.deg2rad(test_file['lat']))
    weights.name = "weights"
    if mask_name == 'land_mask':
        mask = open_landmask()
    elif mask_name == 'sea_mask':
        mask = open_seamask()
    else:
        mask = file_handler.read_netcdf_to_xr(directory=new_masks_dir, file_name=savename_new_mask_name)
    run_global_climatology(mask, mask_name, weights, main_data_dir, main_save_dir_temporal, main_save_dir_spatial, [var], SSPs)

    return mask
# %%
