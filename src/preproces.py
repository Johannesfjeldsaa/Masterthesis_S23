"""
written by: Johannes Fjeldså

This module contains functions for preprocessing data.
Import resources:
- https://docs.xarray.dev/en/stable/user-guide/time-series.html
- https://docs.xarray.dev/en/stable/user-guide/weather-climate.html

"""

import os
import datetime

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


class Handle_Files:

    def __init__(self, working_dir=None):
        self.working_dir = working_dir if working_dir is not None else os.getcwd()

        # Where to save the figures and data files
        self.project_results_dir = self.working_dir + " Results"
        self.results_figure_dir = self.working_dir + " Results/FigureFiles"
        self.data_dir = self.working_dir + " DataFiles/"

        if not os.path.exists(self.project_results_dir):

            os.mkdir(self.project_results_dir)

        if not os.path.exists(self.results_figure_dir):
            os.makedirs(self.results_figure_dir)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def __str__(self):
        return f"Working directory: {self.working_dir}"

    def image_path(self, fig_id):
        return os.path.join(self.results_figure_dir, fig_id)

    def data_path(self, dat_id):
        return os.path.join(self.data_dir, dat_id)

    def save_fig(self, fig_id):
        plt.savefig(self.image_path(fig_id) + ".png", format='png')

    def read_netcdf_to_xr(self, file_path=None, directory=None, file_name=None):
        """
        Read a NetCDF file and return its content as an xarray dataset.

        Parameters:
        - file_path (str): Path to the NetCDF file to be read. If not provided provide directory and file_name.
        - directory (str): Path to the directory where the file is located.
        - file_name (str): Name of the NetCDF file to be read.

        Returns:
        - xr.Dataset: An xarray dataset containing the data from the NetCDF file.
        """
        if file_path is None:
            try:
                directory = directory if directory is not None else self.data_dir
                file_path = os.path.join(directory, file_name)
            except Exception as e:
                print(f"Error reading the NetCDF file: {e}")
                return None
        return xr.open_dataset(file_path)

    def get_all_filenames_in_dir(self, directory, condition=None):
        """
        Get all the filenames in a directory that satisfy a condition.

        Parameters:
        - directory (str): Path to the directory where the files are located.
        - condition (function): A function that takes a filename as input and
        returns a boolean value indicating whether the filename satisfies the
        condition.

        Returns:
        - list: A list of filenames that satisfy the condition.
        """
        filenames = os.listdir(directory)
        if condition is None:
            return filenames
        else:
            return [filename for filename in filenames if condition(filename)]

    def get_all_netcdf_files_in_dir(self, directory):
        """
        Get all the NetCDF files in a directory.

        Parameters:
        - directory (str): Path to the directory where the files are located.

        Returns:
        - list: A list of filenames that satisfy the condition.
        """
        return self.get_all_filenames_in_dir(directory, condition=lambda filename: filename.endswith(".nc"))

    def save_dataset_to_netcdf(self, dataset, file_name, directory=None):
        """
        Save an xarray dataset to a NetCDF file.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.
        - directory (str): Path to the directory where the file should be saved.
        - file_name (str): Name of the NetCDF file.
        """
        try:
            directory = directory if directory is not None else self.data_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_path = os.path.join(directory, file_name)
            dataset.to_netcdf(save_path, mode="w")
            return save_path
        except Exception as e:
            print(f"Error saving the NetCDF file: {e}")


class Preprocess_Climate_Data:

    def __init__(self, working_dir=None, model=None, path_to_mask=None):
        self.file_handler = Handle_Files(working_dir=working_dir)
        self.model = model if model is not None and model in ["ACCESS-ESM1-5"] else None
        
        # Set path to mask to by provided path or default path from model 
        self.path_to_mask = path_to_mask if path_to_mask is not None else None

        if self.path_to_mask is None:
            if self.model == "ACCESS-ESM1-5":
                self.path_to_mask = '/nird/projects/NS9188K/bjornhs/ACCESS-ESM1-5/masks'
            

    def covert_from_K_to_C(self, dataset, var_name, reverse=False):
        """
        Convert the temperature from Kelvin to Celsius.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.

        Returns:
        - xr.Dataset: An xarray dataset with the temperature converted from Kelvin to Celsius.
        """
        if reverse:
            dataset[var_name] = dataset[var_name] + 273.15
            dataset[var_name].attrs["units"] = "K"
        else:
            dataset[var_name] = dataset[var_name] - 273.15
            dataset[var_name].attrs["units"] = "deg C"

        return dataset

    def convert_from_Pa_to_hPa(self, dataset, var_name, reverse=False):
        """
        Convert the pressure from Pascal to hectoPascal.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.

        Returns:
        - xr.Dataset: An xarray dataset with the pressure converted from Pascal to hectoPascal.
        """
        if reverse:
            dataset[var_name] = dataset[var_name] * 100
            dataset[var_name].attrs["units"] = "Pa"
        else:
            dataset[var_name] = dataset[var_name] / 100
            dataset[var_name].attrs["units"] = "hPa"

        return dataset
    
    def check_mask_compatability(self, xr_file, var_name, mask):
        if np.array_equal(xr_file[var_name].lat, mask.lat):
            lat_compatabiliy = True
        else:
            lat_compatabiliy = False
        
        if np.array_equal(xr_file[var_name].lon, mask.lon):
            lon_compatability = True
        else:
            lon_compatability = False
        
        if lat_compatabiliy and lon_compatability:
            return True 
        else: 
            raise ValueError(f'Mask is not compatible with xr_file\n'
                             f'lat compatibility: {lat_compatabiliy}\n'
                             f'lon compatability: {lon_compatability}')

    

    def apply_mask(self, xr_file, var_name,  mask_name, path_to_mask=None):
        
        if mask_name is None:
            raise ValueError("A mask name must be provided.")
        
        if path_to_mask and self.path_to_mask is None:
            raise ValueError("A path to the mask must be provided. Either through the constructor or as an argument.")
        elif path_to_mask is not None:
            if self.path_to_mask is not None and path_to_mask != self.path_to_mask:
                print(f"Path to mask is reset from {self.path_to_mask} to {path_to_mask}")
            
            self.path_to_mask = path_to_mask
            
        mask_path = '/'.join([self.path_to_mask, mask_name])
        print(f"Applying mask from {mask_path}")

        mask = xr.open_dataset(mask_path)

        if self.check_mask_compatability(xr_file, var_name, mask):
            print('mask is compatible')


        return mask

    def create_global_mean(self, dataset, var_name,
                           write_to_original_dataset=True,
                           save_to_dataset=False,
                           file_name=None,
                           directory=None):
        """
        Create a global mean from a dataset. averaging over horizontal dimensions.
        1. Average over the horizontal dimensions.
        2. If write_to_dataset is True, write the global mean to the dataset but drop the horizontal coordinates.
        else keep the global mean as xr.DataArray.
        3. Return original dataset with global mean as variable or just the global mean.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.
        - var_name (str): The name of the variable to create the global mean from.
        - write_to_original_dataset (bool): Whether or not to write the global mean to the original dataset.
        - save_to_dataset (bool): Whether or not to create a new file saved in the dataset directory. If True,
        the dataset will be saved in the dataset directory with the name original_name+"_global_mean.nc".
        The new dataset will be opened and returned as an xarray dataset.

        Returns:
        - xr.Dataset: An xarray dataset with a global mean.
        """

        global_mean = dataset[var_name].mean(dim=['lon', 'lat'])
        global_mean.attrs["units"] = dataset[var_name].attrs["units"]

        if write_to_original_dataset:
            var_name = var_name + "_global_mean"
            dataset[var_name] = global_mean
            climatology_dataset = dataset
        else:
            global_mean.name = var_name + "_global_mean"
            climatology_dataset = global_mean

        if save_to_dataset:
            if file_name is None:
                file_name = input("Please provide the original file name that the global_mean is calculated from.")

            file_name = file_name + "_global_mean.nc"
            save_path = self.file_handler.save_dataset_to_netcdf(climatology_dataset, file_name, directory=directory)
            climatology_dataset = self.file_handler.read_netcdf_to_xr(file_path=save_path)

        return climatology_dataset

    # Common function for saving climatology dataset
    def save_climatology_dataset(self, dataset, climatology_type, file_name, directory, is_original_name=True, re_open=True):
        """
        Function for saving climatology dataset to file.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.
        - climatology_type (str): The type of climatology that is saved.
        - file_name (str): The name of the file to save the dataset to.
        - directory (str): The directory to save the dataset to. If None, the dataset will be saved in the dataset directory of the file_handler.
        - is_original_name (bool): Whether or not to use the original file name as the name of the climatology file. If True, the climatology_type 
        will be appended to the file name. If False, it assumed that the file_name is the wanted name of the climatology file.

        Returns:
        """
        if is_original_name:
            if file_name is not None:
                if file_name.endswith('.nc'):
                    file_name = file_name[:-3]
                
                file_name = file_name + "_" + climatology_type + "_climatology.nc"
            else:
                raise ValueError("File name must be provided when saving to dataset")
        
        if directory is None:
            directory = self.file_handler.data_dir
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
                

        
        save_path = self.file_handler.save_dataset_to_netcdf(dataset, file_name, directory=directory)
        if re_open:
            climatology_dataset = self.file_handler.read_netcdf_to_xr(file_path=save_path)
            return climatology_dataset
        else:
            return None
            

    def create_temporal_climatology(self, dataset, var_name=None,
                                    climatology_type="monthly",
                                    save_to_dataset=False,
                                    file_name=None,
                                    directory=None,
                                    is_original_name=True, 
                                    re_open=True):
        """
        Create a time climatology from a dataset. averaging over time dimension.
        1. Average over the time dimension.
        2. Write the climatology to new dataset with new time dimension.


        Resources:
        - https://xcdat.readthedocs.io/en/latest/generated/xarray.Dataset.temporal.climatology.html
        - https://docs.xarray.dev/en/stable/examples/monthly-means.html

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.
        - var_name (str): The name of the variable to create the climatology from. If it is not provided
        the dataset is assumed to be a xarray.Dataset with a single variable.
        - climatology_type (str): The type of climatology to create.
        Can be "weekly", "monthly", "seasonal", "yearly", or "decade".

        returns: xr.DataArray: An xarray dataset with a temporal climatology.

        Note:
            - The climatology is calculated by averaging over the time dimension. Seasonal will be skewed if the dataset
            does not start at the beginning of a season or contains leep years.
            - Method assumes that the time dimension is named "time". if not, the method will fail.
            -> add flexility by allowing the user to specify the time dimension name? FIXED
            - should add option for custom seasonal periods.
                - https://docs.xarray.dev/en/stable/examples/monthly-means.html
        """

        time_axis_name = dataset.coords['time'].name

        # Group by the appropriate dimension based on climatology_type
        if climatology_type == "monthly":
            key_word = time_axis_name + ".month"
        elif climatology_type == "seasonal":
            key_word = time_axis_name + ".season"
        elif climatology_type == "yearly":
            key_word = time_axis_name + ".year"
        elif climatology_type == "decade":
            key_word = time_axis_name + ".decade"
        else:
            raise ValueError("Invalid climatology_type. Must be 'monthly', 'seasonal', 'yearly', or 'decade'.")

        if var_name is not None:
            climatology = dataset[var_name].groupby(key_word).mean(dim=time_axis_name)
            climatology_dataset = xr.Dataset({var_name: climatology})
        else:
            climatology = dataset.groupby(key_word).mean(dim=time_axis_name)
            climatology_dataset = xr.Dataset({dataset.name: climatology})

        if save_to_dataset:
            climatology_dataset = self.save_climatology_dataset(climatology_dataset, 
                                                                climatology_type, 
                                                                file_name, 
                                                                directory, 
                                                                is_original_name, 
                                                                re_open)

        return climatology_dataset

    def create_spatial_climatology(self, dataset, var_name,
                                   climatology_type="zonal",
                                   save_to_dataset=False,
                                   file_name=None,
                                   directory=None,
                                   is_original_name=True, 
                                   re_open=True):   
                                    
        """
        Create a spatial climatology from a dataset by averaging over horizontal dimensions.
        If global_climatology is True, the climatology will be calculated globally by averaging over all dimensions.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.
        - var_name (str): The name of the variable to create the climatology from.
        - climatology_type (str): The type of climatology to create. Either "zonal" or "meridional".
        - save_to_dataset (bool): Whether or not to create a new file saved in the dataset directory. If True,
        the dataset will be saved in the dataset directory with the name original_name+"_zonal/meridional.nc".
        - file_name (str): The name of the original file that the climatology is calculated from. If save_to_dataset
        this must be provided.
        - directory (str): The directory to save the climatology dataset. If save_to_dataset is True, this must be provided.
        - global_climatology (bool): Whether to calculate a global climatology. If True, the climatology will be calculated
        by averaging over all dimensions.

        Returns:
        - xr.Dataset: An xarray dataset with a spatial climatology.
        """
        if climatology_type == "global":
            climatology = dataset[var_name].mean(dim=["lat", "lon"])
        else:
            if climatology_type == "zonal":
                key_word = "lon"
            elif climatology_type == "meridional":
                key_word = "lat"
            else:
                raise ValueError("Invalid climatology_type. Must be 'zonal' or 'meridional'."
                                    "For global climatology, set global_climatology=True.")

            climatology = dataset[var_name].mean(dim=[key_word])

        climatology_dataset = xr.Dataset({var_name: climatology})

        if save_to_dataset:
            climatology_dataset = self.save_climatology_dataset(climatology_dataset, 
                                                                climatology_type, 
                                                                file_name, 
                                                                directory, 
                                                                is_original_name, 
                                                                re_open)

        return climatology_dataset

    def create_spatial_temporal_climatology(self, dataset, var_name,
                                            spatial_climatology_type="zonal",
                                            temporal_climatology_type="monthly",
                                            save_to_dataset=False,
                                            file_name=None,
                                            directory=None, 
                                            is_original_name=True, 
                                            re_open=True):
        """
        """
        file_name = file_name if file_name is not None else dataset.name
        spatial_climatology = self.create_spatial_climatology(dataset, var_name,
                                                              climatology_type=spatial_climatology_type,
                                                              save_to_dataset=False)
        climatology_dataset = self.create_temporal_climatology(spatial_climatology, var_name,
                                                               climatology_type=temporal_climatology_type,
                                                               save_to_dataset=False)
        if save_to_dataset:
            climatology_type = spatial_climatology_type + "_" + temporal_climatology_type
            climatology_dataset = self.save_climatology_dataset(climatology_dataset, 
                                                                climatology_type, 
                                                                file_name, 
                                                                directory, 
                                                                is_original_name, 
                                                                re_open)

        return climatology_dataset        



    def standardize_names(self, dataset):
        pass
        # time_axis_names = ["time", "t", "date"]
        # lat_axis_names = ["lat", "latitude"]
        # lon_axis_names = ["lon", "longitude"]
        # elev_axis_names = ["plev"]

        # return dataset

    def seasonal_trend_decompose(self):
        pass

    def detrend(self, dataset):
        """
        Detrend the dataset.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.

        Returns:
        - xr.Dataset: An xarray dataset with the trend removed.
        """
        pass

    def standardize_names(self, dataset):
        pass
        # time_axis_names = ["time", "t", "date"]
        # lat_axis_names = ["lat", "latitude"]
        # lon_axis_names = ["lon", "longitude"]
        # elev_axis_names = ["plev"]

        # return dataset

    def seasonal_trend_decompose(self):
        pass

    def detrend(self, dataset):
        """
        Detrend the dataset.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.

        Returns:
        - xr.Dataset: An xarray dataset with the trend removed.
        """
        pass

    def remove_outliers(self, dataset):
        """
        Remove outliers from the dataset.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.

        Returns:
        - xr.Dataset: An xarray dataset with the outliers removed.
        """
        pass