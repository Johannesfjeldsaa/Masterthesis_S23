'''

'''
import cmocean
import matplotlib.pyplot as plt 
import xarray as xr 
import cartopy.crs as ccrs 
import regionmask
from src.preproces import *
from xclim import ensembles
import math
from pathlib import Path



file_handler = Handle_Files(working_dir='/nird/home/johannef/Masterthesis_S23 DataFiles')


def plot_on_map(data, ax=None,
                center_lon=0,
                center_lat=0, 
                cmap=None,
                title=None, 
                v_min=None,
                v_max=None,
                migrate_colorbar=False, 
                percipitation=False, 
                temperature=False, 
                include_regionmask=False,
                map_projection='Robinson'):
    """
    Parameters:
    - data (xr.DataArray): The data to be plotted. Horizontal 2D data at a single time step.
    - center_lat (float): The latitude for the center of the plot.
    - center_lon (float): The longitude for the center of the plot.
    - cmap (str or Colormap): The colormap to be used for the plot.
    - title (str): The title of the plot.
    - v_min (float): The minimum value for the colorbar.
    - v_max (float): The maximum value for the colorbar.
    - map_projection (str): The map projection to be used.
    """
    cmap_dict = {'percipitation': {'abs': cmocean.cm.tempo, 
                                   'pctg': cmocean.cm.tarn, 
                                   'anomaly': cmocean.cm.tarn},
                 'temperature': {'abs': plt.cm.coolwarm, 
                                 'anomaly': cmocean.cm.balance}}

    if cmap is None:
        if percipitation is not False:
            if percipitation not in list(cmap_dict['percipitation'].keys()):
                raise ValueError(f"percipitation must be either in {cmap_dict['percipitation'].keys()}") 
            cmap = cmap_dict['percipitation'][percipitation]
        elif temperature is not False:
            if temperature not in list(cmap_dict['temperature'].keys()):
                raise ValueError(f"Temperature must be either in {cmap_dict['temperature'].keys()}") 
            cmap = cmap_dict['temperature'][temperature]
        else:
            cmap = 'viridis'
    
   
    projection_dict = {'Robinson': dict(projection=ccrs.Robinson(central_longitude=center_lon), facecolor="gray"),
                       'PlateCarree': dict(projection=ccrs.PlateCarree(central_longitude=center_lon), facecolor="gray"),
                       'Mollweide': dict(projection=ccrs.Mollweide(central_longitude=center_lon), facecolor="gray"),
                       'LambertConformal': dict(projection=ccrs.LambertConformal(central_longitude=center_lon), facecolor="gray"),
                       'EuroPP': dict(projection=ccrs.EuroPP()), 
                       'NearsidePerspective': dict(projection=ccrs.NearsidePerspective(central_longitude=center_lon, central_latitude=center_lat), facecolor="gray")}

    settings = {'transform': ccrs.PlateCarree(),
                'cmap': cmap,
                'vmin': v_min,
                'vmax': v_max}

    if migrate_colorbar:
        settings['add_colorbar'] = False
    else:
        settings['cbar_kwargs'] = {"orientation": "horizontal", "shrink": 0.7}
        settings['robust'] = True
    
    if ax is not None:
        settings['ax'] = ax
    else:    
        if map_projection not in list(projection_dict.keys()):
            raise ValueError(f"Map projection must be in {list(projection_dict.keys())}")
        settings['subplot_kws'] = projection_dict[map_projection]

    p = data.plot(**settings)
    
    if include_regionmask:
        regionmask.defined_regions.ar6.all.plot(ax=p.axes, add_label=False)
    p.axes.set_global()
    p.axes.coastlines()

    if title is None:
        title = "Map plot"
    if ax is None:
        plt.title(title)

        plt.show()
    else:
        ax.set_title(title)

def legend_without_duplicate_labels(fig):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    fig.legend(*zip(*unique), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

def plot_annual_global_ensambles(main_data_dir, SSPs, variable, mask_names=None, temporal_range=None):

    if mask_names is None:
        mask_names = file_handler.get_all_filenames_in_dir(main_data_dir)
    
    num_masks = len(mask_names)
    num_cols = 3
    num_rows = math.ceil(num_masks / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))#, sharex='col')
    axs = axs.flatten()
    fig2, axs2 = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))#, sharex='col')
    axs2 = axs2.flatten()

    fig.suptitle('SSP development')
    fig2.suptitle('SSP development (Meanscaled)')


    colors = plt.cm.coolwarm(np.linspace(0, 1, len(SSPs)))
    color_map = dict(zip(SSPs, colors))

    for i, mask in enumerate(mask_names):
        ax = axs[i]
        ax2 = axs2[i]

        ax.set_ylabel(variable)
        ax.set_title(f'{variable} ({mask})')
        ax2.set_ylabel(variable)
        ax2.set_title(f'{variable} ({mask})')

        means = {}
        stds = {}
        for scenario in SSPs:
            data_dir = '/'.join([main_data_dir, mask, variable, scenario])
            ens = ensembles.create_ensemble(Path(data_dir).glob("*.nc"))
            
            if temporal_range is not None and type(temporal_range) is tuple and len(temporal_range) == 2:
                start_slice = str(temporal_range[0])
                end_slice = str(temporal_range[1])
                ens = ens.sel(year=slice(start_slice, end_slice))
            
            ens_stats = ensembles.ensemble_mean_std_max_min(ens)
            
            ens_mean = ens_stats[f'{variable}_mean']
            ens_std = ens_stats[f'{variable}_stdev']
            ax.plot(ens_stats.year, ens_mean, 
                            label=scenario,
                            color=color_map[scenario])
            ax.fill_between(ens_stats.year, ens_mean - ens_std, ens_mean + ens_std, 
                                    color=color_map[scenario],
                                    alpha=0.5)

            means[scenario] = ens_mean
            stds[scenario] = ens_std
        
        subfig_mean = [scenario_mean.mean().values.item() for scenario_mean in means.values()]
        subfig_mean = np.mean(subfig_mean)

        for scenario in means.keys(): 
            # meanscaled version
            # Kan ikke skalere med gruppegjennomsnittet må benytte gjennomsnittet av alle sspene
            meanscaled_ens_mean = means[scenario] - subfig_mean
            ax2.fill_between(ens_stats.year, meanscaled_ens_mean - stds[scenario], meanscaled_ens_mean + stds[scenario], 
                                    color=color_map[scenario],
                                    alpha=0.5)
            ax2.plot(ens_stats.year, meanscaled_ens_mean, 
                            label=scenario,
                            color=color_map[scenario])
            

    legend_without_duplicate_labels(fig)
    legend_without_duplicate_labels(fig2)
    plt.tight_layout()
    plt.show()

def exctract_data_for_mapplots_for_investigation(ensambles, var, years, SSPs):

    all_plot_data_overview = {}
    all_plot_data_std = {}
    all_plot_data_anom = {}
    # max_values for vmax
    scenario_q = -np.inf
    anomaly_q = -np.inf


    for scenario in SSPs:
        # Dictionaries for scenario plot data
        scenario_plot_data_overview = {key: {} for key in years}
        scenario_plot_data_std = {key: {} for key in years} 
        scenario_plot_data_anom = {key: {} for key in ['baseline']+years}
        # Scenario baseline for anomaly constuction
        scenario_baseline = ensambles[scenario].sel(year=slice(2015, 2030)).mean(dim='realization').mean(dim='year')[var]
        scenario_plot_data_anom['baseline'][f'{scenario} ensamble mean (2015-2030)'] = scenario_baseline

        for indx, year in enumerate(years):
            # Overview: ensamble mean
            scenario_mean_data = ensambles[scenario].sel(year=year).mean(dim='realization')[var]
            key_overview = f'{scenario} ({year}) ensamble mean'
            scenario_plot_data_overview[year][key_overview] = scenario_mean_data
            scenario_q = np.quantile(scenario_mean_data.values, 0.95) if np.quantile(scenario_mean_data.values, 0.95) > scenario_q else scenario_q
            # Spread: ensamble standard deviation
            scenario_sd_data = ensambles[scenario].sel(year=year).std(dim='realization')[var]
            key_std = f'{scenario} ({year}) ensamble standard deviation'
            scenario_plot_data_std[year][key_std] = scenario_sd_data
            # Anomaly: ensamble mean - scenario_baseline
            anom_plot_data = scenario_mean_data - scenario_baseline
            key_anom = f'{scenario} ({year}) - {scenario} ensamble mean (2015-2030)'
            scenario_plot_data_anom[year][key_anom] = anom_plot_data
            anomaly_q = np.quantile(abs(anom_plot_data.values), 0.95) if np.quantile(abs(anom_plot_data.values), 0.95) > anomaly_q else anomaly_q


        all_plot_data_overview[scenario] = scenario_plot_data_overview   
        all_plot_data_std[scenario] = scenario_plot_data_std  
        all_plot_data_anom[scenario] = scenario_plot_data_anom

    return all_plot_data_overview, all_plot_data_std, all_plot_data_anom, scenario_q, anomaly_q

def plot_mapplots_for_investigation(ensambles, var, years, SSPs, 
                                    v_min_meanplot=None, v_max_meanplot=None, 
                                    v_min_anomplot=None, v_max_anomplot=None):
    
    all_plot_data_overview, all_plot_data_std, all_plot_data_anom, scenario_q, anomaly_q = exctract_data_for_mapplots_for_investigation(ensambles, var, years, SSPs)
    v_min_meanplot = v_min_meanplot if v_min_meanplot is not None else -scenario_q 
    v_max_meanplot = v_max_meanplot if v_max_meanplot is not None else scenario_q
    v_min_anomplot = v_min_anomplot if v_min_anomplot is not None else -anomaly_q
    v_max_anomplot = v_max_anomplot if v_max_anomplot is not None else anomaly_q


    fig, axs = plt.subplots(ncols=len(SSPs), nrows=len(years), figsize=(5*len(SSPs), 5*len(years)), 
                        subplot_kw=dict(projection=ccrs.Robinson(central_longitude=0), facecolor="gray"))
    fig.suptitle(f'{var} ensamble mean', fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust the rect parameter to position the suptitle

    fig_std, axs_std = plt.subplots(ncols=len(SSPs), nrows=len(years), figsize=(5*len(SSPs), 5*len(years)), 
                                    subplot_kw=dict(projection=ccrs.Robinson(central_longitude=0), facecolor="gray"))
    fig_std.suptitle(f'{var} ensemble standard deviation', fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust the rect parameter to position the suptitle

    fig_anom, axs_anom = plt.subplots(ncols=len(SSPs), nrows=len(years)+1, figsize=(5*len(SSPs), 5*(len(years)+1)), 
                                      subplot_kw=dict(projection=ccrs.Robinson(central_longitude=0), facecolor="gray"))
    fig_anom.suptitle(f'{var} anomaly from scenario baseline (2015-2030)', fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust the rect parameter to position the suptitle

    for col, scenario in enumerate(SSPs):
        key_top_row_anom = f'{scenario} ensamble mean (2015-2030)'
        top_row_anom_settings = dict(data=all_plot_data_anom[scenario]['baseline'][key_top_row_anom], 
                                     ax=axs_anom[0, col],
                                     title=key_top_row_anom,
                                     v_min=v_min_meanplot,
                                     v_max=v_max_meanplot,
                                     include_regionmask=True)
        if var == 'tas' or var == 'txxETCCDI':
            top_row_anom_settings['temperature'] = 'abs'
            top_row_anom_settings['v_min'] = v_max_meanplot - 80
        if var == 'pr' or var == 'r5xdayETCCDI':
            top_row_anom_settings['percipitation'] = 'abs'
            top_row_anom_settings['v_min'] = 0 
        plot_on_map(**top_row_anom_settings)
        
        for row, year in enumerate(years):
            ax = axs[row, col]
            ax_std = axs_std[row, col]
            ax_anom = axs_anom[row+1, col]
            
            key_overview = f'{scenario} ({year}) ensamble mean'
            overview_settings = dict(data=all_plot_data_overview[scenario][year][key_overview], 
                                     ax=ax, 
                                     title=key_overview, 
                                     v_min=v_min_meanplot, 
                                     v_max=v_max_meanplot,
                                     include_regionmask=True)
            if var == 'tas' or var == 'txxETCCDI':
                overview_settings['temperature'] = 'abs'
                overview_settings['v_min'] = v_max_meanplot - 80
            if var == 'pr' or var == 'r5xdayETCCDI':
                overview_settings['percipitation'] = 'abs'
                overview_settings['v_min'] = 0 
            plot_on_map(**overview_settings)  

            key_std = f'{scenario} ({year}) ensamble standard deviation'
            plot_on_map(all_plot_data_std[scenario][year][key_std], ax_std, 
                        title=key_std, 
                        include_regionmask=True)   
            
            key_anom = f'{scenario} ({year}) - {scenario} ensamble mean (2015-2030)'
            anom_settings = dict(data=all_plot_data_anom[scenario][year][key_anom], 
                                 ax=ax_anom,
                                 title=key_anom,
                                 v_min=v_min_anomplot,
                                 v_max=v_max_anomplot,
                                 include_regionmask=True)
            if var == 'tas' or var=='txxETCCDI':
                anom_settings['temperature'] = 'anomaly'
            if var == 'pr' or 'r5xdayETCCDI':
                anom_settings['percipitation'] = 'anomaly'
            plot_on_map(**anom_settings)
    
    plt.show()

def disp_external_fig(path_to_image, wanted_height=None, wanted_width=None):
    img = plt.imread(path_to_image)

    if wanted_height is not None and wanted_width is not None:
        height = wanted_height
        width = wanted_width
    
    elif wanted_height is not None:
        height = wanted_height
        width = int(height * img.shape[1] / img.shape[0])
        
    elif wanted_width is not None:
        width = wanted_width
        height = int(width * img.shape[0] / img.shape[1])
    else:
        height = img.shape[0]
        width = img.shape[1]

    
    plt.figure(figsize=(width, height))
    plt.imshow(img)
    plt.axis('off')  # to turn off the axis labels
    plt.show()
