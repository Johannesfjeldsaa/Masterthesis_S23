'''

'''
#import cmocean
import matplotlib.pyplot as plt 
import seaborn as sns
import xarray as xr 
import cartopy.crs as ccrs 
import regionmask
from src.preproces import *
from xclim import ensembles
import math
from pathlib import Path
import imageio.v2 as imageio
import pandas as pd
import sklearn
import cmocean



file_handler = Handle_Files()#working_dir='/nird/home/johannef/Masterthesis_S23')


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
                map_projection='Robinson', 
                plot_masks=False):
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
                                   'anomaly': cmocean.cm.tarn}, # also tarn
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

    if plot_masks:
        settings['alpha'] = 0.5
        
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

    if plot_masks:
        p.axes.stock_img()

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

def plot_annual_global_ensambles(main_data_dir, SSPs, variable, mask_names=None, temporal_range=None, show_fig2=True):


    color_map = {'ssp126': '#1c3354', 
                 'ssp245': '#ebdf4a', 
                 'ssp370': '#f43030', 
                 'ssp585': '#8f2035'}
    label_map = {'ssp126': 'SSP1-2.6', 
                 'ssp245': 'SSP2-4.5', 
                 'ssp370': 'SSP3-7.0', 
                 'ssp585': 'SSP5-8.5'}

    if mask_names is None:
        mask_names = file_handler.get_all_filenames_in_dir(main_data_dir)
    
    num_masks = len(mask_names)
    num_cols = 3
    num_rows = math.ceil(num_masks / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
    axs = axs.flatten()
    
    fig.suptitle('SSP development')


    for i, mask in enumerate(mask_names):
        ax = axs[i]
        ax.grid(True)
        ax.set_ylabel(variable)
        ax.set_title(f'{variable} ({mask})')

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
                            label=label_map[scenario],
                            color=color_map[scenario])
            ax.fill_between(ens_stats.year, ens_mean - ens_std, ens_mean + ens_std, 
                                    color=color_map[scenario],
                                    alpha=0.5)

            means[scenario] = ens_mean
            stds[scenario] = ens_std
        
        if show_fig2:
            fig2, axs2 = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
            axs2 = axs2.flatten()
            fig2.suptitle('SSP development (Meanscaled)')

            for scenario in means.keys(): 
                subfig_mean = [scenario_mean.mean().values.item() for scenario_mean in means.values()]
                subfig_mean = np.mean(subfig_mean)

                # meanscaled version
                # Kan ikke skalere med gruppegjennomsnittet må benytte gjennomsnittet av alle sspene
                meanscaled_ens_mean = means[scenario] - subfig_mean
                axs2.fill_between(ens_stats.year, meanscaled_ens_mean - stds[scenario], meanscaled_ens_mean + stds[scenario], 
                                        color=color_map[scenario],
                                        alpha=0.5)
                axs2.plot(ens_stats.year, meanscaled_ens_mean, 
                                label=scenario,
                                color=color_map[scenario])

            legend_without_duplicate_labels(fig2)
            plt.tight_layout()
            plt.show()

    legend_without_duplicate_labels(fig)
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

def autolabel(ax, ranks):
        rects = ax.patches
        max_height = np.max([rect.get_height() for rect in rects])
        for rect, rank in zip(rects, ranks):
            rotation = 90
            height = rect.get_height()
            if height <= 0.25*max_height and height > 0.1*max_height:
                y = .5*height
            elif height <= 0.1*max_height and height > 0.05*max_height:
                y = .2*height
            elif height < 0.05*max_height:
                y = .8*height
                rotation = 0
            else:
                y = .8*height

            ax.text(rect.get_x() + rect.get_width() / 2., y,
                    rank,
                    ha='center', va='bottom', rotation=rotation, color='black')

def plot_cumulative_mRMR_scores(mRMR_scores_df, title=None, filter_name=None, period=None, 
                                save=False, save_folder=None, save_name=None, return_plot_data=False,
                                include_rank_paste=True, 
                                fontsize=20, y_label=None):
        
    if period is not None:
        mRMR_scores_for_plotting = mRMR_scores_df[(mRMR_scores_df['year'] >= period[0]) & (mRMR_scores_df['year'] <= period[1])]
    else:
        mRMR_scores_for_plotting = mRMR_scores_df
    mRMR_scores_for_plotting = mRMR_scores_for_plotting.drop('year', axis=1).sum().reset_index()
    mRMR_scores_for_plotting.columns = ['var: mask', 'cumulative_mRMR_score']
    mRMR_scores_for_plotting['rank'] = mRMR_scores_for_plotting['cumulative_mRMR_score'].rank(ascending=False)
    mRMR_scores_for_plotting['rank'] = mRMR_scores_for_plotting['rank'].astype(int)
    mRMR_scores_for_plotting['var'] = mRMR_scores_for_plotting['var: mask'].str.split(':').str[0]

    # Plot the sum as a barplot with rank
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
    if period is not None:
        start_year = period[0]
        stop_year = period[1]
    else:
        start_year = mRMR_scores_df['year'].iloc[0]
        stop_year = mRMR_scores_df['year'].iloc[-1]
    
    title = title if title is not None else f'Cumulative {filter_name}-scores {start_year}-{stop_year}'
    fig.suptitle(title, fontsize=fontsize)
    sns.barplot(x=mRMR_scores_for_plotting['var: mask'], y=mRMR_scores_for_plotting['cumulative_mRMR_score'], 
                hue=mRMR_scores_for_plotting['var'])
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='left')  # Rotate labels 90 degrees to the left
    y_label = y_label if y_label is not None else f'Cumulative {filter_name} for var: mask'
    ax.set_ylabel(y_label)
    if include_rank_paste:
        autolabel(ax, mRMR_scores_for_plotting['rank'])
    plt.legend(loc='upper left')
    plt.tight_layout()

    save_folder = save_folder if save_folder is not None else '/nird/home/johannef/Masterthesis_S23 Results/FigureFiles/Feature selection'
    save_name = save_name if save_name is not None else f'cumulative_{filter_name}_scores_{start_year}to{stop_year}.png'
    if save:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(f'{save_folder}/{save_name}')
    
    plt.show()

    if return_plot_data:
        return mRMR_scores_for_plotting

def add_note_at_bottom(fig, note):
    fig.text(0.5, 0.02, note, ha='center', fontsize=12)

def animate_scores_barplot(scores_df, title=None, filter_name=None, save_folder=None, save_name=None, supress_counter=False, period=None):

    save_folder = save_folder if save_folder is not None else '/nird/home/johannef/Masterthesis_S23 Results/FigureFiles/Feature selection/animations'
    given_title = title
    png_files = []
    
    start_year = period[0] if period is not None else np.min(scores_df['year'])
    stop_year = period[1] if period is not None else np.max(scores_df['year'])+1

    for year in scores_df['year']:
        if year > stop_year:
            break
        scores_for_plotting = scores_df[(scores_df['year'] >= start_year) & (scores_df['year'] <= year)]
        scores_for_plotting = scores_for_plotting.drop('year', axis=1).sum().reset_index()
        scores_for_plotting.columns = ['var: mask', 'cumulative_score']
        scores_for_plotting['rank'] = scores_for_plotting['cumulative_score'].rank(ascending=False)
        scores_for_plotting['rank'] = scores_for_plotting['rank'].astype(int)
        scores_for_plotting['var'] = scores_for_plotting['var: mask'].str.split(':').str[0]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        if supress_counter:
            title = given_title if given_title is not None else f'Cumulative {filter_name}-scores'
        else:
            title = f'{given_title} {year}:{scores_df["year"].iloc[-1]}' if given_title is not None else f'Cumulative {filter_name}-scores {year}:{scores_df["year"].iloc[-1]}'

        fig.suptitle(title, fontsize=16)

        sns.barplot(x=scores_for_plotting['var: mask'], y=scores_for_plotting['cumulative_score'], 
                    hue=scores_for_plotting['var'])
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='left')  # Rotate labels 90 degrees to the left
        plt.gca().set_xlabel(None)
        ax.set_ylabel('Cumulative score for var: mask')
        autolabel(ax, scores_for_plotting['rank'])
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        # saving 
        file_name = f'fig_{year}.png'
        png_files.append(file_name)
        if f'raw_imgs_{filter_name}' not in os.listdir(save_folder):
            os.makedirs('/'.join([save_folder, f'raw_imgs_{filter_name}']))
        plt.savefig('/'.join([save_folder, f'raw_imgs_{filter_name}', file_name]))
        plt.close()

    # create gif
    save_name = save_name if save_name is not None else f'{filter_name}_scores_animation.gif'
    with imageio.get_writer('/'.join([save_folder, save_name]), mode='i', fps=2) as writer:
        for file_name in png_files:
            image = imageio.imread('/'.join([save_folder, f'raw_imgs_{filter_name}', file_name]))
            writer.append_data(image)
            os.remove('/'.join([save_folder, f'raw_imgs_{filter_name}', file_name]))
    
    os.removedirs('/'.join([save_folder, f'raw_imgs_{filter_name}']))


def plot_cm(y_test, y_pred, cm_title, best_parameters):
    
    fig, ax = plt.subplots()
    f1_score = sklearn.metrics.f1_score(y_test, y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=14)
    sns.heatmap(cm, ax=ax, cmap=cmap, norm=norm, annot=True, fmt='d')

    ax.set_xticklabels([rev_target_mapping[tick] for tick in [0, 1]])
    ax.set_yticklabels([rev_target_mapping[tick] for tick in [0, 1]])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(cm_title)


    plt.text(1, 2.4, 
            f'F1 Score: {f1_score:.2f},      Accuracy: {accuracy:.2f}', 
            fontsize=12, ha='center')
    plt.text(1, 2.6,
            f'Best parameters: {best_parameters}',
            fontsize=12, ha='center')
    
    plt.show()

    return fig

def plot_performance(classification_summaries, metric, 
                     years=None, model_name=None, title=None, 
                     spread=False, summary_subplot_for_spread=True, 
                     include_train=False, 
                     report_performance=False, 
                     report_crossing=False,
                     notitle=False, 
                     turn_on_subplot_legend=True):

    years = years if years is not None else list(next(iter(classification_summaries.values())).keys()) # get the years from the values of the first feature_comb 
    if isinstance(report_performance, list):
        report_years = report_performance
    elif report_performance:
        report_years = [2036, 2037, 2038, 2039, 2040]
    else:
        report_years = None

    if report_years is not None:
        if report_crossing:
            raise ValueError('Cannot report performance and crossing at the same time')
        
        if all(number in years for number in report_years):
            reported_performances = []
        else:
            raise ValueError(f'The years {report_years} must be in the data to report performance')

    if isinstance(report_crossing, float):
        if report_crossing < 0 or report_crossing > 1:
            raise ValueError('The crossing value must be between 0 and 1')
        reported_crossings = []
    elif report_crossing:
        report_crossing = 0.8
        reported_crossings = []
    else:
        report_crossing = None
    

    plot_data = pd.DataFrame(columns=['Year', metric, 'feature_comb_key'])

    for feature_comb_key, classification_summary in classification_summaries.items():
        metric_in_feature_comb = []
        metric_in_feature_comb_training = []
        year_list = []
        for year in years:
            if year in classification_summary.keys():
                metric_in_feature_comb = metric_in_feature_comb + classification_summary[year][metric].tolist()
                year_list = year_list + [year for _ in range(len(classification_summary[year][metric].tolist()))]
                if include_train:
                    metric_in_feature_comb_training = metric_in_feature_comb_training + classification_summary[year][f'training_{metric}'].tolist()
        
        feature_comp_plot_data = pd.DataFrame({'Year': year_list,
                                               metric: metric_in_feature_comb, 
                                               'feature_comb_key': [feature_comb_key for _ in range(len(metric_in_feature_comb))]})
        if include_train:
            feature_comp_plot_data[f'training_{metric}'] = metric_in_feature_comb_training

        plot_data = pd.concat([plot_data, feature_comp_plot_data], ignore_index=True)

    if spread:
        if summary_subplot_for_spread:
            num_subplots = len(classification_summaries)+1 
            summary_plot = True
        else:
            num_subplots = len(classification_summaries)
            summary_plot = False
    else:
        num_subplots = 1
        summary_plot = True
    
    fig, axs = plt.subplots(1, num_subplots,  figsize=(4*num_subplots, 4), sharey=True)
    axs = axs.flatten() if num_subplots > 1 else [axs]

    if not notitle:
        title = title if title is not None else f'{model_name.capitalize()} {metric} development' # generate title if not given
        fig.suptitle(title, fontsize = 22)  # Add the suptitle
    
    
    

    full_palette = ['blue', 'red', 'green', 'purple']
    for i, ax in enumerate(axs):
        
        if i == np.shape(axs)[0]-1 and summary_plot:
            subset = plot_data
            palette = full_palette
            turn_on_legend = False 
        else:
            subset = plot_data[plot_data['feature_comb_key'] == list(classification_summaries.keys())[i]]
            palette=[full_palette[i]]
            turn_on_legend = True if turn_on_subplot_legend else False
            if report_performance:
                reported_performance = subset[subset['Year'].isin(report_years)][metric].mean()
                reported_performances.append(reported_performance)

            if report_crossing:
                yearly_mean = subset.groupby('Year')[metric].mean()
                reported_crossing = yearly_mean.index[np.where(yearly_mean < 0.8)[0][-1]]
                reported_crossings.append(reported_crossing)

        ax.set_xlabel('Year')
        if i == 0:
            ax.set_ylabel(metric)
        ax.set_xlim(min(years), max(years))
        ax.set_ylim(0, 1.05)
        x_tick_dist = 5 if spread else 3
        ax.set_xticks(np.arange(min(years), max(years)+1, x_tick_dist))
        ax.grid(True)
        sns.lineplot(
                data=subset, x='Year', y=metric, 
                errorbar='sd', 
                hue='feature_comb_key', 
                ax=ax, 
                palette=palette,
                legend=turn_on_legend,
            )
        
        if i != np.shape(axs)[0]-1 and turn_on_legend:
            sns.move_legend(ax, loc='lower right', frameon=True, title=None)

        if include_train and i != np.shape(axs)[0]-1:
             sns.lineplot(
                data=subset, x='Year', y=f'training_{metric}', 
                errorbar='sd', 
                ax=ax, 
                color='gray',
                linestyle='dashed'
                )

    if report_years:
        labels = ['Training']+[f"{key} \n{metric}={performance:.4f}" for key, performance in zip(list(classification_summaries.keys()), reported_performances)]
    elif report_crossing:
        labels = ['Training']+[f"{key} \n{metric}>={report_crossing} in {crossing:.0f}" for key, crossing in zip(list(classification_summaries.keys()), reported_crossings)]
    else:
        labels = ['Training']+list(classification_summaries.keys())
    handles = [plt.Line2D([0], [0], color=color, label=label, linestyle='dashed' if i == 0 else 'solid') for i, (color, label) in enumerate(zip(['gray']+full_palette, labels))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5)  # Put the legend below the plot        

    plt.tight_layout()
    plt.show()

def plot_roc_curve(roc_information, years=None, model_name=None, title=None, 
                   spread=False, summary_subplot_for_spread=False , notitle=False,):
    """
    Plots the ROC-curves for the given roc_information. If spread is True, the ROC-curves will be plotted in separate subplots.
    If summary_subplot_for_spread is True, a summary plot will be added to the end of the subplots. If spread is False,
    a single plot will be generated. 
    linestyle documentation: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html

    Parameters:
    - roc_information (dict): A dictionary containing the roc-information for each feature_combination.
    - years (list): A list of years to plot the ROC-curves for. If None, the years from the first feature_combination will be used.
    - model_name (str): The name of the model to be used in the title. If None, the title will be 'ROC-curves'.
    - title (str): The title of the plot. If None, the title will be generated.
    - spread (bool): If True, the ROC-curves will be plotted in separate subplots. If False, a single plot will be generated.
    - summary_subplot_for_spread (bool): If True, a summary plot will be added to the end of the subplots. If False, no summary plot will be added.

    Returns:
    - None
    """
    
    years = years if years is not None else list(next(iter(roc_information.values())).keys()) # get the years from the values of the first feature_comb 
    base_fpr = list(roc_information.values())[0][years[0]]['plot_data']['base_fpr']

    if spread:
        if summary_subplot_for_spread:
            num_subplots = len(roc_information)+1 
            summary_plot = True
        else:
            num_subplots = len(roc_information)
            summary_plot = False
    else:
        num_subplots = 1
        summary_plot = True
    
    fig, axs = plt.subplots(1, num_subplots,  figsize=(4*num_subplots, 4), sharey=True)
    axs = axs.flatten() if num_subplots > 1 else [axs]   
    if not notitle:
        title = title if title is not None else f'{model_name.capitalize()} ROC-curves' # generate title if not given
        fig.suptitle(title, fontsize = 22)  # Add the suptitle
    

    full_palette = ['blue', 'red', 'green', 'purple']

    for i, ax in enumerate(axs):
        
        if i == np.shape(axs)[0]-1 and summary_plot:
            print('psyce... Not impolemented yet!')
        else:
            feature_comb_key = list(roc_information.keys())[i]
            roc_info = list(roc_information.values())[i]
            mean_tprs_dict = {year: roc_info[year]['plot_data']['mean_tprs'] for year in years}
            tprs_upper_dict = {year: roc_info[year]['plot_data']['tprs_upper'] for year in years}
            tprs_lower_dict = {year: roc_info[year]['plot_data']['tprs_lower'] for year in years}
            
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        
        alphas = np.linspace(0.1, 0.4, len(years))
        linestyles_dict = {0: 'solid', 1: 'dashed', 2: (0, (5, 10)), 3: (0, (1, 10))}
        linestyles = [linestyles_dict[i] for i in range(len(years))]
        linestyles.reverse()
               
        for j, year in enumerate(years):
            auc = sklearn.metrics.auc(base_fpr, mean_tprs_dict[year])
            ax.plot(
                base_fpr, 
                mean_tprs_dict[year],
                label=f'{year} (AUC = {auc:.4f})', 
                color=full_palette[i],
                linestyle=linestyles[j]
            )
            ax.fill_between(
                base_fpr, 
                tprs_lower_dict[year], 
                tprs_upper_dict[year], 
                color=full_palette[i], 
                alpha=alphas[j]
            )

        ax.plot(base_fpr, base_fpr, lw=1, color='black')  # Plot the chance line
        ax.legend(loc='lower right')
        
    labels = ['chance']+list(roc_information.keys())
    handles = [plt.Line2D([0], [0], color=color, label=label) for color, label in zip(['black']+full_palette, labels)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5)  # Put the legend below the plot        
    
    plt.tight_layout()
    plt.show()
    

def exctract_hyperparameter(target_summaries, hyperparameter_name, years):
    hyperparameter_values = pd.DataFrame(
        columns=['feature_combination', 'year', hyperparameter_name]
    )
    
    for feature_comb_key, target_summary in target_summaries.items():
        for year in years:
            model_column = target_summary[year].model
            hyperparameter_values_year = []
            for row in model_column:
                hyperparameter_values_year.append(row[hyperparameter_name])
            
            feature_comb_key_list = [feature_comb_key for _ in range(len(hyperparameter_values_year))]
            year_list = [year for _ in range(len(hyperparameter_values_year))]
            
            hyperparameter_values_for_fc_y = pd.DataFrame(
                {'feature_combination': feature_comb_key_list,
                 'year': year_list, 
                 hyperparameter_name: hyperparameter_values_year}
            )

            hyperparameter_values = pd.concat([hyperparameter_values, hyperparameter_values_for_fc_y], ignore_index=True)
    
    return hyperparameter_values

def plot_hyperparameters(target_summaries, model_name, years, param_grid):

    for hyperparameter_name, hyperparameter_grid in param_grid.items():
        if len(hyperparameter_grid) > 1:        
            full_palette = ['blue', 'red', 'green', 'purple']

            hyperparameter_values = exctract_hyperparameter(target_summaries, hyperparameter_name, years)
        
            fig, axs = plt.subplots(nrows=len(years), ncols=len(target_summaries.keys()), 
                                    figsize=(12, 8), 
                                    sharey=True,)
            fig.suptitle(f'Tune distribution for "{hyperparameter_name}" in {model_name}', fontsize=16)    
                
            for row, year in enumerate(years):
                for col, feature_combination in enumerate(hyperparameter_values.feature_combination.unique()):

                    ax = axs[row, col]

                    hyperparameter_values_year = hyperparameter_values[
                        (hyperparameter_values.year == year) & 
                        (hyperparameter_values.feature_combination == feature_combination)
                    ]
                    if len(hyperparameter_grid) >= 20:
                        sns.histplot(
                            data=hyperparameter_values_year, 
                            x=hyperparameter_name, 
                            ax=ax, 
                            color=full_palette[col],
                            bins=int(len(hyperparameter_grid/5))
                        )
                    else:
                        sns.countplot(
                            data=hyperparameter_values_year, 
                            x=hyperparameter_name, 
                            ax=ax, 
                            color=full_palette[col],
                            order=hyperparameter_grid
                        )
                    
                    ax.set_title(f'{feature_combination} ({year})', fontsize=10)
                    ax.set_ylim(0, 50)
            
            labels = list(target_summaries.keys())
            handles = [plt.Line2D([0], [0], color=color, label=label) for i, (color, label) in enumerate(zip(full_palette, labels))]

            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5)  # Put the legend below the plot        
            plt.tight_layout()
            plt.show()    

from sklearn.metrics import confusion_matrix
import numpy as np 
import pandas as pd


def calc_mean_cm(list_of_cms):
    cm_list = np.stack(list_of_cms, axis=0)
    mean_cm = np.mean(cm_list, axis=0)
    return mean_cm


def calculate_cm__featurecomb_year(df):
    y_train_true = list(df['y_train_true'])
    y_train_pred = list(df['y_train_pred'])
    y_test_true = list(df['y_test_true'])
    y_test_pred = list(df['y_test_pred'])

    cm_list_train = []
    cm_list_test = []
    for i in range(len(y_train_true)):
        seed_cm_train = confusion_matrix(y_train_true[i], y_train_pred[i])
        cm_list_train.append(seed_cm_train)
        seed_cm_test = confusion_matrix(y_test_true[i], y_test_pred[i])
        cm_list_test.append(seed_cm_test)

    return cm_list_train, cm_list_test

def calculate_cm_feature_comb(target_summaries, feature_comb_key, report_performance):
    
    train = []
    test = []
    for year in report_performance:
        df = target_summaries[feature_comb_key][year]
        cm_list_train, cm_list_test = calculate_cm__featurecomb_year(df)
        train = train + cm_list_train
        test = test + cm_list_test     
    
    train_cm = calc_mean_cm(train)
    test_cm = calc_mean_cm(test)

    return train_cm, test_cm

def create_cm_labels(test_cm, train_cm):

    test_group_counts = ['{0:0.2f}'.format(value) for value in test_cm.flatten()]
    test_group_percentages = ["{:.1%}".format(cell/sum(row)) for row in test_cm for cell in row ]
    train_group_counts = ['{0:0.2f}'.format(value) for value in train_cm.flatten()]
    train_group_percentages = ["{:.1%}".format(cell/sum(row)) for row in train_cm for cell in row]

    labels = [f"{v1}, {v2}\n ({v4})" for v1, v2, v3, v4 in zip(test_group_counts, test_group_percentages, 
                                                                                          train_group_counts, train_group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    return labels

def plot_cms(target_summaries, model_name, feature_comb_keys=None, report_performance=None, title=None, notitle=False):
    
    
    if report_performance is True:
        print('TRUEResting')
        report_performance = list(target_summaries[feature_comb_key].keys())
    elif type(report_performance) is int:
        print('INTeresting')
        report_performance = [report_performance]
    elif type(report_performance) is list:
        print('LISTeresting')
        report_performance = list(report_performance)
    else:
        raise ValueError('ELSEteresting')
    full_palette = ['blue', 'red', 'green', 'purple']
    scenario_indx_key = {'SSP1-2.6': 0, 'SSP5-8.5': 1}
    included_feature_combs = feature_comb_keys if feature_comb_keys is not None else list(target_summaries.keys())
    
    # Colormaps for the heatmaps
    custom_cmaps = {
        'nomask_baseline': plt.colormaps['Blues'],
        'boruta_RF_features': plt.colormaps['Reds'],
        'mRMR_f_mut_features': plt.colormaps['Greens'],
        'supervised_features': plt.colormaps['Purples']
    }

    num_subplots = len(included_feature_combs)
    fig, axs = plt.subplots(1, num_subplots,  figsize=(4*num_subplots, 4))
    axs = axs.flatten() if num_subplots > 1 else [axs]   

    if not notitle:
        title = title if title is not None else f'{model_name.capitalize()} confusion matrices (average across 2035-2040)' # generate title if not given
        fig.suptitle(title, fontsize = 22)  # Add the suptitle
    

    for col, feature_comb_key in enumerate(included_feature_combs):
        ax = axs[col]
        train_cm, test_cm  = calculate_cm_feature_comb(target_summaries, feature_comb_key, report_performance)
        
        data_df = pd.DataFrame(test_cm)
        labels = create_cm_labels(test_cm, train_cm)

        sns.heatmap(
            data_df, # Used for colors
            annot=labels, #used for data 
            #annot_kws={'color': 'black'}, 
            fmt='', 
            cmap=custom_cmaps[feature_comb_key], 
            ax=ax, 
            vmax=10, 
            vmin=0, 
            cbar=False
            )
    
        # Set y-axis and x-axis labels using scenario_indx_key
        ax.set_yticklabels([key for key, value in scenario_indx_key.items()], rotation=45)
        ax.set_xticklabels([key for key, value in scenario_indx_key.items()], rotation=0)
        ax.set_xlabel("Predicted scenario")
        ax.set_ylabel('True scenario')

    full_palette = ['blue', 'red', 'green', 'purple']
    labels = list(target_summaries.keys())
    handles = [plt.Line2D([0], [0], color=color, label=label) for i, (color, label) in enumerate(zip(full_palette, labels))]

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)  # Put the legend below the plot        
    plt.tight_layout()
    plt.show()

# Illustration of the classifiers?
    # SVC
    # https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py