from typing import List
import pandas as pd
import plotly
import plotly.express as px
import numpy as np
from src.utils.data_tools import time_decoder, find_final_data_bytime
from src.utils.testing import timer
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from IPython.display import HTML


@timer
def aggregate_data_by_month(years_selected: List[str], n_time: str, single_chart: bool = False):

    found_data = find_final_data_bytime(
        years_selected=years_selected, by_time=n_time)

    data = []

    for _, path in found_data:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        agg = df.resample('M').sum()

        data.append(agg)

    full_data = pd.concat(data, axis=0)

    dec = seasonal_decompose(
        full_data['fire_count'], model='additive', period=12)
    if single_chart:
        px.line(full_data, x=full_data.index, y="fire_count").show()
        dec.plot()
        plt.show()


def montage_monthly(years_selected: List[str], n_time: str):
    found_data = find_final_data_bytime(years_selected, n_time)
    data = []
    for _, path in found_data:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        agg = df.resample('M').sum()
        data.append(agg)

    full_data = pd.concat(data, axis=0)
    months_groups = full_data.groupby(full_data.index)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    def update(frame):
        ax.cla()
        ax.coastlines()
        ax.set_global()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        data = months_groups.get_group(frame)
        data = data['fire_count'].fillna(1)
        if data.empty:
            return
        ax.scatter(data['long_bin'], data['lat_bin'],
                   s=data['fire_count']*2, c='red')
        ax.set_title(f"Month {frame.strftime('%Y-%m')}")
        ax.set_xlim(data['long_bin'].min() - 1,
                    data['long_bin'].max() + 1)
        ax.set_ylim(data['lat_bin'].min() - 1,
                    data['lat_bin'].max() + 1)

    ani = FuncAnimation(
        fig, update, frames=months_groups.groups.keys(), repeat=False)
    HTML(ani.to_jshtml())


if __name__ == "__main__":
    montage_monthly(['2022', '2023', '2024'], '1d')
