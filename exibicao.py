from doctest import testfile
from turtle import width
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
import plotly.express as px
import plotly.graph_objects as go
from plotly import offline
import tkinter as tk
from tkinter import filedialog as fd
from datetime import datetime, timedelta
from scipy.stats import iqr
from PIL import Image
from calibragem import obter_resolucao_tela
from skimage import io

def freedman_diaconis_rule(data_x,data_y):

    bin_width_x = 2.0 * iqr(data_x) / np.power(len(data_x), 1.0/3.0)
    bin_count_x = int(np.ceil((np.max(data_x) - np.min(data_x)) / bin_width_x))
    # print(f'bin_width_x = {bin_width_x}\tbin_count_x = {bin_count_x}')

    bin_width_y = 2.0 * iqr(data_y) / np.power(len(data_y), 1.0/3.0)
    bin_count_y = int(np.ceil((np.max(data_y) - np.min(data_y)) / bin_width_y))
    # print(f'bin_width_y = {bin_width_y}\tbin_count_y = {bin_count_y}')

    return bin_count_x,bin_count_y

def exibir(coords):

    data_x, data_y = [v[1] for v in coords],[v[2] for v in coords]
    
    bin_count_x,bin_count_y = freedman_diaconis_rule(data_x,data_y)

    width, height = obter_resolucao_tela()

    fig = px.density_heatmap(coords, x=data_x, y=data_y, nbinsx=64, nbinsy=36, range_x=[0,width], range_y=[0,height])

    fig.update_layout(
        width=width,
        height=height,
        images=[
            dict(
                source=Image.open("./assets/planeta.png"),
                xref="x", 
                yref="y",
                x=0,
                y=height,
                sizex=width, 
                sizey=height,
                xanchor="left",
                yanchor="top",
                sizing="stretch",
                layer="above",
                opacity=0.5
            )    
        ]
    )
    
    fig.show()
   # offline.plot(fig)