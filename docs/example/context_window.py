from shiny import render
from shiny.express import input, ui
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from scipy.interpolate import interp1d
import numpy as np

ui.page_opts(fillable=True)
ui.panel_title("Does context size matter for viral dna models?")

with ui.card():
    ui.input_selectize(  
    "plot_type",  
    "Select your model size:",
    ["14M", "31M", "70M", "160M", "410M"],
    multiple=False,  
)

def plot_loss_rates(df, type): 
    # interplot each column to be same number of points
    x = np.linspace(0, 1, 1000)
    loss_rates = []
    labels = ['32', '64', '128', '256', '512', '1024']
    #drop the column step
    df = df.drop(columns=['Step'])
    for col in df.columns:
        # drop row values that are equal to Training Loss
        
        y = df[col].dropna().astype('float').values
        f = interp1d(np.linspace(0, 1, len(y)), y)
        loss_rates.append(f(x))
    fig, ax = plt.subplots()
    for i, loss_rate in enumerate(loss_rates):
        ax.plot(x, loss_rate, label=labels[i])
    ax.legend()
    ax.set_title(f'Loss rates for {type} model')
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Loss rate')
    return fig

 
    
    
    
@render.plot
def plot():
    fig = None
    if input.plot_type() == "14M":
        df = pd.read_csv('14m.csv')
        fig = plot_loss_rates(df, '14M')
    return fig
        
