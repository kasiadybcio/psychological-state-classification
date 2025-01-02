"""
This is a boilerplate pipeline 'eda'
generated using Kedro 0.19.10
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_missing(data :pd.DataFrame):
    sns.set_style("whitegrid",{'font_scale':'1.6'})
    plt.figure(figsize=(10, 6))
    plt.title("Missing data per column %")
    sns.barplot(y = data.isna().sum()/data.shape[0],
                x = data.columns)
    plt.ylim(0,1)
    plt.xticks(rotation=90)
    return plt

def plot_dist_hrv(data :pd.DataFrame):
    sns.set_style("whitegrid",{'font_scale':'1.6'})
    plt.figure(figsize=(10, 6))
    sns.histplot(data['HRV'])
    plt.title('Distribution of Heart Rate Variability')
    plt.xlabel('HRV')
    plt.ylabel('Count')
    return plt