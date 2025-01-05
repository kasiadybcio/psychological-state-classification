"""
This is a boilerplate pipeline 'eda'
generated using Kedro 0.19.10
"""
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

def plot_mood_state_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'mood'
    """
    return (data.groupby('Mood_State')
            .count()[['Time']]
            .reset_index()
            .rename(columns={'Time':'Count'}))

def plot_HRV_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'HRV'
    """
    return data[['HRV']]

def plot_GSR_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'GSR'
    """
    return data[['GSR']]

def plot_Age_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'Age'
    """
    return data[['Age']]

def plot_Heart_Rate_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'Heart_Rate'
    """
    return data[['Heart_Rate']]

def plot_Gender_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'Gender'
    """
    return (data.groupby('Gender')
            .count()[['Time']]
            .reset_index()
            .rename(columns={'Time':'Count2'}))

def plot_Task_Type_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'Task_Type'
    """
    return (data.groupby('Task_Type')
            .count()[['Time']]
            .reset_index()
            .rename(columns={'Time':'Count3'}))

def plot_Educational_Level_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'Educational_Level'
    """
    return (data.groupby('Educational_Level')
            .count()[['Time']]
            .reset_index()
            .rename(columns={'Time':'Count4'}))

def plot_Mood_State_by_Gender_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 
    """
    return data[['Mood_State','Gender']]

def plot_Age_vs_Mood_State(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 
    """
    return data[['Mood_State','Age']]

def plot_Task_Type_vs_Educational_Level(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 
    """
    return data[['Task_Type','Educational_Level']]

def plot_Skin_Temp_vs_Mood_State(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 
    """
    return data[['Mood_State','Skin_Temp']]

def plot_Heart_Rate_vs_HRV(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 
    """
    return data[['Heart_Rate','HRV']]

def plot_Time(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 
    """
    return  pd.to_datetime(data['Time']).dt.time

    