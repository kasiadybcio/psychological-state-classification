"""
This is a boilerplate pipeline 'eda'
generated using Kedro 0.19.10
"""
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def plot_mood_state_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'mood'
    """
    return (data.groupby('Mood_State', observed=False)
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
    return (data.groupby('Gender', observed=False)
            .count()[['Time']]
            .reset_index()
            .rename(columns={'Time':'Count2'}))

def plot_Task_Type_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'Task_Type'
    """
    return (data.groupby('Task_Type', observed=False)
            .count()[['Time']]
            .reset_index()
            .rename(columns={'Time':'Count3'}))

def plot_Educational_Level_distribution(data: pd.DataFrame):
    """
    Plot the distribution of the target variable 'Educational_Level'
    """
    return (data.groupby('Educational_Level', observed=False)
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

def _conf_matrix(y_preds: pd.DataFrame, model_name) -> dict:
    y_preds=y_preds.query(f"model=='{model_name}'")
    true_label = y_preds['true_label']
    predicted_label = y_preds[[x for x in y_preds.columns if 'proba' in x]].idxmax(axis=1).str.replace('_proba', '')
    return pd.DataFrame(zip(predicted_label,true_label),columns=['Predicted label','True label'])

def plot_confusion_matrix_xgb(y_preds: pd.DataFrame):
    """
    Plot the confusion matrix of the target variable 
    """
    return _conf_matrix(y_preds, 'xgb')

def plot_confusion_matrix_lgbm(y_preds: pd.DataFrame):
    """
    Plot the confusion matrix of the target variable 
    """
    return _conf_matrix(y_preds, 'lgbm')

def plot_confusion_matrix_lr(y_preds: pd.DataFrame):
    """
    Plot the confusion matrix of the target variable 
    """
    return _conf_matrix(y_preds, 'lr')

def _plot_roc_auc(y_preds: pd.DataFrame, model_name):
    y_preds=y_preds.query(f"model=='{model_name}'")
    labels = y_preds['true_label'].unique()
    roc_data = []
    for label in labels:
        # Binarize true labels for the current class
        true_labels = (y_preds["true_label"] == label).astype(int)
        
        # Get probabilities for the current class
        proba = y_preds[f"{label}_proba"]

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(true_labels, proba)
        roc_auc = auc(fpr, tpr)

        # Append data for plotting
        roc_data.extend([
            {"FPR": f, "TPR": t, "Model": model_name.upper(), "Class": label, "AUC": roc_auc} for f, t in zip(fpr, tpr)
        ])
    return pd.DataFrame(roc_data)

def plot_roc_auc_xgb(y_preds: pd.DataFrame):
    """
    Plot the ROC-AUC curve of the target variable 
    """
    return _plot_roc_auc(y_preds, 'xgb')

def plot_roc_auc_lgbm(y_preds: pd.DataFrame):
    """
    Plot the ROC-AUC curve of the target variable 
    """
    return _plot_roc_auc(y_preds, 'lgbm')

def plot_roc_auc_lr(y_preds: pd.DataFrame):
    """
    Plot the ROC-AUC curve of the target variable 
    """
    return _plot_roc_auc(y_preds, 'lr')