"""
This is a boilerplate pipeline 'preprocess_data'
generated using Kedro 0.19.10
"""
import pandas as pd

def _parse_eeg_bands(x :pd.Series):
    """
    Parse the string representation of a list of EEG bands (Delta, ALPHA and Beta bands)
    into a list of floats.
    """
    x = x.str.replace('[','').str.replace(']','')
    x = x.str.split(',', expand=True).astype(float)
    return x

def _parse_blood_pressure(x :pd.Series):
    """
    Parse the string representation of a blood pressure (Systolic/Diastolic) 
    into a list of floats.
    """
    x = x.str.split('/', expand=True)
    print(type(x))
    return x

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by splitting the EEG Power Bands and Blood Pressure columns
    """
    eeg_power_bands_split = _parse_eeg_bands(data['EEG Power Bands'])
    data['EEG Power Bands D'] = eeg_power_bands_split[0]
    data['EEG Power Bands A'] = eeg_power_bands_split[1]
    data['EEG Power Bands B'] = eeg_power_bands_split[2]
    data.drop(columns=['EEG Power Bands'], inplace=True)

    blood_pressure_split = _parse_blood_pressure(data['Blood Pressure (mmHg)'])
    data['Blood Pressure Systolic'] = blood_pressure_split[0]
    data['Blood Pressure Diastolic'] = blood_pressure_split[1]
    data.drop(columns=['Blood Pressure (mmHg)'], inplace=True)

    return data
