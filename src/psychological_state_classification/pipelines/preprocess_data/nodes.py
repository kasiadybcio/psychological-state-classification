"""
This is a boilerplate pipeline 'preprocess_data'
generated using Kedro 0.19.10
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

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

def _rename_cols(columns :list):
    """
    Removes units from column names, replaces whitespace with "_"
    """
    columns = [x.split(' (')[0].replace(" ","_") for x in columns]
    return columns

def _cast_data(data :pd.DataFrame, params: dict):
    """
    Cast the data types of the columns in the DataFrame based on the parameters_proces_data.yml
    """
    mapping_dict = {**{x:'category' for x in params['categorical']},
                        **{x:float for x in params['float']},
                        **{x:int for x in params['integer']},
                        **{x:'datetime64[s]' for x in params['datetime']}}

    cols_to_cast = params['categorical']+params['float']+params['integer']+params['datetime']
    duplicated_cols = set([x for x in cols_to_cast if cols_to_cast.count(x) > 1])

    if duplicated_cols:
        logger.critical("Column type defined more than once in parameters_proces_data.yml: %s",
                        duplicated_cols)

    data = data.astype(mapping_dict)
    missing_cols = set(data.columns) - set(mapping_dict.keys())

    if missing_cols:
        logger.warning("Column type not defined in parameters_proces_data.yml: %s", missing_cols)
    return data

def preprocess_data(data: pd.DataFrame, params: dict):
    """
    Preprocess the data by splitting the EEG Power Bands and Blood Pressure columns
    """
    # Drop ID column
    data.drop(columns=['ID'], inplace=True)

    eeg_power_bands_split = _parse_eeg_bands(data['EEG Power Bands'])
    data['EEG Power Bands D'] = eeg_power_bands_split[0]
    data['EEG Power Bands A'] = eeg_power_bands_split[1]
    data['EEG Power Bands B'] = eeg_power_bands_split[2]
    data.drop(columns=['EEG Power Bands'], inplace=True)

    blood_pressure_split = _parse_blood_pressure(data['Blood Pressure (mmHg)'])
    data['Blood Pressure Systolic'] = blood_pressure_split[0]
    data['Blood Pressure Diastolic'] = blood_pressure_split[1]
    data.drop(columns=['Blood Pressure (mmHg)'], inplace=True)

    renamed_cols = _rename_cols(data.columns)
    data.columns = renamed_cols

    data = _cast_data(data, params)

    return data
