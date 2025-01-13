"""
This is a boilerplate pipeline 'split_data'
generated using Kedro 0.19.10
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(2025)

def split_data(data: pd.DataFrame, parameters: dict):
    """
    Split the data into training and test sets.
    It uses the parameters defined in conf/base/parameters_split_data.yml
    """
    df_train, df_test = train_test_split(data,
                                        test_size=parameters["test_size"],
                                        random_state=parameters["random_state"],
                                        stratify=data[parameters["target_col"]],
                                        shuffle=parameters["shuffle"]
                                        )
    return df_train, df_test
