import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # aviso al crear muchas columnas en pandas


def df_health( dataframe : pd.DataFrame, verbose : bool = False, graph : bool = False ) -> dict:
    
    missing = {}

    n_cols : int = dataframe.shape[1]
    n_rows : int = dataframe.shape[0]

    for column in dataframe.columns:
        missing[column] = dataframe[column].isna().sum()
        if verbose:
            print(f'{column}\t {missing[column]}')

    missing_rate = sum(missing.values()) / (dataframe.shape[0] * dataframe.shape[1])

    logging.info(f'N_samples: {n_rows}')
    logging.info(f'N_features: {n_cols}')
    # logging.info(f'N_missing: {sum(missing.values())}')
    logging.info(f'Missing data ratio: {round(missing_rate,4)}')

    if graph:
        fig = plt.figure(figsize=(14, 5))
        plt.bar(missing.keys(), missing.values())
        plt.xticks(rotation=90)
        plt.show()

    return missing


def clean_missing(df_original : pd.DataFrame,
                  cols_to_drop : list[str] = None,
                  missing_threshold : float = 0.1,
                  dropRows : bool = True ) -> pd.DataFrame:

    df = df_original.copy()

    n_rows = df.shape[0]

    if cols_to_drop is not None and cols_to_drop[0] in df.columns:
        df.drop(columns=cols_to_drop, inplace=True)

    for column in df.columns:
        if df[column].isna().sum() / n_rows > missing_threshold:
            df.drop(column, axis=1, inplace=True)

    if dropRows:
        df.dropna(inplace=True)

    if df.isna().sum().sum()!=0:
        logging.warning("There is still missing values in dataframe")

    return df


def rename_target(df_original : pd.DataFrame, 
                  target_name : str) -> pd.DataFrame:
    
    logging.info(f"Renaming target data from {target_name} to 'target'")

    if target_name not in df_original.columns:
        raise ValueError('target column is not in dataframe')
    
    df = df_original.copy()
    df.rename(columns={target_name:'target'}, inplace=True)
    return df


def remove_outliers(df_original:pd.DataFrame, 
                    threshold_n_sigmas:float) -> pd.DataFrame:
    df = df_original.copy()

    for column in df.columns:
        if df[column].dtype == 'float64':
            mean = df[column].mean()
            std = df[column].std()
            df = df[np.abs(df[column] - mean) <= threshold_n_sigmas * std]
    return df


def normalize_df(df_original : pd.DataFrame, 
                 filter_col : str = "target", 
                 filter_value : any = None) -> pd.DataFrame:

    logging.info(f"Normalizing data, filter_col={filter_col}, filter_value={filter_value}")

    df = df_original.copy()
    for column in df.columns:
        if df[column].dtype == 'float64':
            if filter_value is not None:
                mean = df.loc[df[filter_col] == filter_value,column].mean()
                std = df.loc[df[filter_col] == filter_value,column].std()
            else:
                mean = df[column].mean()
                std = df[column].std()

            df[column] = (df[column]-mean)/std
    return df


def one_hot(df_original: pd.DataFrame):

    df = df_original.copy()

    for column in df.columns:
        if df[column].dtype == 'object':
            df = pd.get_dummies(df, columns=[column], drop_first=True, dtype=int)
    return df


def create_cuadratic_features(df_original:pd.DataFrame, normalize:bool=True) -> pd.DataFrame:
    df = df_original.copy()
    columns = df.select_dtypes(include=["number"]).columns.tolist()

    if 'target' in df.columns.tolist():
        columns.remove('target')

    n = len(columns)

    for i in range(n):
        for j in range(i,n):
            df[f"{i}_{j}"] = df[columns[i]] * df[columns[j]]

    if normalize:
        df = normalize_df(df_original=df)

    return df.copy()


def create_rate_features(df_original:pd.DataFrame, 
                         time_feature:str, 
                         cols_to_ignore:list[str]) -> pd.DataFrame:
    
    logging.info(f"creating rate feature, time feature:{time_feature}")
    
    df = df_original.copy()

    columns = df.select_dtypes(include=["number"]).columns.tolist()
    if 'target' in df.columns.tolist():
        columns.remove('target')

    for column in columns:
        if column != time_feature and column not in cols_to_ignore:
            df[f"{column}_rate"] = df[column] / df[time_feature]

    return df
    
    
def x_y_split(dataframe: pd.DataFrame, 
              y_label : str = "target",
              shuffle:bool=False) -> tuple[pd.DataFrame]:
    """
    returs X_data, y_data
    """
    if shuffle:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    x = dataframe.drop(columns=[y_label])
    y = dataframe[y_label]

    return x,y


def prepare_df(df_original : pd.DataFrame,
               train_mask : np.ndarray,
               normalize : int = 0,
               cuad_features : bool = False,
               rate_features : bool = False,
                ) -> tuple[pd.DataFrame]:
    """
    returns train and test data after being cleaned
    """

    df = df_original.copy()
    logging.info(f"Shape before processing: {df.shape} ")

    df = rename_target(df_original=df, target_name='wbit_error')

    df = clean_missing(df_original=df,
                       missing_threshold = 0.10,
                       cols_to_drop=['Unnamed: 0','key'],
                       dropRows=True)


    df = one_hot(df_original=df)

    if cuad_features:
        df = create_cuadratic_features(df_original=df, normalize=False)

    if rate_features:
        df = create_rate_features(df_original=df, time_feature='hrs_between_cbcs', cols_to_ignore=['sex_M'])

    if normalize==1:
        df = normalize_df(df_original=df) 
    if normalize==2:
        df = normalize_df(df_original=df, filter_col='target',filter_value=0)
    
    logging.info(f"Shape after processing: {df.shape} ")

    dummy_col :str = df_original.columns[0]

    df_train = df.join(df_original.loc[train_mask, dummy_col], how="inner", rsuffix="_to_drop")
    df_test = df.join(df_original.loc[~train_mask, dummy_col], how="inner", rsuffix="_to_drop")

    df_train = df_train.drop(columns=[dummy_col])
    df_test = df_test.drop(columns=[dummy_col])

    logging.info(f"Shape of train data: {df_train.shape} ")
    logging.info(f"Shape of test data: {df_test.shape} ")
    
    train_x, train_y = x_y_split(df_train)
    test_x, test_y = x_y_split(df_test)

    return train_x, train_y, test_x, test_y
