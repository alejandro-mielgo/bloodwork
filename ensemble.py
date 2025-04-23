import glob
import df_utils
import numpy as np
import logging
import pandas as pd
import pickle

from Dataset import Dataset
import utils

TRAIN_TEST_SPLIT : float = 0.85
SEED_NUMBER : int = 0

def get_predictions_paths() -> tuple[list[str]]:
    val_predictions_paths:list[str] = glob.glob("./predictions/individual_predictions/*_val_predictions.pkl")
    test_predictions_paths:list[str] = glob.glob("./predictions/individual_predictions/*_test_predictions.pkl")

    val_probabilities_paths:list[str] = glob.glob("./predictions/*_val_probabilities.pkl")
    test_probabilities_paths:list[str] = glob.glob("./predictions/*_test_probabilities.pkl")
    
    logging.info(val_predictions_paths)
    logging.info(test_predictions_paths)
    logging.info(val_probabilities_paths)
    logging.info(test_probabilities_paths)

    return val_predictions_paths, test_predictions_paths


def load_data(seed_number:int,train_test_split:float) -> Dataset:

    diff:pd.DataFrame = pd.read_csv('./data/diff.csv')
    np.random.seed(seed_number)

    dataset:Dataset = Dataset(data=diff)
    dataset.drop_columns(['Unnamed: 0','key','pat_age_yrs','sex'])
    dataset.rename_target('wbit_error')
    dataset.split_train_test(seed_number=seed_number, train_test_split=train_test_split)
    dataset.clean_missing(missing_threshold=0.15)
    dataset.standarize()
    dataset.split_x_y()

    with open(f'./predictions/val_labels.pkl', 'wb') as fid:
        logging.info(f"saving model to ./predictions/val_labels.pkl")
        pickle.dump(dataset.train_y.to_numpy(), fid)

    with open(f'./predictions/test_labels.pkl', 'wb') as fid:
        logging.info(f"saving model to ./predictions/test_labels.pkl")
        pickle.dump(dataset.test_y.to_numpy(), fid)
    
    return dataset


def load_predictions(path_list:list[str],test:bool=False) -> pd.DataFrame:

    predictions_dict:dict = {}

    for file_path in path_list:
        if test:
            model_name:str = file_path.split('test')[0].split('\\')[1][:-1]
        else:
            model_name:str = file_path.split('val')[0].split('\\')[1][:-1]
        with open(f"{file_path}", 'rb') as fid:
            predictions = pickle.load(fid)
        
        predictions_dict[model_name] = predictions

    
    predictions_df:pd.DataFrame = pd.DataFrame.from_dict(predictions_dict)

    return predictions_df


def merge_predicitions_and_features(feat_df:pd.DataFrame,predictions_df:pd.DataFrame) -> pd.DataFrame:
    feat_df = feat_df.reset_index(drop=True)
    merged_df:pd.DataFrame = pd.concat([feat_df,predictions_df],axis=1)
    return merged_df


def get_mayority_predictions(df_val:pd.DataFrame, df_test:pd.DataFrame) -> None:
    prediction_columns:list[str] = ['ada_boost','gradient_boosting','kn','lr','svc']
    majority_vote_val = round(df_val[prediction_columns].sum(axis=1) / df_val[prediction_columns].shape[1])
    majority_vote_test = round(df_test[prediction_columns].sum(axis=1) / df_test[prediction_columns].shape[1])

    print(majority_vote_val[0:10])

    with open(f'./predictions/majority_val_predictions.pkl', 'wb') as fid:
        logging.info(f"saving model to ./predictions/majority_val_predictions.pkl")
        pickle.dump(majority_vote_val, fid)
    with open(f'./predictions/majority_test_predictions.pkl', 'wb') as fid:
        logging.info(f"saving model to ./predictions/majority_test_predictions.pkl")
        pickle.dump(majority_vote_test, fid)
    
    return majority_vote_val, majority_vote_test
    

if __name__=="__main__":

    utils.start_logs()

    val_predictions_paths,test_predictions_paths = get_predictions_paths()

    dataset:Dataset = load_data(seed_number=SEED_NUMBER, train_test_split=TRAIN_TEST_SPLIT)
    
    train_predictions:pd.DataFrame = load_predictions(val_predictions_paths)
    test_predictions:pd.DataFrame = load_predictions(test_predictions_paths,test=True)
    train_predictions['target'] = dataset.train_y.to_numpy()
    test_predictions['target'] = dataset.test_y.to_numpy()
    train_predictions.to_csv('./predictions/train_predictions.csv')
    test_predictions.to_csv('./predictions/test_predictions.csv')

    train:pd.DataFrame = merge_predicitions_and_features(feat_df=dataset.train_x, predictions_df=train_predictions) 
    test:pd.DataFrame = merge_predicitions_and_features(feat_df=dataset.test_x, predictions_df=test_predictions)

    
    train.to_csv('./predictions/train.csv')
    test.to_csv('./predictions/test.csv')