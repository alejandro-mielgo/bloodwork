import glob
import numpy as np
import logging
import pandas as pd
import pickle

from Dataset import Dataset
import utils

TRAIN_TEST_SPLIT : float = 0.85
SEED_NUMBER : int = 0

def get_predictions_paths(get_ensemble:bool=False) -> tuple[list[str]]:
    val_predictions_paths:list= glob.glob("./predictions/individual_predictions/*_val_predictions.pkl")
    test_predictions_paths:list= glob.glob("./predictions/individual_predictions/*_test_predictions.pkl")

    val_probabilities_paths:list = glob.glob("./predictions/*_val_probabilities.pkl")
    test_probabilities_paths:list = glob.glob("./predictions/*_test_probabilities.pkl")
    
    if get_ensemble:
        val_predictions_paths.append("./predictions/individual_predictions\\majority_vote_val_pred.pkl")
        test_predictions_paths.append("./predictions/individual_predictions\\majority_vote_test_pred.pkl")

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
    prediction_columns:list[str] = ['kn','catboost','nn']
    majority_vote_val = round(df_val[prediction_columns].sum(axis=1) / df_val[prediction_columns].shape[1]).to_numpy().astype(int)
    majority_vote_test = round(df_test[prediction_columns].sum(axis=1) / df_test[prediction_columns].shape[1]).to_numpy().astype(int)
    utils.save_prediction(majority_vote_val, "majority_vote_val_pred")
    utils.save_prediction(majority_vote_test, "majority_vote_test_pred")

    return majority_vote_val, majority_vote_test
    

if __name__=="__main__":

    utils.start_logs()
    
    val_predictions_paths,test_predictions_paths = get_predictions_paths()
    dataset:Dataset = load_data(seed_number=SEED_NUMBER, train_test_split=TRAIN_TEST_SPLIT)
    
    # Dataframes que solo contienen las predicciones individuales + targets
    train_predictions:pd.DataFrame = load_predictions(val_predictions_paths)
    test_predictions:pd.DataFrame = load_predictions(test_predictions_paths,test=True)
    train_predictions['target'] = dataset.train_y.to_numpy()
    test_predictions['target'] = dataset.test_y.to_numpy()
  
    # Ensemble majority vote
    majority_vote_val, majority_vote_test = get_mayority_predictions(df_val=train_predictions, df_test=test_predictions)
    utils.get_metrics(y_true=dataset.train_y, y_pred=majority_vote_val, model_name="majority vote validation")
    utils.get_metrics(y_true=dataset.test_y, y_pred=majority_vote_test, model_name="majority vote test")
    aux_train_predictions = train_predictions.copy()
    aux_test_predictions = test_predictions.copy()
    aux_train_predictions.insert(0, 'majority_vote', majority_vote_val)
    aux_test_predictions.insert(0, 'majority_vote', majority_vote_test)
    
    # Guardar los resultados de las predicciones individuales, incluyendo el mayority vote
    aux_train_predictions.to_csv('./predictions/train_predictions.csv')
    aux_test_predictions.to_csv('./predictions/test_predictions.csv')



    # Dataframes que contienen las features (limpias) + predicciones + targets
    # Para hacer una red neuronal que use de entradas las predicciones y las features
    train:pd.DataFrame = merge_predicitions_and_features(feat_df=dataset.train_x, predictions_df=train_predictions) 
    test:pd.DataFrame = merge_predicitions_and_features(feat_df=dataset.test_x, predictions_df=test_predictions)
    train.to_csv('./predictions/train.csv')
    test.to_csv('./predictions/test.csv')