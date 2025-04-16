import glob
import df_utils
import numpy as np
import logging
import pandas as pd
import os
import pickle
import utils

TRAIN_TEST_SPLIT : float = 0.85
SEED_NUMBER : int = 0

def get_predictions_paths() -> tuple[list[str]]:
    val_predictions_paths:list[str] = glob.glob("./predictions/*_val_predictions.pkl")
    test_predictions_paths:list[str] = glob.glob("./predictions/*_test_predictions.pkl")

    val_probabilities_paths:list[str] = glob.glob("./predictions/*_val_probabilities.pkl")
    test_probabilities_paths:list[str] = glob.glob("./predictions/*_test_probabilities.pkl")
    
    logging.info(val_predictions_paths)
    logging.info(test_predictions_paths)
    logging.info(val_probabilities_paths)
    logging.info(test_probabilities_paths)

    return val_predictions_paths, test_predictions_paths


def create_true_labels_file(seed_number:int,train_test_split:float) -> None:
    # if os.path.exists('./predictions/val_labels.pkl'):
    #     print('file already exists')
    #     return
    
    # extract true labels for validation
    diff : pd.DataFrame = pd.read_csv('./data/diff.csv')
    np.random.seed(seed_number)
    train_mask : np.ndarray = np.random.rand(len(diff)) < train_test_split

    train_x, train_y, test_x, test_y = df_utils.prepare_df( df_original = diff,
                                                            target_name = "wbit_error",
                                                            train_mask = train_mask,
                                                            standarize = 1,
                                                            cuad_features = False,
                                                            rate_features = False )
    

    with open(f'./predictions/val_labels.pkl', 'wb') as fid:
        logging.info(f"saving model to ./predictions/val_labels.pkl")
        pickle.dump(train_y.to_numpy(), fid)

    with open(f'./predictions/test_labels.pkl', 'wb') as fid:
        logging.info(f"saving model to ./predictions/test_labels.pkl")
        pickle.dump(test_y.to_numpy(), fid)


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

    generate_aggregated_prediction(predictions_df)

    if test:
        with open(f"./predictions/test_labels.pkl", 'rb') as fid:
            true_labels= pickle.load(fid)   
    else:
        with open(f"./predictions/val_labels.pkl", 'rb') as fid:
            true_labels= pickle.load(fid)
    

    predictions_df['true_labels'] = true_labels
    return predictions_df


def generate_aggregated_prediction(predictions:pd.DataFrame) -> np.ndarray:
    n_cols:int = predictions.shape[1]
    predictions['mayority'] = (predictions.sum(axis=1)/n_cols).round(0).astype(int)
    


if __name__=="__main__":

    logging.basicConfig(
    filename='bloodwork.log',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    )

    val_predictions_paths,test_predictions_paths = get_predictions_paths()

    create_true_labels_file(seed_number=SEED_NUMBER, train_test_split=TRAIN_TEST_SPLIT)

    val_predictions:pd.DataFrame = load_predictions(val_predictions_paths)
    test_predictions:pd.DataFrame = load_predictions(test_predictions_paths,test=True)


    utils.get_metrics(y_true=val_predictions['true_labels'].to_numpy(),
                      y_pred=val_predictions['mayority'].to_numpy(),
                      model_name="assemble validation",
                      graph=True)

    utils.get_metrics(y_true=test_predictions['true_labels'].to_numpy(),
                      y_pred=test_predictions['mayority'].to_numpy(),
                      model_name="assemble test",
                      graph=True)


    print(val_predictions.head(10))
    print(test_predictions.head(10))

    val_predictions.to_csv('./predictions/val_predictions.csv')
    test_predictions.to_csv('./predictions/test_predictions.csv')

