from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import logging

import utils
from Dataset import Dataset

#__ Config ___________________________________________________________________________________
OPTIMIZE_MODEL : bool = False
TRAIN_MODEL : bool = True
SAVE_MODEL : bool = True
COMPUTE_PREDICTIONS : bool = True
SAVE_PREDICTIONS : bool = True

TRAIN_TEST_SPLIT : float = 0.85
SEED_NUMBER : int = 0


def load_data(route:str) -> Dataset:
    original_df : pd.DataFrame = pd.read_csv(route)
    np.random.seed(SEED_NUMBER)
    dataset = Dataset(data=original_df)
    dataset.drop_columns(['Unnamed: 0','key','pat_age_yrs','sex']).rename_target('wbit_error').split_train_test(seed_number=SEED_NUMBER, train_test_split=TRAIN_TEST_SPLIT).clean_missing(missing_threshold=0.15)
    dataset.standarize().split_x_y()
    logging.info(dataset.train.head())
    return dataset


def get_cross_validation_predictions(model:CatBoostClassifier, dataset:Dataset, n_splits:int=5) -> np.ndarray:
    kf = KFold(n_splits=n_splits, shuffle=False)
    predictions = np.zeros((dataset.train_x.shape[0]))

    for train_index, val_index in kf.split(dataset.train_x):
        X_train, X_val = dataset.train_x.iloc[train_index], dataset.train_x.iloc[val_index]
        y_train, y_val = dataset.train_y.iloc[train_index], dataset.train_y.iloc[val_index]

        model.fit(X=X_train, y=y_train, verbose=False)
        preds = model.predict(X_val)
        predictions[val_index] = preds

    return predictions


if __name__=="__main__":

    utils.start_logs()

    dataset = load_data('./data/diff.csv')
    model = CatBoostClassifier(iterations=2000, learning_rate=0.1, depth=6)

    # Compute cross-validation predictions
    val_predictions = get_cross_validation_predictions(model, dataset, n_splits=5)
    
    # Compute test predictions
    model.fit(X=dataset.train_x, y=dataset.train_y, verbose=False)
    test_predictions = model.predict(dataset.test_x)
    
    # Calculate metrics
    utils.get_metrics(y_true=dataset.train_y, y_pred=val_predictions, model_name="CatBoost validation", graph=True)
    utils.get_metrics(y_true=dataset.test_y, y_pred=test_predictions, model_name="CatBoost test", graph=True)
    
    # Save predictions
    utils.save_prediction(val_predictions, "catboost_val_predictions")
    utils.save_prediction(test_predictions, "catboost_test_predictions")
    
