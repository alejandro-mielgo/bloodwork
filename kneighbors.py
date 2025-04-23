from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import logging
import time

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



def optimize_kn(x_train : pd.DataFrame,
                y_train: pd.DataFrame,
                n_neigbours : list[int] = np.linspace(2,11,10).astype(int),
                scoring = "precision",
                n_points: int = 10000) -> dict:
    
    if n_points > x_train.shape[0]:
        n_points = x_train.shape[0]

    parameters : dict = {"n_neighbors":n_neigbours, 'p': [1,2], 'weights':['uniform','distance']}

    kn_classifier = KNeighborsClassifier()
    grid_search = GridSearchCV(kn_classifier, param_grid=parameters, cv=5, scoring=scoring )
    grid_search.fit(x_train.iloc[:n_points,:], y_train[:n_points])

    best_score = grid_search.best_score_

    logging.info(f"Best score : {best_score:.3f}")
    logging.info(f"Best parameters:")
    logging.info(grid_search.best_params_)

    return grid_search.best_params_


def execute_kn(optimize_model : bool,
               train_model : bool,          #Train model, if False, load from file
               save_model : bool,
               compute_predictions : bool,  #Calculate predictions, if False, load from file
               save_predictions : bool,
               train_test_split : float,
               seed_number : int) -> None:
    


    #__ Load Data ________________________________________________________________________________
    original_df : pd.DataFrame = pd.read_csv('./data/diff.csv')
    np.random.seed(seed_number)

    dataset = Dataset(data=original_df)
    dataset.drop_columns(['Unnamed: 0','key','pat_age_yrs','sex']).rename_target('wbit_error').split_train_test(seed_number=seed_number, train_test_split=train_test_split).clean_missing(missing_threshold=0.15)
    dataset.standarize().split_x_y()
    logging.info(dataset.train.head())

    #__ Find best parameters ______________________________________________________________________
    if optimize_model:
        parameters:dict = optimize_kn(x_train = dataset.train_x,
                                      y_train = dataset.train_y,
                                      n_neigbours = [3,4,5],
                                      scoring = "f1",
                                      n_points = 2000)
    else:
        parameters:dict = { "n_neighbors":4, 'p':1, 'weights':'distance' }
    
    #__ Train model _______________________________________________________________________________
    if train_model:
        logging.info("start k neighbors classifier training")
        kn_classifier = KNeighborsClassifier(n_neighbors=parameters["n_neighbors"],p=parameters['p'],weights=parameters['weights'])
        kn_classifier.fit(X=dataset.train_x, y=dataset.train_y)
    else:
        kn_classifier = utils.load_file('./models/kn_classifier.pkl')

    if save_model:
        utils.save_model(model=kn_classifier, model_name='kn_classifier')

    #__ Get_predictions ___________________________________________________________________________
    kn_val_predictions, kn_test_predictions, kn_val_probabilities, kn_test_probabilities = utils.manage_predictions(
        model = kn_classifier,
        model_name = "kn",
        x_train = dataset.train_x,
        y_train = dataset.train_y,
        x_test = dataset.test_x,
        y_test = dataset.test_y,
        compute_predictions = compute_predictions,
        save_predictions = save_predictions)


    utils.get_metrics(y_true = dataset.train_y, y_pred = kn_val_predictions, model_name="k-neighbors validation")
    utils.get_metrics(y_true = dataset.test_y,  y_pred = kn_test_predictions, model_name="k-neighbors test")

if __name__ == "__main__":
    utils.start_logs()

    logging.info("K Neighbors classifier")

    execute_kn(optimize_model=OPTIMIZE_MODEL,
               train_model=TRAIN_MODEL,
               save_model=SAVE_MODEL,
               compute_predictions=COMPUTE_PREDICTIONS,
               save_predictions=SAVE_PREDICTIONS,
               train_test_split=TRAIN_TEST_SPLIT,
               seed_number=SEED_NUMBER)