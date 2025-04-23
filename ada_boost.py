from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
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


def optimize_ada_boost( x_train : pd.DataFrame,
                                y_train: pd.DataFrame,
                                n_estimators:list[int] = [100,200],
                                learning_rate:list[float] = [1],
                                scoring = "precision",
                                n_points: int = 10000
                                ) -> dict:
    
    if n_points > x_train.shape[0]:
        n_points = x_train.shape[0]

    parameters : dict = {"n_estimators":n_estimators,"learning_rate":learning_rate}
    

    ada_boost_classifier = AdaBoostClassifier(random_state=SEED_NUMBER)
    logging.info(f"Optimizing {ada_boost_classifier}, with grid search {parameters}")
    grid_search = GridSearchCV(ada_boost_classifier, param_grid=parameters, cv=5, scoring=scoring, verbose=3 )
    grid_search.fit(x_train.iloc[:n_points,:], y_train[:n_points])

    logging.info(f"Best {scoring} : {grid_search.best_score_:.3f}")
    logging.info(f"Best parameters:")
    logging.info(grid_search.best_params_)

    return grid_search.best_params_


def execute_ada_boost( optimize_model : bool,
                            train_model : bool,          #Train model, if False, load from file
                            save_model : bool,
                            compute_predictions : bool,  #Calculate predictions, if False, load from file
                            save_predictions : bool,
                            train_test_split : float,
                            seed_number : int
                            ) -> None:

    #__ Load Data ________________________________________________________________________________
    original_df : pd.DataFrame = pd.read_csv('./data/diff.csv')
    np.random.seed(seed_number)

    dataset = Dataset(data=original_df)
    dataset.drop_columns(['Unnamed: 0','key','pat_age_yrs','sex']).rename_target('wbit_error').split_train_test(seed_number=seed_number, train_test_split=train_test_split).clean_missing(missing_threshold=0.15)
    dataset.standarize().split_x_y()
    logging.info(dataset.train.head())
    
    #__ Find best parameters ______________________________________________________________________
    if optimize_model:
        parameters:dict = optimize_ada_boost(   x_train = dataset.train_x,
                                                y_train = dataset.train_y,
                                                n_estimators= [150,250],
                                                learning_rate= [1],
                                                scoring = "f1",
                                                n_points = 15000 )
    else:
        parameters:dict = {"n_estimators":250,"learning_rate":1}
    
    
    #__ Train model _______________________________________________________________________________
    if train_model:
        
        ada_boost_classifier = AdaBoostClassifier(n_estimators=parameters['n_estimators'],
                                                  learning_rate=parameters['learning_rate'],
                                                  random_state=SEED_NUMBER)
        logging.info("start ada_boost classifier training")
        ada_boost_classifier.fit(X=dataset.train_x, y=dataset.train_y)
    else:
        ada_boost_classifier = utils.load_file(ada_boost_classifier)

    if save_model:
        utils.save_model(model=ada_boost_classifier, model_name='ada_boost_classifier')

    #__ Get_predictions ___________________________________________________________________________
    ada_boost_val_predictions, ada_boost_test_predictions, ada_boost_val_probabilities, ada_boost_test_probabilities = utils.manage_predictions(
        model = ada_boost_classifier,
        model_name = "ada_boost",
        x_train = dataset.train_x,
        y_train = dataset.train_y,
        x_test = dataset.test_x,
        y_test = dataset.test_y,
        compute_predictions = compute_predictions,
        save_predictions = save_predictions)

    utils.get_metrics(y_true = dataset.train_y, y_pred = ada_boost_val_predictions, model_name="ada-boost validation")
    utils.get_metrics(y_true = dataset.test_y,  y_pred = ada_boost_test_predictions, model_name="ada-boost test")


if __name__ == '__main__':

    utils.start_logs()

    execute_ada_boost(  optimize_model=OPTIMIZE_MODEL,
                        train_model=TRAIN_MODEL,
                        save_model=SAVE_MODEL,
                        compute_predictions=COMPUTE_PREDICTIONS,
                        save_predictions=SAVE_PREDICTIONS,
                        train_test_split=TRAIN_TEST_SPLIT,
                        seed_number=SEED_NUMBER)