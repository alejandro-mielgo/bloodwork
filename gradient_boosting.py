from sklearn.ensemble import GradientBoostingClassifier
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


def optimize_gradient_boosting( x_train : pd.DataFrame,
                                y_train: pd.DataFrame,
                                learning_rate:list[float],
                                n_estimators:list[int],
                                min_samples_split:list[int],
                                min_samples_leaf:list[int],
                                scoring = "precision",
                                n_points: int = 10000
                                ) -> dict:
    
    if n_points > x_train.shape[0]:
        n_points = x_train.shape[0]

    parameters : dict = {"learning_rate":learning_rate,
                         "n_estimators":n_estimators,
                         "min_samples_split":min_samples_split,
                         "min_samples_leaf":min_samples_leaf}

    gradient_boosting_classifier = GradientBoostingClassifier(random_state=SEED_NUMBER)
    grid_search = GridSearchCV(gradient_boosting_classifier, param_grid=parameters, cv=5, scoring=scoring,verbose=3 )
    grid_search.fit(x_train.iloc[:n_points,:], y_train[:n_points])

    logging.info(f"Best {scoring} : {grid_search.best_score_:.3f}")
    logging.info(f"Best parameters:")
    logging.info(grid_search.best_params_)

    return grid_search.best_params_


def execute_gradient_boost( optimize_model : bool,
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
        parameters:dict = optimize_gradient_boosting( x_train = dataset.train_x,
                                                      y_train = dataset.train_y,
                                                      learning_rate=[0.1],
                                                      n_estimators=[50],
                                                      min_samples_split=[2,3],
                                                      min_samples_leaf=[1,2],
                                                      scoring = "f1",
                                                      n_points = 10000)
    else:
        parameters : dict = {"learning_rate":0.1,
                             "n_estimators":200,
                             "min_samples_split":2,
                             "min_samples_leaf":2}
    
    #__ Train model _______________________________________________________________________________
    if train_model:
        logging.info("start gradient_boosting classifier training")
        gradient_boosting_classifier = GradientBoostingClassifier(random_state=SEED_NUMBER,
                                                                  learning_rate=parameters['learning_rate'],
                                                                  n_estimators=parameters['n_estimators'],
                                                                  min_samples_split=parameters['min_samples_split'],
                                                                  min_samples_leaf=parameters['min_samples_leaf'])
        gradient_boosting_classifier.fit(X=dataset.train_x, y=dataset.train_y)
    else:
        gradient_boosting_classifier = utils.load_file('./models/gradient_boosting_classifier.pkl')

    if save_model:
        utils.save_model(model=gradient_boosting_classifier, model_name='gradient_boosting_classifier')

    #__ Get_predictions ___________________________________________________________________________
    gradient_boosting_val_predictions, gradient_boosting_test_predictions, gradient_boosting_val_probabilities, gradient_boosting_test_probabilities = utils.manage_predictions(
        model = gradient_boosting_classifier,
        model_name = "gradient_boosting",
        x_train = dataset.train_x,
        y_train = dataset.train_y,
        x_test = dataset.test_x,
        y_test = dataset.test_y,
        compute_predictions = compute_predictions,
        save_predictions = save_predictions)

    utils.get_metrics(y_true = dataset.train_y, y_pred = gradient_boosting_val_predictions, model_name="gradient-boosting validation")
    utils.get_metrics(y_true = dataset.test_y,  y_pred = gradient_boosting_test_predictions, model_name="gradient-boosting test")


if __name__ == '__main__':

    utils.start_logs()

    execute_gradient_boost( optimize_model=OPTIMIZE_MODEL,
                            train_model=TRAIN_MODEL,
                            save_model=SAVE_MODEL,
                            compute_predictions=COMPUTE_PREDICTIONS,
                            save_predictions=SAVE_PREDICTIONS,
                            train_test_split=TRAIN_TEST_SPLIT,
                            seed_number=SEED_NUMBER )