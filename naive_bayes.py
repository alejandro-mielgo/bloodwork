from sklearn.naive_bayes import GaussianNB
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


def optimize_naive_bayes(   x_train : pd.DataFrame,
                            y_train: pd.DataFrame,
                            var_smoothing:list[float] = [1e-9],
                            scoring = "precision",
                            n_points: int = 10000
                            ) -> dict:
    
    if n_points > x_train.shape[0]:
        n_points = x_train.shape[0]

    parameters : dict = {"var_smoothing":var_smoothing,
                        }

    naive_bayes_classifier = GaussianNB(var_smoothing=parameters['var_smoothing'])
    grid_search = GridSearchCV(naive_bayes_classifier, param_grid=parameters, cv=5, scoring=scoring,verbose=3 )
    grid_search.fit(x_train.iloc[:n_points,:], y_train[:n_points])

    logging.info(f"Best {scoring} : {grid_search.best_score_:.3f}")
    logging.info(f"Best parameters:")
    logging.info(grid_search.best_params_)

    return grid_search.best_params_


def execute_naive_bayes(optimize_model : bool,
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
        parameters:dict = optimize_naive_bayes( x_train = dataset.train_x,
                                                y_train = dataset.train_y,
                                                var_smoothing = [1e-9],
                                                scoring = "precision",
                                                n_points = 5000 )
    else:
        parameters : dict = {"var_smoothing":1e-9}
    
    #__ Train model _______________________________________________________________________________
    if train_model:
        logging.info("start gradient_boosting classifier training")
        naive_bayes_classifier = GaussianNB(var_smoothing=parameters['var_smoothing'])
        naive_bayes_classifier.fit(X=dataset.train_x, y=dataset.train_y)
    else:
        naive_bayes_classifier = utils.load_file('./models/naive_bayes_classifier.pkl')

    if save_model:
        utils.save_model(model=naive_bayes_classifier, model_name='naive_bayes_classifier')

    #__ Get_predictions ___________________________________________________________________________
    gradient_boosting_val_predictions, gradient_boosting_test_predictions, gradient_boosting_val_probabilities, gradient_boosting_test_probabilities = utils.manage_predictions(
        model = naive_bayes_classifier,
        model_name = "naive_bayes",
        x_train = dataset.train_x,
        y_train = dataset.train_y,
        x_test = dataset.test_x,
        y_test = dataset.test_y,
        compute_predictions = compute_predictions,
        save_predictions = save_predictions)

    utils.get_metrics(y_true = dataset.train_y, y_pred = gradient_boosting_val_predictions, model_name="naive_bayes validation")
    utils.get_metrics(y_true = dataset.test_y,  y_pred = gradient_boosting_test_predictions, model_name="naive_bayes test")


if __name__ == '__main__':

    utils.start_logs()

    execute_naive_bayes( optimize_model=OPTIMIZE_MODEL,
                            train_model=TRAIN_MODEL,
                            save_model=SAVE_MODEL,
                            compute_predictions=COMPUTE_PREDICTIONS,
                            save_predictions=SAVE_PREDICTIONS,
                            train_test_split=TRAIN_TEST_SPLIT,
                            seed_number=SEED_NUMBER )