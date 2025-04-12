from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import logging

import utils
import df_utils

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
                                C:list = [0.01,0.1],
                                scoring = "precision",
                                n_points: int = 10000
                                ) -> dict:
    
    if n_points > x_train.shape[0]:
        n_points = x_train.shape[0]

    parameters : dict = {"C":C}

    gradient_boosting_classifier = GradientBoostingClassifier(random_state=SEED_NUMBER)
    grid_search = GridSearchCV(gradient_boosting_classifier, param_grid=parameters, cv=5, scoring=scoring )
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
    diff : pd.DataFrame = pd.read_csv('./data/diff.csv')
    np.random.seed(seed_number)
    train_mask : np.ndarray = np.random.rand(len(diff)) < train_test_split

    train_x, train_y, test_x, test_y = df_utils.prepare_df( df_original = diff,
                                                            target_name = "wbit_error",
                                                            train_mask = train_mask,
                                                            standarize = 1,
                                                            cuad_features = False,
                                                            rate_features = False )
    
    #__ Find best parameters ______________________________________________________________________
    if optimize_model:
        parameters:dict = optimize_gradient_boosting( x_train = train_x,
                                                      y_train = train_y,
                                                      C=[0.1,1],
                                                      scoring = "precision",
                                                      n_points = 10000)
    else:
        parameters:dict = {"C":1}
    
    #__ Train model _______________________________________________________________________________
    if train_model:
        logging.info("start gradient_boosting classifier training")
        gradient_boosting_classifier = GradientBoostingClassifier(random_state=SEED_NUMBER)
        gradient_boosting_classifier.fit(X=train_x, y=train_y)
    else:
        gradient_boosting_classifier = utils.load_file('./models/gradient_boosting_classifier.pkl')

    if save_model:
        utils.save_model(model=gradient_boosting_classifier, model_name='gradient_boosting_classifier')

    #__ Get_predictions ___________________________________________________________________________
    gradient_boosting_val_predictions, gradient_boosting_test_predictions, gradient_boosting_val_probabilities, gradient_boosting_test_probabilities = utils.manage_predictions(
        model = gradient_boosting_classifier,
        model_name = "gradient_boosting",
        x_train = train_x,
        y_train = train_y,
        x_test = test_x,
        y_test = test_y,
        compute_predictions = compute_predictions,
        save_predictions = save_predictions)

    utils.get_metrics(y_true = train_y, y_pred = gradient_boosting_val_predictions, model_name="gradient-boosting validation")
    utils.get_metrics(y_true = test_y,  y_pred = gradient_boosting_test_predictions, model_name="gradient-boosting test")


if __name__ == '__main__':

    logging.basicConfig(
        filename='bloodwork.log',
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info("gradient-boosting classifier")

    execute_gradient_boost( optimize_model=OPTIMIZE_MODEL,
                            train_model=TRAIN_MODEL,
                            save_model=SAVE_MODEL,
                            compute_predictions=COMPUTE_PREDICTIONS,
                            save_predictions=SAVE_PREDICTIONS,
                            train_test_split=TRAIN_TEST_SPLIT,
                            seed_number=SEED_NUMBER)