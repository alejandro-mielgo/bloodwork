from sklearn.svm import SVC
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


def optimize_svc(x_train : pd.DataFrame,
                 y_train: pd.DataFrame,
                 C:list = [0.01,0.1],
                 scoring = "precision",
                 n_points: int = 10000) -> dict:
    
    if n_points > x_train.shape[0]:
        n_points = x_train.shape[0]

    parameters : dict = {"C":C}

    svc_classifier = SVC()
    grid_search = GridSearchCV(svc_classifier, param_grid=parameters, cv=5, scoring=scoring )
    grid_search.fit(x_train.iloc[:n_points,:], y_train[:n_points])

    logging.info(f"Best {scoring} : {grid_search.best_score_:.3f}")
    logging.info(f"Best parameters:")
    logging.info(grid_search.best_params_)

    return grid_search.best_params_


def execute_svc(optimize_model : bool,
                train_model : bool,          #Train model, if False, load from file
                save_model : bool,
                compute_predictions : bool,  #Calculate predictions, if False, load from file
                save_predictions : bool,
                train_test_split : float,
                seed_number : int) -> None:

    #__ Load Data ________________________________________________________________________________
    diff : pd.DataFrame = pd.read_csv('./data/diff.csv')
    np.random.seed(seed_number)
    train_mask : np.ndarray = np.random.rand(len(diff)) < train_test_split

    train_x, train_y, test_x, test_y = df_utils.prepare_df( df_original = diff,
                                                            train_mask = train_mask,
                                                            normalize = 1,
                                                            cuad_features = False,
                                                            rate_features = False )
    
    #__ Find best parameters ______________________________________________________________________
    if optimize_model:
        parameters:dict = optimize_svc(x_train = train_x,
                                      y_train = train_y,
                                      C=[0.1,1],
                                      scoring = "precision",
                                      n_points = 10000)
    else:
        parameters:dict = {"C":1}
    
    #__ Train model _______________________________________________________________________________
    if train_model:
        logging.info("start svc classifier training")
        svc_classifier = SVC(C=parameters['C'])
        svc_classifier.fit(X=train_x, y=train_y)
    else:
        svc_classifier = utils.load_file('./models/svc_classifier.pkl')

    if save_model:
        utils.save_model(model=svc_classifier, model_name='svc_classifier')

    #__ Get_predictions ___________________________________________________________________________
    svc_val_predictions, svc_test_predictions, svc_val_probabilities, svc_test_probabilities = utils.manage_predictions(
        model = svc_classifier,
        model_name = "svc",
        x_train = train_x,
        y_train = train_y,
        x_test = test_x,
        y_test = test_y,
        compute_predictions = compute_predictions,
        save_predictions = save_predictions)

    utils.get_metrics(y_true = train_y, y_pred = svc_val_predictions, model_name="svc validation")
    utils.get_metrics(y_true = test_y,  y_pred = svc_test_predictions, model_name="svc test")


def compare_svc_features() -> None:
    pass

if __name__ == '__main__':

    logging.basicConfig(
        filename='bloodwork.log',
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info("SVC classifier")

    execute_svc(optimize_model=OPTIMIZE_MODEL,
                train_model=TRAIN_MODEL,
                save_model=SAVE_MODEL,
                compute_predictions=COMPUTE_PREDICTIONS,
                save_predictions=SAVE_PREDICTIONS,
                train_test_split=TRAIN_TEST_SPLIT,
                seed_number=SEED_NUMBER)