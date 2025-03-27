from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import logging
import numpy as np


import df_utils
import utils

#__ Config ___________________________________________________________________________________
OPTIMIZE_MODEL : bool = False
TRAIN_MODEL : bool = False
SAVE_MODEL : bool = False
COMPUTE_PREDICTIONS : bool = False
SAVE_PREDICTIONS : bool = False

TRAIN_TEST_SPLIT : float = 0.85
SEED_NUMBER : int = 0

def optimize_c(x_train : pd.DataFrame,
               y_train: pd.DataFrame,
               c_params : list[float] = [0.01, 0.1, 1, 10],
               scoring : str = 'precision',
               n_points: int = 40000 ,
               n_iterations : int = 2) -> float:
    
    best_score: float = 0
    best_c : float = 0

    if n_points > x_train.shape[0]:
        n_points = x_train.shape[0]

    logging.info(f"C Parameters iteration 0: {c_params}")

    for i in range(n_iterations):

        clf = LogisticRegression(random_state=0, max_iter=1000)
        parameters = {'C': c_params}

        grid_search = GridSearchCV(clf, param_grid=parameters, cv=5, scoring=scoring , n_jobs=2)
        grid_search.fit(x_train.iloc[:n_points,:], y_train[:n_points])

        best_index = grid_search.best_index_

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_c = grid_search.best_params_['C']

        c_params[0], c_params[3] = c_params[best_index]/2, c_params[best_index]*2
        c_params[1], c_params[2] = (c_params[0] + c_params[3])/3, (c_params[0] + c_params[3])*2/3

        logging.info(f"C Parameters iteration {i+1}: {c_params}")
    
    logging.info(f"Best C : {best_c:.4f}")
    logging.info(f"Best score : {best_score} ")

    return best_c


def execute_lr(optimize_model : bool,
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
                                                            cuad_features = True,
                                                            rate_features = False )
    
    #__ Find best parameters ______________________________________________________________________
    if optimize_model:
        best_c = optimize_c(x_train = train_x,
                            y_train = train_y,
                            c_params=[0.1,1,5,20],
                            scoring='precision',
                            n_points = 1000,
                            n_iterations = 2)
    else:
        best_c = 10

    #__ Train model _______________________________________________________________________________
    if train_model:
        logging.info("start logistic regression training")
        lr_classifier = LogisticRegression(C=best_c, max_iter=1000)
        lr_classifier.fit(X=train_x, y=train_y)
    else:
        lr_classifier = utils.load_file('./models/lr_classifier.pkl')

    if save_model:
        utils.save_model(model=lr_classifier, model_name='lr_classifier')

    #__ Get_predictions ___________________________________________________________________________

    lr_val_predictions, lr_test_predictions, lr_val_probabilities, lr_test_probabilities = utils.manage_predictions(
        model = lr_classifier,
        model_name = "lr",
        x_train = train_x,
        y_train = train_y,
        x_test = test_x,
        y_test = test_y,
        compute_predictions = compute_predictions,
        save_predictions = save_predictions)


    utils.get_metrics(y_true = train_y, y_pred = lr_val_predictions, model_name="logistic regression validation")
    utils.get_metrics(y_true = test_y,  y_pred = lr_test_predictions, model_name="logistic regression test")


if __name__ == "__main__":

    logging.basicConfig(
        filename='bloodwork.log',
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    execute_lr( optimize_model = OPTIMIZE_MODEL,
                train_model = TRAIN_MODEL,
                save_model = SAVE_MODEL,
                compute_predictions = COMPUTE_PREDICTIONS,
                save_predictions= SAVE_PREDICTIONS,
                train_test_split = TRAIN_TEST_SPLIT,
                seed_number = SEED_NUMBER )










