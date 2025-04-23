from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import logging
import numpy as np
import time


import Dataset
import utils

#__ Config ___________________________________________________________________________________
OPTIMIZE_MODEL : bool = False
TRAIN_MODEL : bool = True
SAVE_MODEL : bool = True
COMPUTE_PREDICTIONS : bool = True
SAVE_PREDICTIONS : bool = True

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
    original_df : pd.DataFrame = pd.read_csv('./data/diff.csv')
    np.random.seed(seed_number)

    dataset = Dataset.Dataset(data=original_df)
    dataset.drop_columns(['Unnamed: 0','key','pat_age_yrs','sex']).rename_target('wbit_error').split_train_test(seed_number=seed_number, train_test_split=train_test_split).clean_missing(missing_threshold=0.15)
    dataset.create_cuadratic_features().standarize().split_x_y()
    logging.info(dataset.train.head())
    
    #__ Find best parameters ______________________________________________________________________
    if optimize_model:
        best_c = optimize_c(x_train = dataset.train_x,
                            y_train = dataset.train_y,
                            c_params=[0.1,1,5,20],
                            scoring='precision',
                            n_points = 1000,
                            n_iterations = 2)
    else:
        best_c = 1

    #__ Train model _______________________________________________________________________________
    if train_model:
        logging.info("start logistic regression training")
        lr_classifier = LogisticRegression(C=best_c, max_iter=1000)
        lr_classifier.fit(X=dataset.train_x, y=dataset.train_y)
    else:
        lr_classifier = utils.load_file('./models/lr_classifier.pkl')

    if save_model:
        utils.save_model(model=lr_classifier, model_name='lr_classifier')

    #__ Get_predictions ___________________________________________________________________________

    lr_val_predictions, lr_test_predictions, lr_val_probabilities, lr_test_probabilities = utils.manage_predictions(
        model = lr_classifier,
        model_name = "lr",
        x_train = dataset.train_x,
        y_train = dataset.train_y,
        x_test = dataset.test_x,
        y_test = dataset.test_y,
        compute_predictions = compute_predictions,
        save_predictions = save_predictions)


    utils.get_metrics(y_true = dataset.train_y, y_pred = lr_val_predictions, model_name="logistic regression validation")
    utils.get_metrics(y_true = dataset.test_y,  y_pred = lr_test_predictions, model_name="logistic regression test")


if __name__ == "__main__":  

    utils.start_logs()

    execute_lr( optimize_model = OPTIMIZE_MODEL,
                train_model = TRAIN_MODEL,
                save_model = SAVE_MODEL,
                compute_predictions = COMPUTE_PREDICTIONS,
                save_predictions= SAVE_PREDICTIONS,
                train_test_split = TRAIN_TEST_SPLIT,
                seed_number = SEED_NUMBER )










