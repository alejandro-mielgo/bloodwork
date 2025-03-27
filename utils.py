import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
import logging
import seaborn as sns
from scipy import stats
import pickle
import numpy as np
import os.path

import df_utils

def save_model(model:any, model_name:str) -> bool:
    with open(f'./models/{model_name}.pkl', 'wb') as fid:
        logging.info(f"saving model to ./models/{model_name}.pkl")
        pickle.dump(model, fid)
    return True


def load_file(file_path:str) -> any:
    logging.info(f"loading {file_path}")
    with open(f"{file_path}", 'rb') as fid:
        objecto = pickle.load(fid)
    return objecto   


def save_prediction( prediction : np.ndarray, prediction_name : str) -> bool:
        with open(f'./predictions/{prediction_name}.pkl', 'wb') as fid:
            logging.info(f"saving model to ./predictions/{prediction_name}.pkl")
            pickle.dump(prediction, fid)


def get_predictions( model : any,
                     x_data : pd.DataFrame
                     ) -> tuple:

    logging.info(f'Getting test predictions for {model}')
    predictions = model.predict(x_data)

    try:
        logging.info(f'Trying to get test predictions for {model}')
        predictions_proba = model.predict_proba(x_data)
    except:
        logging.info(f'no probabilities available for {model}')
        predictions_proba = None


    return predictions, predictions_proba


def get_cross_val_predictions(model : any, 
                              x_data : pd.DataFrame, 
                              y_data : pd.DataFrame,
                              ) -> tuple:
    
    logging.info(f'Getting cross validation predictions for {model}')

    cross_val_predictions = cross_val_predict(estimator = model,
                                                X=x_data, 
                                                y=y_data,
                                                cv = 5,
                                                method="predict")
    try:
        logging.info(f'Trying to get cross validation probabilities for {model}')
        cross_val_probabilities = cross_val_predict(estimator = model,
                                                    X=x_data, 
                                                    y=y_data,
                                                    cv = 5,
                                                    method="predict_proba")
    except:
        logging.info(f'No probabilities available for {model}')
        cross_val_probabilities = None
    
    return  cross_val_predictions, cross_val_probabilities


def get_metrics(y_true, y_pred, model_name:str="model", graph:bool=True) -> None:
    
    metrics = {}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    metrics['accuracy'] = acc
    metrics['f1'] = f1
    metrics['recall'] = recall
    metrics['precision'] = precision

    logging.info(f"{model_name} accuracy: {round(acc,4)}")
    logging.info(f"{model_name} f1 score: {round(f1,4)}")
    logging.info(f"{model_name} recall: {round(recall,4)}")
    logging.info(f"{model_name} precision: {round(precision,4)}")
    cm = confusion_matrix(y_true, y_pred)
    logging.info(f"{model_name} confusion matrix \n {str(cm)}")
    
    if graph:
        palette = sns.diverging_palette(10, 145, as_cmap=True)

        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)

        fig,ax = plt.subplots(figsize=(5, 4))
        plt.title(f"{model_name}")
        sns.heatmap(cm, annot=labels, fmt="", ax=ax, cmap="flare")
        plt.savefig(f'./images/{model_name}.png', bbox_inches='tight')

    return metrics


def compare_features(x_a : pd.DataFrame,
                     y_a : pd.DataFrame,
                     x_b : pd.DataFrame,
                     y_b : pd.DataFrame,
                     feat_name_a : str,
                     feat_name_b : str,
                     model,
                     metric : str = "precision",
                     n_folds : int = 10,
                     n_samples : int = None,
                     significance_threshold : int = 0.05
                     ) -> tuple:
    
    if n_samples is not None:
        if n_samples > x_a.shape[0]:
            n_samples = x_a.shape[0]
        
        x_a = x_a.iloc[:n_samples]
        y_a = y_a.iloc[:n_samples]
        x_b = x_b.iloc[:n_samples]
        y_b = y_b.iloc[:n_samples]

    scores_a = cross_val_score(model, x_a, y_a, cv=n_folds, scoring=metric)
    scores_b = cross_val_score(model, x_b, y_b, cv=n_folds, scoring=metric)
    print(f"{feat_name_a} average {metric} : {scores_a.mean()}")
    print(f"{feat_name_b} average {metric} : {scores_b.mean()}")
    
    # statistical test
    t_stat,p_value = stats.ttest_ind(scores_a, scores_b)
    print(f"t_stat: {t_stat} ")
    print(f"p_value: {p_value}")

    if p_value < significance_threshold:
        print(f"p value is smaller than {significance_threshold}")
    else:
        print(f"Can't reject null hypothesis, both datasets behave equally")

    
    # plot
    scores = {feat_name_a:scores_a, feat_name_b:scores_b}
    scores = pd.DataFrame.from_dict(scores)
    sns.histplot(data = scores).set_title("Comparison between datasets")
    
    return t_stat, p_value


def manage_predictions(model : any,
                       model_name : str,
                       x_train : pd.DataFrame,
                       y_train : pd.DataFrame,
                       x_test : pd.DataFrame,
                       y_test : pd.DataFrame,
                       compute_predictions : bool,
                       save_predictions : bool ) -> tuple:
   
    if compute_predictions:
        val_predictions, val_probabilities = get_cross_val_predictions(model = model, x_data=x_train, y_data=y_train)
        test_predictions, test_probabilities = get_predictions(model=model, x_data = x_test )

        if save_predictions:
            save_prediction(prediction = val_predictions, prediction_name=f"{model_name}_val_predictions")
            save_prediction(prediction = test_predictions, prediction_name=f"{model_name}_test_predictions")
            if val_probabilities is not None:
                save_prediction(prediction = val_probabilities, prediction_name=f"{model_name}_val_probabilities")
            if test_probabilities is not None:
                save_prediction(prediction = test_probabilities, prediction_name=f"{model_name}_test_probabilities")

    else:
        val_predictions = load_file(f"./predictions/{model_name}_val_predictions.pkl")
        test_predictions = load_file(f"./predictions/{model_name}_test_predictions.pkl")
        
        if os.path.isfile(f"./predictions/{model_name}_val_probabilities.pkl"):
            logging.info(f"No probability files to load for {model_name}")
            val_probabilities = load_file(f"./predictions/{model_name}_val_probabilities.pkl")
            test_probabilities = load_file(f"./predictions/{model_name}_test_probabilities.pkl")
    
    return val_predictions, test_predictions, val_probabilities, test_probabilities