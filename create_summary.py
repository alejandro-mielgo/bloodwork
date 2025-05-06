import pandas as pd
import numpy as np

from ensemble import get_predictions_paths, load_predictions, load_data
import utils
from Dataset import Dataset


TRAIN_TEST_SPLIT : float = 0.85
SEED_NUMBER : int = 0

def create_summary_df(models:list[str],predictions:pd.DataFrame, true_labels:np.ndarray)->pd.DataFrame:
    summary_df:pd.DataFrame = pd.DataFrame()
    for model in models:
        metrics = utils.get_metrics(    y_true=true_labels,
                                        y_pred=predictions[model],
                                        model_name=model,
                                        graph=False)   
        # metrics_dict = {key: [value] for key, value in metrics.items()}
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        print(metrics_df.head(10))
        summary_df[model] = metrics_df
    return summary_df



if __name__=='__main__':

    utils.start_logs()
    
    val_predictions_paths,test_predictions_paths = get_predictions_paths(get_ensemble=True)
    dataset:Dataset = load_data(seed_number=SEED_NUMBER, train_test_split=TRAIN_TEST_SPLIT)
    
    train_predictions:pd.DataFrame = load_predictions(val_predictions_paths)
    test_predictions:pd.DataFrame = load_predictions(test_predictions_paths,test=True)
    train_true_labels:np.ndarray = dataset.train_y.to_numpy()
    test_true_labels:np.ndarray = dataset.test_y.to_numpy()

    models:list[str] = train_predictions.columns.to_list()
    print(models)

    val_summary_df = create_summary_df(models=models, predictions=train_predictions,true_labels=train_true_labels)
    val_summary_df.to_csv('./predictions/val_summary.csv')
    test_summary_df = create_summary_df(models=models, predictions=test_predictions,true_labels=test_true_labels)
    test_summary_df.to_csv('./predictions/test_summary.csv')








