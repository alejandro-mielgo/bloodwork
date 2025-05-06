import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import df_utils
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix

import utils

# streamlit run c:/Users/a_mie/Desktop/bloodwork/main.py

@st.cache_data
def get_data(csv_path:str) -> tuple[pd.DataFrame,list[str]]:
    #Load raw data for histogram and scatter plots
    df:pd.DataFrame = pd.read_csv(csv_path)
    df.drop(columns=["Unnamed: 0","key"],inplace=True)
    df.rename(columns={"wbit_error":'target'}, inplace=True)
    df = df_utils.one_hot(df_original=df)

    columns:list[str] = df.columns.to_list()
    columns.remove("target")

    return df,columns


def get_feature_stats(df:pd.DataFrame, active_feature:str) -> pd.DataFrame:

    stats:dict = {"mean":[df[df["target"]==0][active_feature].mean(), df[df["target"]==1][active_feature].mean()],
                  "stdv":[df[df["target"]==0][active_feature].std(), df[df["target"]==1][active_feature].std()],
                  "min":[df[df["target"]==0][active_feature].min(), df[df["target"]==1][active_feature].min()],
                  "quantile 25":[df[df["target"]==0][active_feature].quantile(0.25), df[df["target"]==1][active_feature].quantile(0.25)],
                  "median":[df[df["target"]==0][active_feature].median() , df[df["target"]==1][active_feature].median()],
                  "quantile 75":[df[df["target"]==0][active_feature].quantile(0.75), df[df["target"]==1][active_feature].quantile(0.75)],
                  "max":[df[df["target"]==0][active_feature].max(), df[df["target"]==1][active_feature].max()]
                  }

    df = pd.DataFrame.from_dict( stats,
                                 orient="index",
                                 columns=["target 0", "target 1"] )
    return df


def df_health_streamlit( dataframe : pd.DataFrame,verbose:bool=True, show_plot:bool=True) -> dict[str,int]:
    
    missing: dict[str,int] = {}

    n_cols : int = dataframe.shape[1]
    n_rows : int = dataframe.shape[0]

    for column in dataframe.columns:
        missing[column] = dataframe[column].isna().sum()

    missing_rate:float = sum(missing.values()) / (n_cols * n_rows)
    if verbose:
        st.write(f'Number of samples: **{n_rows}**  \n',
                f'Number of features: **{n_cols}**  \n',
                f'Missing data ratio: **{round(missing_rate,4)}**')
    
    if show_plot:
        fig = plt.figure(figsize=(14, 5))
        # add a color gradient to the bars red for high values and green for low values
        colors = ['#FF0000' if v > 0.15*len(dataframe) else '#00FF00' for v in missing.values()]
        plt.bar(missing.keys(), missing.values(), color=colors)
        plt.title("Missing data per feature")
        plt.xlabel("Feature name")
        plt.xticks(rotation=90)
        st.pyplot(fig)
    return { "n_rows":n_rows, "n_cols":n_cols }


def compare_populations(df:pd.DataFrame, feature:str, significant_threshold:float = 0.05) -> pd.DataFrame:

    u_1,p = mannwhitneyu(df[df["target"]==0][feature], df[df["target"]==1][feature], alternative="two-sided",nan_policy="omit")
    n_0:int = len(df[df["target"]==0][feature]) 
    n_1:int = len(df[df["target"]==1][feature])
    u_2:float = n_0*n_1 - u_1

    st.write(f"### Mann-Whitney U test for {feature}  \n",
             f"h_0: Both populations are the same  \n",
             f"h_1: Populations are different  \n")

    st.write(f"U statistic target 0: {u_1}  \n",
             f"U statistic target 1: {u_2}  \n",
             f"p-value: {p}  \n")
    
    if p < significant_threshold:
        st.write(f"The difference between the two populations **is significant** with alpha = {significant_threshold}  \n",)
    else:
        st.write(f"The difference between the two populations **is not significant** with alpha = {significant_threshold}  \n",)


@st.cache_data
def load_predictions() -> tuple[pd.DataFrame,pd.DataFrame]:
    val_predictions = pd.read_csv("./predictions/train_predictions.csv")
    test_predictions = pd.read_csv("./predictions/test_predictions.csv")
    return val_predictions, test_predictions


@st.cache_data
def load_summary() -> tuple[pd.DataFrame,pd.DataFrame]:
    val_summary:pd.DataFrame = pd.read_csv("./predictions/val_summary.csv")
    test_summary:pd.DataFrame = pd.read_csv("./predictions/test_summary.csv")
    val_summary = val_summary.drop(columns=['Unnamed: 0'])
    test_summary = test_summary.drop(columns=['Unnamed: 0'])
    metrics:list[str] = ['accuracy','f1','recall','precision','true_neg','false_pos','false_neg','true_pos']
    val_summary['metric'] = metrics
    test_summary['metric'] = metrics
    val_summary.set_index('metric', inplace=True)
    test_summary.set_index('metric', inplace=True)
    val_summary = val_summary.transpose()
    test_summary = test_summary.transpose()
    
    return val_summary, test_summary

def get_model_performance(model_name:str,val_predictions:pd.DataFrame,test_predictions:pd.DataFrame) -> None:

    val_metrics = utils.get_metrics(y_true=val_predictions['target'], y_pred=val_predictions[model_name], graph=False)
    confusion_matrix_val:np.ndarray = confusion_matrix(y_true=val_predictions['target'], y_pred=val_predictions[model_name])
    
    test_metrics = utils.get_metrics(y_true=test_predictions['target'], y_pred=test_predictions[model_name], graph=False)
    confusion_matrix_test:np.ndarray = confusion_matrix(y_true=test_predictions['target'], y_pred=test_predictions[model_name])

    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts_v = ["{0:0.0f}".format(value) for value in confusion_matrix_val.flatten()]
    group_percentages_v = ["{0:.2%}".format(value) for value in confusion_matrix_val.flatten() / np.sum(confusion_matrix_val)]
    labels_v = [f"{v1}\n{v2}\n{v3}"
                for v1, v2, v3 in zip(group_names, group_counts_v, group_percentages_v)]
    labels_v = np.asarray(labels_v).reshape(2, 2)

    group_counts_t = ["{0:0.0f}".format(value) for value in confusion_matrix_test.flatten()]
    group_percentages_t = ["{0:.2%}".format(value) for value in confusion_matrix_test.flatten() / np.sum(confusion_matrix_test)]
    labels_t = [f"{v1}\n{v2}\n{v3}"
                for v1, v2, v3 in zip(group_names, group_counts_t, group_percentages_t)]
    labels_t= np.asarray(labels_t).reshape(2, 2)


    #plot confusion matrix
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    
    sns.heatmap(confusion_matrix_val,annot=labels_v, fmt="", cmap="Blues", ax=ax[0])
    ax[0].set_title("Validation confusion matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    
    sns.heatmap(confusion_matrix_test, annot=labels_t, fmt="", cmap="Blues", ax=ax[1])
    ax[1].set_title("Test confusion matrix")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    st.pyplot(fig)

    df:pd.DataFrame = pd.DataFrame.from_dict(val_metrics, orient="index", columns=["validation"])
    df["test"] = test_metrics.values()
    st.write(df)

    
if __name__ == "__main__":

    raw_data,columns = get_data(csv_path = "./data/diff.csv" )
    st.write("# Wrong blood in tube data")
    about_tab, hist_tab, scatter_tab, health_tab, model_tab, comparative = st.tabs(["About","Histograms", "Scatter plot", "Data health", "Models","Model Comparative"])

    with about_tab:
        st.write("### About this app")
        st.write("This app is designed to help you explore the WBIT dataset.  \n",
                 "You can visualize the data using histograms and scatter plots,  \n",
                 "check the health of the data, and evaluate the performance of different models.  \n",
                 "The dataset contains information about blood samples and whether they were correctly labeled or not.  \n")
        st.write("### Data Source")
        st.write("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XCYHPX \n")


    with hist_tab:

        active_feature:str = st.selectbox(label="Select feature:", options=columns)
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            points_to_show:int = st.slider(label="Points to show",value=10000, min_value=1000, max_value=len(raw_data), step=1000,key="points to show hist")
        with col_a2:
            n_bins:int = st.slider(label="Number of bins",value=35, min_value=10, max_value=70, step=1, key='n_bins')

        fig, ax = plt.subplots()
        ax.set_title(f'{active_feature} histogram')
        sns.histplot(data=raw_data.iloc[:points_to_show], x=active_feature, hue="target", palette="Set2", bins=n_bins, kde=False, ax=ax)
        st.pyplot(fig)

        stats:pd.DataFrame = get_feature_stats(df=raw_data,active_feature=active_feature)
        st.table(stats)

        compare_populations(df=raw_data, feature=active_feature)


    with scatter_tab:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            feature_x:str = st.selectbox(label="Select feature for x axis:", options=columns, key="selectbox_x", index=2)
        with col_b2:
            feature_y:str = st.selectbox(label="Select feature for y axis:", options=columns, key="selectbox_y",index=3)
        points_to_show_2:int = st.slider(label="Points to show",value=5000, min_value=1000, max_value=len(raw_data), step=1000, key="points to show scatter")
        fig, ax = plt.subplots()
        sns.scatterplot(data=raw_data.iloc[:points_to_show_2], x=feature_x,y=feature_y, hue="target", palette="Set2", ax=ax)
        st.pyplot(fig)


    with health_tab:
        st.write("## Original dataset")
        df_health_streamlit(dataframe=raw_data)


    with model_tab:
        options:dict[str,str]={ "ada boost":"ada_boost",
                                "gradient boosting": "gradient_boosting",
                                "k nearest neighbors":"kn",
                                "logistic regression":"lr",
                                "support vector classifier":"svc",
                                "neural network":"nn",
                                "cat boost":"catboost",
                                "majority vote":"majority_vote",
                                # "neural network ensemble":"nn_ensemble"
                                }

        


        option:str = st.selectbox(label="Select model:", options=options.keys(), key="selectbox_model")
        model_name:str = options[option]

        val_predictions, test_predictions = load_predictions()
        st.write(f'### model performance for {option}')
        get_model_performance( model_name=model_name,
                               val_predictions=val_predictions,
                               test_predictions=test_predictions)
        
    
    with comparative:
        train_summary,test_summary = load_summary()
        st.write('### Results with validation data')
        st.write(train_summary)
        st.write('### Results with test data')
        st.write(test_summary)