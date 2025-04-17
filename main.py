import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import df_utils
import numpy as np



@st.cache_data
def get_data(csv_path:str) -> pd.DataFrame:

    df:pd.DataFrame = pd.read_csv(csv_path)
    df.drop(columns=["Unnamed: 0","key","sex"],inplace=True)
    df.rename(columns={"wbit_error":'target'}, inplace=True)

    columns:list[str] = df.columns.to_list()
    columns.remove("target")

    return df,columns


def get_feature_stats(df:pd.DataFrame, active_feature:str) -> pd.DataFrame:

    stats:dict = {"mean":[df[df["target"]==0][active_feature].mean(), df[df["target"]==1][active_feature].mean()],
                  "stdv":[df[df["target"]==0][active_feature].std(), df[df["target"]==1][active_feature].std()],
                  "quantile 25":[df[df["target"]==0][active_feature].quantile(0.25), df[df["target"]==1][active_feature].quantile(0.25)],
                  "median":[df[df["target"]==0][active_feature].median() , df[df["target"]==1][active_feature].median()],
                  "quantile 75":[df[df["target"]==0][active_feature].quantile(0.75), df[df["target"]==1][active_feature].quantile(0.75)]
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
        plt.bar(missing.keys(), missing.values())
        plt.xticks(rotation=90)
        st.pyplot(fig)
    return { "n_rows":n_rows, "n_cols":n_cols }


if __name__ == "__main__":

    raw_data,columns = get_data(csv_path = "./data/diff.csv" )

    st.write("""
    # Wrong blood in tube data
    """)

    hist_tab, scatter_tab, health_tab, model_tab = st.tabs(["Histograms", "Scatter plot", "Data frame health", "Models"])

    with hist_tab:
        active_feature:str = st.selectbox(label="Select feature:", options=columns)
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            points_to_show:int = st.slider(label="Points to show",value=10000, min_value=1000, max_value=len(raw_data), step=1,key="points to show hist")
        with col_a2:
            n_bins:int = st.slider(label="Number of bins",value=35, min_value=10, max_value=70, step=1)

        fig, ax = plt.subplots()
        ax.set_title(f'{active_feature} histogram')
        sns.histplot(data=raw_data.iloc[:points_to_show], x=active_feature, hue="target", palette="Set2", bins=n_bins, kde=False, ax=ax)
        st.pyplot(fig)

        stats:pd.DataFrame = get_feature_stats(df=raw_data,active_feature=active_feature)
        st.table(stats)


    with scatter_tab:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            feature_x:str = st.selectbox(label="Select feature for x axis:", options=columns, key="selectbox_x")
        with col_b2:
            feature_y:str = st.selectbox(label="Select feature for y axis:", options=columns, key="selectbox_y")
        points_to_show_2:int = st.slider(label="Points to show",value=10000, min_value=1000, max_value=len(raw_data), step=1, key="points to show scatter")
        fig, ax = plt.subplots()
        sns.scatterplot(data=raw_data.iloc[:points_to_show_2], x=feature_x,y=feature_y, hue="target", palette="Set2", ax=ax)
        st.pyplot(fig)


    # with health_tab:
    #     st.write("## Original dataset")
    #     df_health_streamlit(dataframe=raw_data)

    #     st.write("### Removing missing data")
    #     threshold:float = st.number_input(label="Threshold of missing rows to remove column",value=0.1,min_value=0.0,max_value=1.0)
    #     clean_df:pd.DataFrame = df_utils.clean_missing(df_original=raw_data,cols_to_drop=None,method="delete",missing_threshold=threshold)
    
    #     df_health_streamlit(dataframe=clean_df,show_plot=False)