
import pandas as pd
import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script is used to find how deleting columns with missing values affects the size of the dataset.
"""

if __name__ == "__main__":
    df:pd.DataFrame = pd.read_csv('data/diff.csv')
    n_cols:list[int]=[]
    n_rows:list[int]=[]
    thresholds:list[float] = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25]
    for threshold in thresholds:
        dataset = Dataset.Dataset(data=df)
        dataset.drop_columns(['Unnamed: 0','key']).rename_target('wbit_error').one_hot_encode().split_train_test()
        dataset.clean_missing(missing_threshold=threshold)
        n_cols.append(dataset.train.shape[1])
        n_rows.append(dataset.train.shape[0])

    
    size:pd.DataFrame = pd.DataFrame(data={"missing_threshold":thresholds,"n_cols":n_cols,"n_rows":n_rows})
    size['n_points'] = size['n_rows'] * size['n_cols']
    
    #line plot in seaborn for size dataframe
    plt.figure(figsize=(10,5))
    sns.lineplot(data=size,x='missing_threshold',y='n_rows',label='n_rows',color='r')
    sns.lineplot(data=size,x='missing_threshold',y='n_cols',label='n_cols',ax=plt.gca().twinx(),color='b')
    plt.title('Number of rows and columns after cleaning')
    plt.show()

    print(size)