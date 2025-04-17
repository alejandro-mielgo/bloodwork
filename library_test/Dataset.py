import pandas as pd
from typing import Self
import numpy as np

class Dataset():
    
    def __init__(self,data:pd.DataFrame,) -> None:
        self.data:pd.DataFrame = data
        self.split:bool = False


    def __str__(self)->str:
        l1:str = f"Split dataframe: {self.split}\n"
        l2:str = f"Shape:           {self.data.shape}\n"
        l3:str = "Data:\n"+str(self.data.head())+"\n"
        return l1+l2+l3


    def clean_missing(self,missing_threshold:float=0.1,test_method:str='delete') -> Self:
        
        cols_to_be_removed = [] #cols to be removed given the train data
        
        for column in self.train.columns:
            if self.train[column].isna().sum() / self.train.shape[0] > missing_threshold:
                self.train.drop(column, axis=1, inplace=True)
                cols_to_be_removed.append(column)
        
        self.train.dropna(inplace=True)

        for column in cols_to_be_removed:
            self.test.drop(column, axis=1, inplace=True)
        
        if test_method == 'delete':
            self.test.dropna(inplace=True)
        
        elif test_method == 'mean':
            for column in self.test.columns:

                mean_value:float = self.train[column].mean()
                self.test.fillna({column:mean_value}, inplace=True)
        
        elif test_method == 'median':
            for column in self.test.columns:
                median_value:float = self.train[column].median()
                self.test.fillna({column:median_value}, inplace=True)
 
        return self
    

    def get_health(self, verbose:bool=False)->dict[str,int]:

        missing:dict = {}
        for column in self.data.columns:
            missing[column] = int(self.data[column].isna().sum())
            if verbose:
                print(f'{column}\t {missing[column]}')
        missing_rate = sum(missing.values()) / (self.data.shape[0] * self.data.shape[1])
        self.missing = missing
        self.missing_rate = missing_rate

        return missing
    

    def drop_columns(self,cols_to_drop:list[str])->Self:
        
        if self.split:
            raise ValueError("Data has already been split. Cannot drop columns.")
        
        for column in self.data.columns:
            if column in cols_to_drop:
                self.data.drop(column, axis=1, inplace=True)
        return self


    def normalize(self,max_value:float=1)->Self:
        for column in self.train.columns:
            if self.data[column].dtype == 'float64' or self.train[column].dtype=="int64":
                column_max = self.train[column].max()
                column_min = self.train[column].min() 
                
                self.train[column] = (self.train[column]-column_min)/(column_max-column_min) * max_value
                self.test[column] = (self.test[column]-column_min)/(column_max-column_min) * max_value
        return self


    def one_hot_encode(self)->Self:
        if self.split:
            raise ValueError("Data has already been split. Cannot one hot encode.")

        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                self.data = pd.get_dummies(self.data, columns=[column], drop_first=True, dtype=bool)
        return self


    def rename_target(self,target_name:str)->Self:
        if self.split:
            raise ValueError("Data has already been split. Cannot rename target.")
        if target_name not in self.data.columns:
            raise ValueError(f"Column {target_name} not found in the DataFrame.")
        
        self.data.rename(columns={target_name:'target'}, inplace=True)
        
        return self  


    def split_data(self,seed_number:int=0, train_test_split:float=0.8)->Self:
        np.random.seed(seed_number)
        train_mask : np.ndarray = np.random.rand(len(self.data)) < train_test_split
        self.train:pd.DataFrame = self.data[train_mask].copy()
        self.test:pd.DataFrame = self.data[~train_mask].copy()
        self.split = True
        return self
    

    def standarize(self) -> Self:

        for column in self.train.columns:
            if self.train[column].dtype == 'float64' or self.train[column].dtype == "int64":
                # Explicitly cast the column to float
                self.train[column] = self.train[column].astype('float64')
                self.test[column] = self.test[column].astype('float64')

                train_mean: float = self.train[column].mean()
                train_std: float = self.train[column].std()
                self.train[column] = ((self.train[column] - train_mean) / train_std)
                self.test[column] = ((self.test[column] - train_mean) / train_std)
        return self




if __name__ == "__main__":

    df = pd.DataFrame()
    df['c1'] = [0,1,2,3,4,1,2,3,4,2,1,2,3,2,3]
    df['c2'] = [5,6,7,8,9,2,3,4,7,6,4,3,5,4,5]
    df['c3'] = [1,2,3,3,4,3,4,5,6,3,5,None,5,4,3]
    df['c4'] = [1,2,3,4,5,2,3,4,5,2,3,None,5,None,12]
    df['c5'] = ['a','a','b','b','c','a','a','b','b','c','a','a','b','b','c']
    dataset = Dataset(data=df)

    dataset.get_health(verbose=False)
    print(dataset.missing)
    dataset.rename_target('c4').one_hot_encode().split_data().clean_missing(test_method='delete').standarize().normalize()
    print(dataset.train)
    print(dataset.test)

    print(dataset)