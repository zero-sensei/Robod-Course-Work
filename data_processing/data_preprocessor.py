import pandas as pd
from pandas import DataFrame


class DataPreprocessor:
    def __init__(self, data):
        self.data: DataFrame = data

    def clean_data(self):
        self.data.drop(columns=['ID', 'BGG Rank', 'Complexity Average'], inplace=True)
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.data = self.data[self.data['Year Published'] > 1900]

        self.data.loc[self.data['Min Players'] <= 0, 'Min Players'] = 1
        self.data.loc[self.data['Max Players'] > 10, 'Max Players'] = 10
        self.data.loc[self.data['Play Time'] < 5, 'Play Time'] = 5
        self.data.loc[self.data['Min Age'] <= 0, 'Min Age'] = 1

    def transform_data(self):
        self.data.loc[:, 'Year Published'] = self.data['Year Published'].astype(int)
        domains_dummies = self.data['Domains'].str.get_dummies(sep=', ').add_prefix('Domain_')
        self.data.drop(columns=['Domains'], inplace=True)
        self.data = pd.concat([self.data, domains_dummies], axis=1)

    def normalize_data(self, features):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data[features])
        return X_scaled
