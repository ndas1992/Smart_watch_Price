from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from scipy.stats import f_oneway
import pandas as pd
import numpy as np
import os, glob, pickle
import re


def read_full_data(new=False):
    df = pd.read_csv(r'src\data\smartwatches.csv')

    if new == True:
        new_df = pd.read_csv(r'src\data\new_data.csv')
        df = df.concat(new_df, axis=1)
    return df


def clean_weight(x):
    match = re.findall(r'\d+', x)
    int_match = list(map(float, match))
    return np.mean(int_match)


def clean_display(x):
    match = re.findall(r'\d+.\d+', x)
    int_match = list(map(float, match))
    return int_match[0]


## Select important Categorical Columns
def select_imp_cat_columns(train_df, categorical_columns):
    important_cat_columns = []
    for column in categorical_columns:
        CategoryGroupLists = train_df.groupby(column)['Discount Percentage'].apply(list)
        AnnovaResults = f_oneway(*CategoryGroupLists)
        print(f'{column} : P-Value for annova is: {AnnovaResults[1]}')
        if AnnovaResults[1]<0.7:
            important_cat_columns.append(column)
    return important_cat_columns


## Perform Target Encoding
def target_encoding(train_df, numerical_columns):
    important_cat_columns = select_imp_cat_columns(important_cat_columns)
    for column in important_cat_columns:
        target_mean = train_df.groupby(column)['Discount Percentage'].mean()
        train_df[column] = train_df[column].map(target_mean)
    important_columns = important_cat_columns + numerical_columns
    return train_df[important_columns]


## Data preprocessing
def process_data(new=False):
    df = read_full_data(new)
    df.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')

    ## Train Test Split
    train_df = df.sample(frac=0.8)
    test_df = df.drop(train_df.index)

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_df.to_csv('src\data\test_df.csv')

    ## Clean the weight column
    train_df['Weight'] = train_df['Weight'].replace(np.nan, '0')
    train_df['Weight'] = train_df['Weight'].map(clean_weight)
    train_df['Weight'] = train_df['Weight'].replace(0.0, np.nan)

    ## Clean the Displaysize column
    train_df['Display Size'] = train_df['Display Size'].replace(np.nan, '0.0 inches')
    train_df['Display Size'] = train_df['Display Size'].map(clean_display)
    train_df['Display Size'] = train_df['Display Size'].replace(0.0, np.nan)

    ## Seperating Numerical and Categorical Data
    categorical_columns = [column for column in train_df.columns if train_df[column].dtype=='O']
    numerical_columns = [column for column in train_df.columns if train_df[column].dtype!='O']

    ## Handling NaN values in data
    for column in numerical_columns:
        train_df[column] = train_df[column].replace(np.nan, train_df[column].median())

    for column in categorical_columns:
        train_df[column] = train_df[column].replace(np.nan, train_df[column].mode()[0])

    ## Performing Target Encoding
    train_df = target_encoding(train_df, numerical_columns)

    ## Split dependent and independent features
    all_columns = [train_df.columns]
    all_columns.remove('Discount Percentage')

    X = train_df[all_columns]
    y = train_df['Discount Percentage']

    return X, y


## Train Model
def train_model():
    X, y = process_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)

    pickle.dump(scalar, open(r'src\models\scalar.pkl', 'wb'))

    en_model = ElasticNet()
    en_model.fit(X_train, y_train)

    y_pred = en_model.predict(scalar.transform(X_test))
    res = r2_score(y_true=y_test, y_pred=y_pred)

    param_grid = {
    'l1_ratio': [.1, .5, .7, .9, .95, .99, 1]
    }

    grid_search = GridSearchCV(estimator=en_model, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_param = grid_search.best_params_

    best_model = ElasticNet(**best_param)
    best_model.fit(X_train, y_train)

    pickle.dump(best_model, open(r'src\models\best_model.pkl', 'wb'))







