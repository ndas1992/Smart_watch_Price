from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from category_encoders.target_encoder import TargetEncoder
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
    df = df.rename(columns={'Current Price': 'Current_Price', 
                       'Original Price': 'Original_Price',
                       'Discount Percentage':'Discount_Percentage',
                       'Number OF Ratings': 'Number_of_Ratings',
                       'Model Name': 'Model_Name',
                       'Dial Shape': 'Dial_Shape',
                       'Strap Color': 'Strap_Color',
                       'Strap Material': 'Strap_Material',
                       'Battery Life (Days)': 'Battery_Life',
                       'Display Size': 'Display_Size'})

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


# ## Select important Categorical Columns
# def select_imp_cat_columns(train_df, categorical_columns):
#     important_cat_columns = []
#     for column in categorical_columns:
#         CategoryGroupLists = train_df.groupby(column)['Discount Percentage'].apply(list)
#         AnnovaResults = f_oneway(*CategoryGroupLists)
#         #print(f'{column} : P-Value for annova is: {AnnovaResults[1]}')
#         if AnnovaResults[1]<0.7:
#             important_cat_columns.append(column)
#     #print(important_cat_columns)
#     return important_cat_columns


## Perform Target Encoding
def target_encoding_categorical(train_df):
    important_cat_columns = ['Brand', 'Model_Name', 'Dial_Shape', 'Strap_Color', 'Strap_Material', 'Touchscreen']
    
    # set up the encoder
    cat_encoder = TargetEncoder(cols=important_cat_columns)

    # fit the encoder - finds the mean target value per category
    cat_encoder.fit(train_df[important_cat_columns], train_df['Discount_Percentage'])
    pickle.dump(cat_encoder, open(r'src\models\cat_encoder.pkl', 'wb'))

    train_df[important_cat_columns] = cat_encoder.transform(train_df[important_cat_columns])
    return train_df, important_cat_columns

## Data preprocessing
def process_data(new=False):
    df = read_full_data(new)
    df.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')

    ## Train Test Split
    train_df = df.sample(frac=0.8)
    test_df = df.drop(train_df.index)

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_df.to_csv(r'src\data\test_df.csv')

    ## Clean the weight column
    train_df['Weight'] = train_df['Weight'].replace(np.nan, '0')
    train_df['Weight'] = train_df['Weight'].map(clean_weight)
    train_df['Weight'] = train_df['Weight'].replace(0.0, np.nan)

    ## Clean the Displaysize column
    train_df['Display_Size'] = train_df['Display_Size'].replace(np.nan, '0.0 inches')
    train_df['Display_Size'] = train_df['Display_Size'].map(clean_display)
    train_df['Display_Size'] = train_df['Display_Size'].replace(0.0, np.nan)

    ## Seperating Numerical and Categorical Data
    categorical_columns = [column for column in train_df.columns if train_df[column].dtype=='O']
    numerical_columns = [column for column in train_df.columns if train_df[column].dtype!='O']

    ## Handling NaN values in data
    for column in numerical_columns:
        train_df[column] = train_df[column].replace(np.nan, train_df[column].median())

    for column in categorical_columns:
        train_df[column] = train_df[column].replace(np.nan, train_df[column].mode()[0])

    ## Performing Target Encoding
    train_df, imp_categorical_columns = target_encoding_categorical(train_df)

    ## Split dependent and independent features
    all_columns = numerical_columns + imp_categorical_columns
    all_columns.remove('Discount_Percentage')

    X = train_df[all_columns]
    y = train_df['Discount_Percentage']

    return X, y


## Train Model
def train(new):
    X, y = process_data(new)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)

    en_model = ElasticNet()
    en_model.fit(X_train, y_train)

    param_grid = {
    'l1_ratio': [.1, .5, .7, .9, .95, .99, 1]
    }

    grid_search = GridSearchCV(estimator=en_model, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_param = grid_search.best_params_

    best_model = ElasticNet(**best_param)
    best_model.fit(X_train, y_train)

    pickle.dump(scalar, open(r'src\models\scalar.pkl', 'wb'))
    pickle.dump(best_model, open(r'src\models\best_model.pkl', 'wb'))


if __name__=='__main__':
    train(False)





