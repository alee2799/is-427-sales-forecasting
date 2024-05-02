import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv('test.csv')
test = pd.read_csv('train.csv')
stores = pd.read_csv('stores.csv')
transactions = pd.read_csv('transactions.csv')
holidays = pd.read_csv('holidays_events.csv')
oil_prices = pd.read_csv('oil.csv')

#Implementation for merging datasets
#combine the features in stores.csv(city, state), oil.csv(dcoilwtico), 
#and holiday_events.csv(holiday date, type, locale, locale name), 
#transactions.csv (transactions) into train.csv, test.csv, valid.csv
# use the store_nbr found in stores.csv, and transactions.csv 
#to connect it to train/test.csv/valid.csv
# use the date attribute found in oil.csv and holiday.csv 
#and connect it to train/test.csv/valid.csv.
def preprocess(data, cols):
    #checks if there is an array of columns to remove then removes them
    if cols:
        data = data.drop(cols, axis=1)
    #removes rows with empty values
    data = data.dropna()
    
    
preprocess(train, [])

#adds columns to training and test sets; also matches columns by date
#def combine_datasets():
