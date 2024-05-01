import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train_csv = pd.read_csv('')
test_csv = pd.read_csv('')
# Must create the valid.csv file and load data
valid_csv = pd.read_csv('')
stores_csv = pd.read_csv('')
transactions_csv = pd.read_csv('')
holidays_events_csv = pd.read_csv('')
oil_csv = pd.read_csv('')


# Implementation for merging datasets
# combine the features in stores.csv(city, state), oil.csv(oil prices), and holiday_events.csv(holiday date, type, locale, locale name), transactions.csv (transactions) into train.csv, test.csv, valid.csv
# use the store_nbr found in stores.csv, and transactions.csv to connect it to train/test.csv/valid.csv
# use the date attribute found in oil.csv and holiday.csv and connect it to train/test.csv/valid.csv.

We will use the store_nbr found in stores.csv, and transactions.csv to connect it to train/test.csv. Then we will use the date attribute found in oil.csv and holiday.csv and connect it to train/test.csv/valid.csv.
def data_preprocess():
    
