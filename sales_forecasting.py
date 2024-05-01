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

def data_preprocess(train_csv, test_csv, valid_csv, oil_csv, stores_csv, transactions_csv, holidays_events_csv):
    # List of datasets to iterate over
    datasets = {
        'train': train_csv,
        'test': test_csv,
        'valid': valid_csv
    }

    # Loop through train, test, and valid datasets to merge files
    for key in datasets:
        # Merging features to test, train, valid
        datasets[key] = datasets[key].merge(oil_csv, on='date', how='left')
        datasets[key] = datasets[key].merge(stores_csv, on='store_nbr', how='left')
        datasets[key] = datasets[key].merge(transactions_csv, on=['date', 'store_nbr'], how='left')
        datasets[key] = datasets[key].merge(holidays_events_csv, on='date', how='left')

    # We can add more pre-processing code here, feel free to update code structure

    
    return datasets['train'], datasets['test'], datasets['valid']

train_processed = merge_datasets(train_csv, test_csv, oil_csv, stores_csv, transactions_csv, holidays_events_csv)
test_processed = merge_datasets(train_csv, test_csv, oil_csv, stores_csv, transactions_csv, holidays_events_csv)
valid_processed =merge_datasets(train_csv, test_csv, oil_csv, stores_csv, transactions_csv, holidays_events_csv)
