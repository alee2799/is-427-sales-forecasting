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

def data_preprocess():
    
