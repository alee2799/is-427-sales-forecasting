import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""
Implementation for preprocessing datasets
combine the features in stores.csv(city, state), oil.csv(dcoilwtico),
and holiday_events.csv(holiday date, type, locale, locale name),
transactions.csv (transactions) into train.csv, test.csv, valid.csv
use the store_nbr found in stores.csv, and transactions.csv
to connect it to train/test.csv/valid.csv
use the date attribute found in oil.csv and holiday.csv
and connect it to train/test.csv/valid.csv.
"""

#reads files and combines them together then returns the created data set
def combine_data(files):
	data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
	return data

train = combine_data(['train.csv', 'stores.csv', 'oil.csv', 'holidays_events.csv', 'transactions.csv'])
test = combine_data(['test.csv', 'stores.csv', 'oil.csv', 'holidays_events.csv', 'transactions.csv'])
train.info()

def preprocess(data, cols):
	#checks if there is an array of columns to remove then removes them from dataset
	if cols:
		data = data.drop(cols, axis=1)
	#removes rows with empty values
	data = data.dropna(how='any')
	return data

train = preprocess(train, ['type', 'cluster', 'description', 'transferred'])
test = preprocess(test, ['type', 'cluster', 'description', 'transferred'])

#creates a valid set from splitting test set
#def split_sets()
