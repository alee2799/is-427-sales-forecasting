import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Implementation for preprocessing datasets
#combine the features in stores.csv(city, state), oil.csv(dcoilwtico), 
#and holiday_events.csv(holiday date, type, locale, locale name), 
#transactions.csv (transactions) into train.csv, test.csv, valid.csv
# use the store_nbr found in stores.csv, and transactions.csv 
#to connect it to train/test.csv/valid.csv
# use the date attribute found in oil.csv and holiday.csv 
#and connect it to train/test.csv/valid.csv.

#reads files and combines them together then returns the created data set
def combine_data(files):
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return data

train = combine_data(['train.csv', 'stores.csv', 'oil.csv', 'holidays_events.csv', 'transactions.csv'])
test = combine_data(['test.csv', 'stores.csv', 'oil.csv', 'holidays_events.csv', 'transactions.csv'])
#train.info()

def preprocess(data, cols):
    encode_cols = ['family', 'city', 'state', 'type', 'locale', 'locale_name', 'transactions']
    #checks if there is an array of columns to remove then removes them from dataset
    if cols:
        data = data.drop(cols, axis=1)
    #changes date column to datetime datatype
    data['date'] = data['date'].astype('datetime64[ns]')
    #fills oilprice and holiday type columns with 0 if there are empty values
    data.fillna({'dcoilwtico':0}, inplace=True)
    encode = OneHotEncoder()
    #encodes each of the string or object data type columns, drops the column, then adds the encoded columns to the data set
    for e in encode_cols:
        data.drop(e, axis=1)
        encoded_data = encode.fit(data[e].unique().reshape(-1, 1))
        new_cols = pd.DataFrame(encoded_data, columns=encode.get_feature_names_out(e))
        data = pd.concat([data, new_cols], axis = 1)
    return data
    
train = preprocess(train, ['cluster', 'description', 'transferred'])
#for checking data in data frame
#train.info()
#pd.set_option('display.max_columns', None)
print(train.head())
test = preprocess(test, ['cluster', 'description', 'transferred']) 

#creates a valid test set by splitting training data set.
#The valid dataset contains 30% of the training data set and leaves the training set the remaining 70%.
train, valid = train_test_split(train, test_size=0.3)
