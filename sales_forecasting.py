import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
        x = pd.DataFrame(encode.fit_transform(data[[e]]))
        y = encode.get_feature_names([e])
        data = pd.concat([x, y], axis = 1)
    return data

# k-means clutering where clusters data based on state
def k_means_clustering(train, test, valid):
    print("\n-- K Means Clustering --")

    # Gets trainX, trainY, testX, and testY
    # target column is 'state' to cluster based on regional sales trends
    tr_X, tr_y = train.drop(columns=['state']), train['state']
    te_X, te_y = test.drop(columns=['state']), test['state']  
    v_X, v_y = valid.drop(columns=['state']), test['state'] 
    
    # List of k values to test
    k_values = [1, 3, 5, 7]

    # Iterate over each k value
    for k in k_values:
        # Initialize KNN classifier with current k value
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        
        # Fit the model on training data
        knn_classifier.fit(tr_X, tr_y)
        
        # Evaluate the model
        train_accuracy = accuracy_score(tr_y, knn_classifier.predict(tr_X))
        test_accuracy = accuracy_score(te_y, knn_classifier.predict(te_X))
        valid_accuracy = accuracy_score(v_y, knn_classifier.predict(v_X))
        
        # Print the accuracies
        print("K =", k)
        print("Training Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("Valid Accuracy:", valid_accuracy)
        print("")

# main function where all calls are made
def main():
    # combine data
    train = combine_data(['train.csv', 'stores.csv', 'oil.csv', 'holidays_events.csv', 'transactions.csv'])
    test = combine_data(['test.csv', 'stores.csv', 'oil.csv', 'holidays_events.csv', 'transactions.csv'])
    #train.info()

    # preprocess data
    train = preprocess(train, ['cluster', 'description', 'transferred'])
    #for checking data in data frame
    #train.info()
    #pd.set_option('display.max_columns', None)
    print(train.head())
    test = preprocess(test, ['cluster', 'description', 'transferred'])

    #creates a valid test set by splitting training data set.
    #The valid dataset contains 30% of the training data set and leaves the training set the remaining 70%.
    train, valid = train_test_split(train, test_size=0.3)

    # k-means 
    k_means_clustering(train, test, valid); 


if __name__ == '__main__':
    main()