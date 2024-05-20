import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sb
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#reads files, changes date column values to date-time format, and creates data frames
def read_file(file):
    data = pd.read_csv(file)
    if 'date' in data.columns: 
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data
train = read_file('train.csv')
stores = read_file('stores.csv')
oil = read_file('oil.csv')
holidays = read_file('holidays_events.csv')
transactions = read_file('transactions.csv')

#remove zeros from stores dataframe and then combine train and stores data frames
zeros = train.groupby(['id', 'store_nbr', 'family']).sales.sum().reset_index().sort_values(['family','store_nbr'])
zeros = zeros[zeros.sales == 0]
join = train.merge(zeros[zeros.sales == 0].drop("sales",axis = 1), how='outer', indicator=True)
train = join[~(join._merge == 'both')].drop(['id', '_merge'], axis = 1).reset_index()
train = train.drop(['index'], axis=1)
#adding number of transactions to train data set by date and store_nbr
train = pd.merge(train, transactions, on=['date', 'store_nbr'])
#adding oil prices by date and then removing days with no oil prices reported
train = pd.merge(train, oil, on=['date'])
train = train[train['dcoilwtico'].notna()]

#dropping description and transferred columns from holidays data frame
#adding the holidays columns to the train data frame, filling the NaN values
holidays = holidays.drop(['description', 'transferred', 'locale', 'locale_name'], axis=1)
train = pd.merge(train, holidays, on=['date'], how='left')
train['type'] = train['type'].fillna(value='No Event or Holiday')
#train['locale'] = train['locale'].fillna(value='National')
#train['locale_name'] = train['locale_name'].fillna(value='Ecuador')

#data vizualization and stats

#shows gas prices by date
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25,15))
oil.plot.line(x="date", y="dcoilwtico", color='g', title ="Oil Prices from 2013 to 2017", ax = axes, rot=0)
plt.show()

#shows total sales by family
temp = train.groupby('family').sum('sales').reset_index().sort_values(by='sales', ascending=False)
temp = temp[['family','sales']]
temp['percent']=(temp['sales']/temp['sales'].sum())
temp['percent'] = temp['percent'].apply(lambda x: f'{x:.0%}')
temp['cumulative']=(temp['sales']/temp['sales'].sum()).cumsum()
temp['cumulative'] = temp['cumulative'].apply(lambda x: f'{x:.0%}')
fig1 = px.bar(temp, x="family",y="sales",title = "Total Sales by Family Type",text="cumulative")
fig1.show()

#shows amount of total sales by holiday type
temp = train.groupby('type').sum('sales').reset_index().sort_values(by='sales', ascending=False)
temp = temp[['type','sales']]
temp['percent']=(temp['sales']/temp['sales'].sum())
temp['percent'] = temp['percent'].apply(lambda x: f'{x:.0%}')
temp['cumulative']=(temp['sales']/temp['sales'].sum()).cumsum()
temp['cumulative'] = temp['cumulative'].apply(lambda x: f'{x:.0%}')
fig1 = px.bar(temp, x="type",y="sales",title = "Total Sales by Holiday Type",text="cumulative")
fig1.show()
#encoding object dtype columns
encoder = OneHotEncoder()
#reduce number of columns to encode
train['family'] = train['family'].replace(['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BOOKS', 'BREAD/BAKERY',
       'CELEBRATION', 'DELI', 'FROZEN FOODS', 'GROCERY II',
       'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II',
       'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN',
       'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES','PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'PREPARED FOODS', 'SCHOOL AND OFFICE SUPPLIES',
       'SEAFOOD'],'OTHERS')
newtrain = train.groupby(['date', 'family']).sum('sales').reset_index()
cols_encode=['family','type']
for col in cols_encode:
  encoded = encoder.fit_transform(train[[col]])
  feature_names = encoder.categories_[0]
  onehot_df = pd.DataFrame(encoded.toarray(), columns=feature_names)
  train= pd.concat([train, onehot_df], axis=1)
train.drop(cols_encode, axis=1, inplace=True)
#splitting train data frame into test, train, and valid data frames by date
test = train.loc[train['date'] > pd.to_datetime('2016-12-31')]
valid = train.loc[train['date'] < pd.to_datetime('2014-01-01')]
train = train.loc[(train['date'] >= pd.to_datetime('2014-01-01')) & (train['date'] <= pd.to_datetime('2016-12-31'))]
#train.info()
#train.head()

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

#Alex: Misread a communication, thought I was doing this
#TODO: Did not test this, but VSCode (because PyCharm is annoying enough, and I don't want to add GitHub on top of that)
def random_forest_classifier(train, test, valid):
    #NOTE: actual printf if possible, instead of this mess-
    ls = ":\t"
    lsa = " Accuracy"+ls
    #Alex: changed word forms because I am (inconsistently) pedantic about grammar
    prints = ["Training","Testing","Validiation"]
    header = "\n-- Random Forest Classifier --"
    #Alex: So I can use one statement
    print_super = ["Maximum Tree Depth"+ls, "\r\nMinimum Leaf Node Samples"+ls]
    #Basing the possible tests off of IP2.
    depths = [3,5]
    leaves = [5,10]
    cols = ['state']
    #Import statements consisent with previous ones
    #var name changes subjective, 
    
    #deep copy might be safer, and protects the input datasets 
    X_train, y_train = (copy(train)).drop(columns = cols), copy(train[cols])
    X_test, y_test = (copy(test)).drop(columns = cols), copy(test[cols])
    X_valid, y_valid = (copy(valid)).drop(columns = cols), copy(test[cols])
    #seperate print and definition.
    print(header)
    #iterate over two parameters: max_depth and min_samples_leaf
    for depth_max in depths:
        for leaf_min in leaves:
            #Single combined print        
            print(print_super[0] + str(depth_max) + print_super[1] + str(leaf_min))
            rfs_classifier = RandomForestClassifier(min_samples_leaf=leaf_min,max_depth=depth_max)
            rfs_classifier.fit(X_train,y_train)
            accuracy_train = accuracy_score(y_train, rfs_classifier.predict(X_train))
            accuracy_test = accuracy_score(y_test, rfs_classifier.predict(X_test))
            accuracy_valid = accuracy_score(y_valid, rfs_classifier.predict(X_valid))
            accuracy_values = [accuracy_train, accuracy_test, accuracy_valid]
            #Shouldn't be different, but just in case
            #ended up being longer than individual print statements
            for i in range(0, min(accuracy_values.size, prints.size)):
                print(prints[i] + lsa + str(accuracy_values[i]))       
                #Keep the additional line of spacing at the end
                if(i == min(accuracy_values.size, prints.size) - 1):
                    print("")

#TODO: Ideally, the K-means function should be called here, with modified prints

def logistic_regression_classifier(train,test,valid):
    ls = ":\t"
    lsa = " Accuracy"+ls
    #To start the index from 1
    indents = ["","\t","\t\t","\t\t\t","\t\t\t\t","\t\t\t\t\t"," "]
    #Alex: changed word forms because I am (inconsistently) pedantic about grammar
    prints = ["Training","Testing","Validiation"]
    header = "\n-- Logistic Regression Classifier --"
    #Alex: So I can use one statement
    print_super = ["Solver"+ls, "\r\nPenalty"+ls]
    cols = ['state']
    #Import statements consisent with previous ones
    #var name changes subjective, 
    
    #deep copy might be safer, and protects the input datasets 
    X_train, y_train = (copy(train)).drop(columns = cols), copy(train[cols])
    X_test, y_test = (copy(test)).drop(columns = cols), copy(test[cols])
    X_valid, y_valid = (copy(valid)).drop(columns = cols), copy(test[cols])
    #seperate print and definition.
    print(header)    
    K_values = [1, 3, 5, 7]
    
    #Alex: Testing how to get clusters
    #(knn_classifier.get_params()).get("n_clusters  ")
    


#Alex: I have to replicate Maria's function directly, as it does not return anything
#per-cluster
#print data mirrored
#FIXME: not a good idea to have two nearly identical functions
#TODO: We should figure out how to merge this with K-means
#NOTE: half the arguments are just to pass print strings,  so they only need to be defined once
def obj_KMeans(train, test, valid,K_list,ls,lsa,prints,indent):
    ls = ":"+indent[1]
    #single space
    lsa = indent[-1]+"Accuracy"+ls
    #Alex: changed word forms because I am (inconsistently) pedantic about grammar
    prints = ["Training","Testing","Validiation"]
    header = "\r\n\t-- K-Means INSTANCE --"
    #Alex: So I can use one statement
    neighbor_text = indent[2]+"Neighbors"+ls
    #Basing the possible tests off of IP2.
    depths = [3,5]
    leaves = [5,10]
    cols = ['state']
    #Import statements consisent with previous ones
    #var name changes subjective, 
    
    #deep copy might be safer, and protects the input datasets 
    X_train, y_train = (copy(train)).drop(cdolumns = cols), copy(train[cols])
    X_test, y_test = (copy(test)).drop(columns = cols), copy(test[cols])
    X_valid, y_valid = (copy(valid)).drop(columns = cols), copy(test[cols])
    #seperate print and definition.
    
    for neighbor in K_list:
        print(neighbor_text+neighbor)
        #random solution, 
        KNN_instance = K
        cluster_map = pd.DataFrame()
        cluster_map['data_indicies'] = train.index.values
        cluster_map['cluster'] = km.labels_
        ##use indent to predefine the indents
        
        
        
    
    



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
