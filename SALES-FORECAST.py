
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split


# Reads the files
def read_file(file):
    data = pd.read_csv(file)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data

# Base directory where files are stored
base_dir = '/Users/friedrice/Desktop/IS427/Project/store-sales-time-series-forecasting/'
train = read_file(base_dir + 'train.csv')
test = read_file(base_dir + 'test.csv')
holidays = read_file(base_dir + 'holidays_events.csv')
oil = read_file(base_dir + 'oil.csv')
stores = read_file(base_dir + 'stores.csv')
transactions = read_file(base_dir + 'transactions.csv')

# merges transactions dataset with stores csv using store nbr as the key
transactions_csv = transactions.merge(stores, on='store_nbr')
# merges train dataset with stores csv using store nbr as the key
train_csv = train.merge(stores, on='store_nbr')

# groups the transactions csv by store nbr and calculates the mean number of transaction for each store
store_performance = transactions.groupby('store_nbr').agg({'transactions': 'mean'}).reset_index()
# Calculates the average sale price per store by grouping the merged sales csv by store nbr
store_sales = train_csv.groupby('store_nbr').agg({'sales': 'mean'}).reset_index()
# merges the store performance dataset with the store sales dataset using store nbr as the key
store_performance = store_performance.merge(store_sales, on='store_nbr')
# merging additional store details from the stores csv
store_performance = store_performance.merge(stores[['store_nbr', 'state']], on='store_nbr')
# calculates the average sales by state
state_average_sales = store_performance.groupby('state')['sales'].mean().reset_index()
# merging state average sales with store performance dataset
store_performance = store_performance.merge(state_average_sales, on='state', suffixes=('', '_state_avg'))
# adding a new column called performance classifying stores as high or low based on the average sales
store_performance['performance'] = np.where(store_performance['sales'] > store_performance['sales_state_avg'], 'High', 'Low')

# function for log regression
def logistic_regression(data):
    # Extracting 2 columns from transactions and sales
    features = data[['transactions', 'sales']]
    # Extracting the performance column, will be used later to output the classification report
    labels = data['performance']
    # using the test, train, split function to randomly split features and labels into training and testing sets
    # 20% of the data will be used for testing and 80% for training
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # initializing the logistic regression model
    logistic_model = LogisticRegression()
    # fitting the model with the training data
    logistic_model.fit(X_train, y_train)
    # model predictions
    predictions = logistic_model.predict(X_test)
    # prints the key metrics such as precison, recall, f1-score, support, accuracy
    print(classification_report(y_test, predictions))
    # Setting up the visual for the graph
    plt.figure(figsize=(12, 7))
    # using the performance column to color the points on the graph
    plot_axis = sns.scatterplot(x='transactions', y='sales', hue='performance', data=data, style='performance', palette='coolwarm')
    # setting the x and y axis limits
    x_limiter = np.array(plot_axis.get_xlim())
    # calculates the x and y limits to plot the red boundary line
    # Separates the feature space into regions with different classification labels
    # logistic_model.coef_[0][0] is the coefficient for transactions
    # and logistic_model.coef_[0][1] is the coefficient for sales
    y_limiter = -(x_limiter * logistic_model.coef_[0][0] + logistic_model.intercept_) / logistic_model.coef_[0][1]
    plt.plot(x_limiter, y_limiter, '--', color='red')
    # prints the graphs information
    plt.title('Store Performance by State Based on Transactions')
    plt.xlabel('Average Transactions')
    plt.ylabel('Average Sales')
    plt.legend(title='Performance')
    plt.show()

# k means clustering function
def kmeans_clustering(data):

    # Extracting 2 columns from transactions and sales
    features = data[['transactions', 'sales']]
    # initializing the kmeans model with 10 clusters
    kmeans = KMeans(n_clusters=10, random_state=0)
    # fits the kmeans model to features
    clusters = kmeans.fit_predict(features)
    # calculates the silhouette score
    # The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters
    # A high silhouette score indicates better defined clusters
    # typically you want a score greater than 0.5
    score = silhouette_score(features, clusters)
    print(f"Silhouette Score: {score:.2f}")
    # adding the cluster column to the data
    data['cluster'] = clusters
    # merging the store name, city, and state to the data
    data = data.merge(stores[['store_nbr', 'city', 'state']], on='store_nbr')
    # creating a new column called store name which is a combination of city and store number
    data['store_name'] = data['city'] + " " + data['store_nbr'].astype(str)
    plt.figure(figsize=(20, 10))
    color = sns.color_palette('viridis', n_colors=10)
    # plotting the clusters
    for cluster in sorted(data['cluster'].unique()):
        # filtering the data by cluster
        clustered_data = data[data['cluster'] == cluster]
        # plotting the scatter plot
        plt.scatter(clustered_data['store_nbr'], clustered_data['transactions'], color=color[cluster], label=f'Cluster {cluster}')
        # annotating the store name on the graph
        for _, row in clustered_data.iterrows():
            plt.annotate(row['store_name'], (row['store_nbr'], row['transactions']), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('K-Means Clustering of Stores Based on Average Sales and Transactions')
    plt.xlabel('Store Number')
    plt.ylabel('Average Sales')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()


logistic_regression(store_performance)
kmeans_clustering(store_performance)
