import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Reads the files
def read_file(file):
    data = pd.read_csv(file)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data

# Remove columns with all zero values
def remove_zero(data):
    non_zero_columns = data.loc[:, (data != 0).any(axis=0)]
    return non_zero_columns

# Base directory where files are stored
base_dir = '/Users/friedrice/Desktop/IS427/Project/store-sales-time-series-forecasting/'
train = read_file(base_dir + 'train.csv')
test = read_file(base_dir + 'test.csv')
holidays = read_file(base_dir + 'holidays_events.csv')
oil = read_file(base_dir + 'oil.csv')
stores = read_file(base_dir + 'stores.csv')
transactions = read_file(base_dir + 'transactions.csv')

train = remove_zero(train)
print("Removed zeros" + train.head())

print("Test csv" + test.head())
print("stores csv" + stores.head())
print("transactions csv" + transactions.head())

# Merging stores and transactions, oil, and holidays datasets into train dataset using store_nbr and date as keys
train = train.merge(stores, on='store_nbr')
print("stores merged" + train.head())
train = train.merge(transactions, on=['store_nbr', 'date'])
print("transactions merged" + train.head())
train = train.merge(holidays, on='date', how='left')
print("holidays merged" + train.head())
train = train.merge(oil, on='date', how='left')
print("oil merged" + train.head())
# Calculating additional features from the train dataset
# Grouping the merged train dataset by store_nbr and calculating the mean number of transactions for each store
store_performance = train.groupby('store_nbr').agg({'transactions': 'mean'}).reset_index()
# Calculating the average sale price per store by grouping the merged train csv by store_nbr
store_sales = train.groupby('store_nbr').agg({'sales': 'mean'}).reset_index()
# Merging the store performance dataset with the store sales dataset using store_nbr as the key
store_performance = store_performance.merge(store_sales, on='store_nbr')
# Merging additional store details from the stores csv
store_performance = store_performance.merge(stores[['store_nbr', 'state']], on='store_nbr')
# Calculating the average sales by state
state_average_sales = store_performance.groupby('state')['sales'].mean().reset_index()
# Merging state average sales with store performance dataset
store_performance = store_performance.merge(state_average_sales, on='state', suffixes=('', '_state_avg'))
# Adding a new column called performance classifying stores as high or low based on the average sales
store_performance['performance'] = np.where(store_performance['sales'] > store_performance['sales_state_avg'], 'High', 'Low')

print("Store perf" + store_performance.head())
print("Store sales" + store_sales.head())
print("Average sales" + state_average_sales.head())
# K-Means Clustering
def kmeans_clustering(data, n_clusters=10):
    # Extracting 2 columns from the train dataset
    features = data[['transactions', 'sales']]
    # Initializing the kmeans model with 10 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # Fitting the kmeans model to features
    clusters = kmeans.fit_predict(features)
    # Calculating the silhouette score
    score = silhouette_score(features, clusters)
    print(f"Silhouette Score: {score:.2f}")
    # Adding the cluster column to the data
    data['cluster'] = clusters

    # Merging the store name, city, and state to the data
    data = data.merge(stores[['store_nbr', 'city', 'state']], on='store_nbr')
    # Creating a new column called store name which is a combination of city and store number
    data['store_name'] = data['city'] + " " + data['store_nbr'].astype(str)

    # Plotting the clusters
    plt.figure(figsize=(20, 10))
    color = sns.color_palette('viridis', n_colors=n_clusters)
    for cluster in sorted(data['cluster'].unique()):
        # Filtering the data by cluster
        clustered_data = data[data['cluster'] == cluster]
        # Plotting the scatter plot
        plt.scatter(clustered_data['store_nbr'], clustered_data['transactions'], color=color[cluster], label=f'Cluster {cluster}')

        for _, row in clustered_data.iterrows():
            plt.annotate(row['store_name'], (row['store_nbr'], row['transactions']), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('K-Means Clustering of Stores Based on Average Sales and Transactions')
    plt.xlabel('Store Number')
    plt.ylabel('Transactions')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

    return data, kmeans

# Logistic Regression
def logistic_regression(data):
    # Using the cluster labels as an additional feature
    features = data[['transactions', 'sales', 'cluster']]
    labels = data['performance']
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.70, random_state=42)
    # Initializing the logistic regression model
    logistic_model = LogisticRegression()
    # Fitting the model with the training data
    logistic_model.fit(X_train, y_train)
    # Model predictions
    predictions = logistic_model.predict(X_test)
    # Printing the key metrics such as precision, recall, f1-score, support, accuracy
    print(classification_report(y_test, predictions))
    # Setting up the visual for the graph
    plt.figure(figsize=(12, 7))
    # Using the performance column to color the points on the graph
    plot_axis = sns.scatterplot(x='transactions', y='sales', hue='performance', data=data, style='performance', palette='coolwarm')
    # Setting the x and y axis limits
    x_limiter = np.array(plot_axis.get_xlim())
    # Calculates the x and y limits to plot the red boundary line
    y_limiter = -(x_limiter * logistic_model.coef_[0][0] + logistic_model.intercept_) / logistic_model.coef_[0][1]
    plt.plot(x_limiter, y_limiter, '--', color='red')
    # Printing the graph's information
    plt.title('Store Performance by State Based on Transactions')
    plt.xlabel('Average Transactions')
    plt.ylabel('Average Sales')
    plt.legend(title='Performance')
    plt.show()


def sarimax_forecasting(train, families):

    # Setting the store number to 1
    store_nbr = 1

    # Using For loop to iterate through the product families in store 1
    for family in families:
        # Prints the store number and product family
        print(f"Forecasting for store {store_nbr} and product family: {family}")
        # Filtering the train dataset by store number and product family
        filtered_train = train[(train['store_nbr'] == store_nbr) & (train['family'] == family)]
        # Setting the date as the index
        filtered_train.set_index('date', inplace=True)
        # Resampling the sales data by day
        filtered_train = filtered_train['sales'].resample('D').sum()  # Calculates daily aggregation
        # Performs the Augmented Dickey-Fuller test on filtered train data and dropping any nan values
        # It checks if the time series is stationary or not
        result = adfuller(filtered_train.dropna())
        # Prints the ADF Statistic and p-value
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])

        # If the p-value is greater than 0.05, the data is non-stationary
        if result[1] > 0.05:
            filtered_train = filtered_train.diff().dropna()

        # Fitting the SARIMAX model to the filtered train data
        model = SARIMAX(filtered_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)

        # Forecast for the next 30 days
        forecast_steps = 30
        # Forecasting the sales for the next 30 days
        forecast = model_fit.forecast(steps=forecast_steps)
        # Creating a date range for the forecasted dates
        forecast_dates = pd.date_range(start=filtered_train.index[-1], periods=forecast_steps+1, freq='D')[1:]

        # Plotting Historical Sales Trend
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_train.index, filtered_train, label='Historical Daily Sales', color='green')
        plt.title(f'Historical Sales Trend for {family}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Plotting Forecasted Sales
        plt.figure(figsize=(10, 6))
        plt.plot(forecast_dates, forecast, label='Forecasted Sales', color='red')
        plt.title(f'Future Sales Forecast for {family}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

# Product families in store 1
product_families = ['BEVERAGES', 'PRODUCE', 'AUTOMOTIVE', 'BOOKS']

# calling functons to run the code
store_performance, kmeans = kmeans_clustering(store_performance)

logistic_regression(store_performance)

