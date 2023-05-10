import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from scipy.optimize import curve_fit
import numpy as np

def load_data():
    """
    This function loads data from an Excel file and returns the dataframe object containing the data.

    Returns:
    df (pandas.DataFrame): Dataframe containing the data.
    df[:] (pandas.DataFrame): A copy of the dataframe containing the data.
    """
    df = pd.read_excel(r'C:\Users\munee\Desktop\SSK assignment\Assignment 3\Assignment 3 Clustering and fitting - poster\API_19_DS2_en_excel_v2_5360124.xls')
    return df, df[:]

def get_encoding(df):
    """
    This function encodes the categorical variables in a given dataframe using LabelEncoder.

    Args:
    df (pandas.DataFrame): Dataframe containing the data.

    Returns:
    df (pandas.DataFrame): Dataframe with encoded categorical variables.
    """

    # Initializing the LabelEncoder object
    le = LabelEncoder()
    
    # Iterating over each column of the dataframe and encoding the categorical variables using LabelEncoder
    for i in df.columns:
        df[i] = le.fit_transform(df[i])

    return df     


def get_scaling(df):
    """
    This function scales the numerical variables in a given dataframe using MinMaxScaler.

    Args:
    df (pandas.DataFrame): Dataframe containing the data.

    Returns:
    df (pandas.DataFrame): Dataframe with scaled numerical variables.
    """
    # Initializing the MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Scaling the data in the dataframe and storing it in a new dataframe 'df_n'
    df_n = scaler.fit_transform(df)
    
    # Creating a new dataframe 'df' from the scaled data and assigning the original column names
    df = pd.DataFrame(df_n, columns= df.columns)
    return df

def get_best_cluster_no(df):
    """
    This function finds the best number of clusters for KMeans clustering using the elbow method.

    Args:
    df (pandas.DataFrame): Dataframe containing the data.

    Returns:
    sse (dict): A dictionary containing the sum of squared errors (SSE) for each number of clusters from 1 to 9.
    """
    # Initializing an empty dictionary to store the SSE values for each value of k
    sse = {}
    
    # Looping over each value of k from 1 to 9 and performing KMeans clustering for each value of k
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df)
        sse[k] = kmeans.inertia_
        
    return sse

def plot_elbow(sse):
    """
    This function plots the sum of squared errors (SSE) vs. number of clusters graph for KMeans clustering.

    Args:
    sse (dict): A dictionary containing the sum of squared errors (SSE) for each number of clusters from 1 to 9.
    """
    # Creating a new figure for the plot
    plt.figure()
    
    # Plotting the SSE values for each value of k
    plt.plot(list(sse.keys()), list(sse.values()))
    
    # Setting the label for x-axis
    plt.xlabel("Number of cluster")
    
    # Setting the label for y-axis
    plt.ylabel("SSE")
    
    # Saving the plot as a PNG file
    plt.savefig('elbow_method.png',dpi=300)

def get_Kmean_fit(df,cluster):
    """
    This function performs KMeans clustering on a given dataframe with a specified number of clusters.

    Args:
    df (pandas.DataFrame): Dataframe containing the data.
    cluster (int): Number of clusters for KMeans clustering.

    Returns:
    kmn (sklearn.cluster.KMeans): Fitted KMeans clustering model.
    """
    # Initializing the KMeans object with the specified number of clusters
    kmn = KMeans(n_clusters=cluster)
    
    # Performing KMeans clustering on the input dataframe
    kmn.fit(df)
    
    return kmn


def get_kmean_label_center(df, kmn):
    """
    Assigns cluster labels to each observation in a given dataframe based on KMeans clustering and returns the cluster labels 
    and the cluster centers.

    Args:
    df (pandas.DataFrame): Dataframe containing the data.
    kmn (sklearn.cluster.KMeans): Fitted KMeans clustering model.

    Returns:
    label (numpy.ndarray): Array containing the cluster labels for each observation.
    centers (numpy.ndarray): Array containing the coordinates of the cluster centers.
    """

    # Obtaining the cluster labels and centers from the fitted KMeans object
    label = kmn.predict(df)
    centers = kmn.cluster_centers_
    
    
    return label, centers


def get_PCA(df):
    """
    Performs Principal Component Analysis (PCA) on a given dataframe and returns the transformed data.

    Args:
    df (pandas.DataFrame): Dataframe containing the data.

    Returns:
    X_pca (numpy.ndarray): Transformed data after PCA.
    """
    # Initializing the PCA object with 2 components
    pca = PCA(2)
    
    # Applying PCA to the input dataframe and returning the transformed data
    return pca.fit_transform(df)


def merge_PCA_Data(df, X_pca):
    """
    Merges the transformed data obtained from PCA with the original dataframe.

    Args:
    df (pandas.DataFrame): Dataframe containing the data.
    X_pca (numpy.ndarray): Transformed data after PCA.

    Returns:
    df (pandas.DataFrame): Dataframe containing the merged data.
    """
    
    pca_df = pd.DataFrame(X_pca, columns=['pca1', 'pca2'])
    
    # Concatenating the input dataframe and the PCA dataframe 'pca_df' along axis 1 and returning the result
    return pd.concat([df, pca_df], axis=1)


def get_year_indicator(data):
    """
    Extracts the indicator values for a given country and returns the years and the indicator values as arrays.

    Args:
    data (pandas.DataFrame): Dataframe containing the data for a single country.

    Returns:
    years (numpy.ndarray): Array containing the years for which the indicator values are available.
    indicator_values (numpy.ndarray): Array containing the indicator values for the chosen country.
    """

    # Extracting the years from the dataframe column names and storing them as an integer numpy array 'years'
    years = np.array(data.columns[4:65], dtype=int)
    
    # Extracting the indicator values for the first country (row 0) from the dataframe and storing them as a float numpy array 'indicator_values'
    indicator_values = data.loc[0, '1960':'2020'].values.astype(float)
    
    # Returning the years and indicator values as a tuple
    return years, indicator_values

def linear_model(x, a, b):
    """
    Computes the linear regression model y = ax + b for a given set of input values.

    Args:
    x (numpy.ndarray): Array of input values.
    a (float): Slope of the linear regression line.
    b (float): Intercept of the linear regression line.

    Returns:
    y (numpy.ndarray): Array of predicted y values.
    """
    return a * x + b


def err_ranges(x, popt, pcov, model_func):
    """
    Calculates the confidence interval for the predicted values of a given model.

    Args:
    x (numpy.ndarray): Array of input values.
    popt (numpy.ndarray): Optimal values for the parameters of the model.
    pcov (numpy.ndarray): Estimated covariance of the model parameters.
    model_func (function): Function defining the model.

    Returns:
    lower (numpy.ndarray): Array of lower confidence interval values for the predicted y values.
    upper (numpy.ndarray): Array of upper confidence interval values for the predicted y values.
    """
    # Using the fitted model function and fitting parameters to predict the output values
    y_pred = model_func(x, *popt)
    
    # Calculating the standard deviation of the fitting parameters
    sigma = np.sqrt(np.diag(pcov))
    
    # Calculating the partial derivatives
    J = np.column_stack((x, np.ones(len(x))))
    
    # Calculating the covariance matrix of the predicted values
    pred_cov = np.dot(J, np.dot(pcov, J.T))
    
    # Calculating the confidence interval using 1.96 as the z-score
    conf = 1.96 * np.sqrt(np.diag(pred_cov))
    upper = y_pred + conf
    lower = y_pred - conf
    
    # Returning the lower and upper bounds of the confidence interval
    return lower, upper


def select_countires(kmn, data):
    """
    Selects a random country from each cluster in the given data based on the cluster labels obtained from KMeans clustering.

    Args:
    kmn (sklearn.cluster.KMeans): Fitted KMeans clustering model.
    data (pandas.DataFrame): Dataframe containing the data.

    Returns:
    selected_countries (list): List containing the selected countries.
    """
    # Initializing an empty list to store the selected countries
    selected_countries = []
    
    # Looping over each cluster and selecting a random country from the cluster
    for cluster in range(kmn.n_clusters):
        c = data[data['ClusterNo'] == cluster]
        ind = random.choice(c.index)
        selected_countries.append(data[data['ClusterNo'] == cluster].loc[ind])
    
    # Returning the selected countries as a list of dataframes
    return selected_countries


def plot_year_ondicator(years, indicator_values,future_years,predicted_values,lower, upper):
    """
    Plots a graph showing the historical data and the predicted values for a given indicator.

    Args:
    years (numpy.ndarray): Array of years for the historical data.
    indicator_values (numpy.ndarray): Array of historical values for the indicator.
    future_years (numpy.ndarray): Array of years for the predicted values.
    predicted_values (numpy.ndarray): Array of predicted values for the indicator.
    lower (numpy.ndarray): Array of lower confidence interval values for the predicted values.
    upper (numpy.ndarray): Array of upper confidence interval values for the predicted values.
    """
    # Creating a scatter plot of the input data
    plt.plot(years, indicator_values, 'bo', label='Data')
    
    # Creating a line plot of the predicted values
    plt.plot(future_years, predicted_values, 'r-', label='Fit')
    
    # Creating a shaded region between the lower and upper bounds of the confidence interval
    plt.fill_between(future_years, lower, upper, color='gray', alpha=0.5, label='Confidence interval')
    
    # Setting the label for the x-axis
    plt.xlabel('Year')
    
    # Setting the label for the y-axis
    plt.ylabel('Indicator Value')
    
    # Adding a legend to the plot
    plt.legend()
    
    # Setting the title of the plot
    plt.title('Year indicator Analysis')
    
    # Saving the plot as a PNG file
    plt.savefig('year_indicator.png',dpi=300)
    
    # Clearing the current figure
    plt.clf()


def visualize_clusters(cluster_column,all_data) :
    """
    Creates a scatter plot to visualize the clusters.

    Args:
    cluster_column (str): Name of the column containing the cluster numbers.
    all_data (pandas.DataFrame): Dataframe containing all the data and the cluster numbers.

    Returns:
    None
    """
    # Clearing the current figure
    plt.clf()
    
    # Extracting the PCA components from the input dataframe
    for_x = all_data['pca1']
    for_y = all_data['pca2']
    
    # Creating a scatter plot of the PCA components with different colors for different clusters
    sns.scatterplot(x=for_x, y=for_y, hue=cluster_column, 
                data=all_data, s=20)
    
    # Adding a legend to the plot
    plt.legend(loc='lower right')
    
    # Saving the plot as a PNG file
    plt.savefig('Clusters_visulization.png',dpi=300)
    
    # Setting the title of the plot
    plt.title('Clusters Visulization')
    
    # Clearing the current figure
    plt.clf()
    

def plot_centers(all_data,label,centers):
    """
    Creates a scatter plot to visualize the cluster centers.

    Args:
    all_data (pandas.DataFrame): Dataframe containing all the data and the cluster numbers.
    label (numpy.ndarray): Array of cluster labels.
    centers (numpy.ndarray): Array of cluster centers.

    Returns:
    None
    """
    X = all_data[['pca1','pca2']].values
    # Extracting the PCA components from the input data and plotting them with different colors for different clusters
    plt.scatter(X[:, 0], X[:, 1], c=label)
    
    # Adding markers for the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, linewidths=3, color='r')
    
    # Saving the plot as a PNG file
    plt.savefig('plot_centers.png',dpi=300)
    
    # Setting the title of the plot
    plt.title('Plot Cluster centers')
    
    # Clearing the current figure
    plt.clf()


def plot_trends(data,selected_countries):
    """
    Plot trends for selected countries.

    Args:
        data (pd.DataFrame): The dataframe containing data for all countries.
        selected_countries (list): List of countries for which trends are to be plotted.

    Returns:
        None

    Raises:
        None
    """
    # Setting the label for the indicator
    indicator = 'Indicators'
    
    # Extracting years from 1960 to 2021
    years = np.array(data.columns[4:65], dtype=int)
    
    # Plotting trends for each selected country
    for country in selected_countries:
        
        # Extracting the indicator values for the country and fitting a linear model
        indicator_values = country.loc['1960':'2020'].values.astype(float)
        popt, pcov = curve_fit(linear_model, years, indicator_values)
        
        # Predicting the indicator values for future years (2022-2041)
        future_years = np.array(range(2022, 2042))
        predicted_values = linear_model(future_years, *popt)
        
        # Calculating the lower and upper bounds of the confidence interval for the predicted values
        lower, upper = err_ranges(future_years, popt, pcov, linear_model)

        # Plotting the indicator values for the country with a label and filling the area between the lower and upper bounds of the confidence interval
        plt.plot(years, indicator_values, label=f"{country['Country Name']} ({country['ClusterNo']})")
        plt.fill_between(future_years, lower, upper, alpha=0.2)

    # Setting the label for the x-axis
    plt.xlabel('Year')
    
    # Setting the label for the y-axis
    plt.ylabel(indicator)
    
    # Adding a legend to the plot
    plt.legend()
    
    # Setting the title of the plot
    plt.title('Trends Analysis Between Countries')
    
    # Saving the plot as a PNG file
    plt.savefig('trend_analysis.png',dpi=300)
    
    # Clearing the current figure
    plt.clf()

def main_app():

    # Load the data and preprocess it
    df, dfx = load_data()   # Load the dataset
    df = get_encoding(df)   # One-hot encode categorical variables
    df = get_scaling(df)    # Scale the numerical variables
    sse = get_best_cluster_no(df)  # Determine the optimal number of clusters using the elbow method

    # Visualize the elbow plot
    plot_elbow(sse)

    # Perform K-means clustering on the preprocessed data
    kmn = get_Kmean_fit(df,3)   # Get a K-means model with 3 clusters
    label , centers = get_kmean_label_center(df,kmn)   # Get the labels and centers of the clusters
    df['ClusterNo'] = label   # Add the cluster labels to the original dataframe

    # Perform PCA on the preprocessed data
    X_pca = get_PCA(df)   # Get the principal components
    all_data = merge_PCA_Data(df,X_pca)   # Merge the principal components with the original dataframe

    # Visualize the clusters in a scatterplot
    visualize_clusters('ClusterNo',all_data)

    # Visualize the cluster centers in a scatterplot
    plot_centers(all_data,label,centers)

    # Analyze the indicator values over the years
    data = df[:]
    years, indicator_values = get_year_indicator(data)   # Get the indicator values for each year
    popt, pcov = curve_fit(linear_model, years, indicator_values)   # Perform curve fitting to get a linear model
    future_years = np.array(range(2022, 2042))
    predicted_values = linear_model(future_years, *popt)   # Predict the indicator values for future years
    lower, upper = err_ranges(future_years, popt, pcov, linear_model)   # Get the confidence interval for the predictions
    plot_year_ondicator(years, indicator_values, future_years, predicted_values, lower, upper)   # Visualize the indicator values and predictions

    # Select random countries from each cluster and analyze their indicator value trends
    selected_countries = select_countires(kmn, data)   # Select random countries from each cluster
    plot_trends(data, selected_countries)   # Visualize the trends in a line chart

if __name__=="__main__":
    main_app()