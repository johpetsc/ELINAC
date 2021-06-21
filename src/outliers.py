import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
pd.options.mode.chained_assignment = None

def generate_data(cluster):
    df = pd.DataFrame(columns=['val1', 'val2'])
    df['val1']  = cluster['Quantity']
    df['val2'] = cluster['Price']
    #print(cluster)

    return df, cluster

def plot_outliers(outliers, inliers, center, df):
    plt.scatter(inliers['Quantity'] , inliers['Price'], label='Inliers')
    plt.scatter(outliers['Quantity'] , outliers['Price'], s=60, color='red', marker='x', label='Outliers')
    plt.scatter(center[:,0] , center[:,1] , s = 80, color='black', marker='^', label='Center')
    plt.ylabel('Price', fontsize=10)
    plt.xlabel('Quantity', fontsize=10)
    plt.title('Cluster ' + str(df['Cluster'].iloc[0]))
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend(loc="upper right")
    plt.show()

def K_means(X, df):
    kmeans = KMeans(n_clusters=1, random_state=0)
    model = kmeans.fit(X)
    center = kmeans.cluster_centers_
    distances = cdist(center, X, 'seuclidean')
    df['Distance'] = np.transpose(distances)
    #outliers = df[df['Distance'] >= np.percentile(df['Distance'], 95)]
    df.loc[df['Distance'] >= np.percentile(df['Distance'], 95), 'Outlier'] = 1
    df = df.drop(columns='Distance')
    #inliers = df[df['Distance'] < np.percentile(df['Distance'], 95)]
    #print(outliers)
    #plot_outliers(outliers, inliers, center, df)
    
    return df

def main():
    data = pd.read_excel('../dataset/minioutput.xlsx')
    data['Outlier'] = 0
    for x in range (0, 2399):
        cluster = data[(data['Cluster'] == x)]
        print("Cluster: ", x)
        if(cluster.shape[0] > 0): 
            df, clusterData = generate_data(cluster)

            df = K_means(df, clusterData)

            data[data['Cluster'] == x] = df

    data.to_excel("../dataset/outliers.xlsx")

if __name__ == '__main__':
    main()