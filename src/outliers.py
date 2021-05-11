import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def generate_data(x):
    data = pd.read_excel('../dataset/output.xlsx')
    cluster = data[(data['Cluster'] == x)]
    df = pd.DataFrame(columns=['val1', 'val2'])
    df['val1']  = cluster['Quantity']
    df['val2'] = cluster['Price']
    print(cluster)

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
    outliers = df[df['Distance'] >= np.percentile(df['Distance'], 95)]
    inliers = df[df['Distance'] < np.percentile(df['Distance'], 95)]
    print(outliers)
    plot_outliers(outliers, inliers, center, df)

def main():
    for x in range (8, 9):
        df, data = generate_data(x)

        K_means(df, data)

if __name__ == '__main__':
    main()