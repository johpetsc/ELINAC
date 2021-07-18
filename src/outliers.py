import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
pd.options.mode.chained_assignment = None

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

def limit_outliers(df):
    Q1, Q3 = np.percentile(df['Quantity'], [20, 80])
    IQR = Q3 - Q1
    x_upper_limit = Q3 + 1.5 * IQR
    x_lower_limit = Q1 - 1.5 * IQR

    Q1, Q3 = np.percentile(df['Price'], [25, 75])
    IQR = Q3 - Q1
    y_upper_limit = Q3 + 1.5 * IQR
    y_lower_limit = Q1 - 1.5 * IQR
    
    x_axis = df['Quantity'].mean()
    y_axis = df['Price'].mean()

    x_threshold = max(1,5 * x_axis, x_upper_limit)
    y_threshold = max(1,5 * y_axis, y_upper_limit)

    center = np.array([x_axis, y_axis]).reshape(1, -1)

    df.loc[(df['Quantity'] > x_threshold), 'Outlier'] = 1
    df.loc[(df['Price'] > y_threshold), 'Outlier'] = 1

    #plot_outliers(df.loc[df['Outlier'] == 1], df.loc[df['Outlier'] == 0], center, df)
    #print(df)
    
    return df

def distance_outliers(df):
    x_axis = df['Quantity'].mean()
    y_axis = df['Price'].mean()

    center = np.array([x_axis, y_axis]).reshape(1, -1)

    distances = cdist(center, df[['Quantity', 'Price']], 'seuclidean')
    df['Distance'] = np.transpose(distances)

    outliers = df[df['Distance'] >= np.percentile(df['Distance'], 95)]
    df.loc[df['Distance'] >= np.percentile(df['Distance'], 95), 'Outlier'] = 1
    inliers = df[df['Distance'] < np.percentile(df['Distance'], 95)]
    print(outliers)
    df = df.drop(columns='Distance')
    plot_outliers(outliers, inliers, center, df)

    
    return df

def main():
    data = pd.read_excel('../dataset/output.xlsx')
    data['Outlier'] = 0
    for x in range (0, 2400):
        cluster = data[(data['Cluster'] == x)]
        print("Cluster: ", x)
        if(cluster.shape[0] > 1):
            df = limit_outliers(cluster)

            data[data['Cluster'] == x] = df

        elif(cluster.shape[0] == 1):
            cluster['Outlier'] = 1

            data[data['Cluster'] == x] = cluster

    data.to_excel("../dataset/outliers.xlsx", index=False)

if __name__ == '__main__':
    main()