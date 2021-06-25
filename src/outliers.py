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

def outliers(df):
    x_axis = df['Quantity'].mean()
    Q1, Q3 = np.percentile(df['Quantity'], [15, 90])
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    df.loc[(df['Quantity'] > upper_limit), 'Outlier'] = 1

    y_axis = df['Price'].mean()
    std = df['Price'].std()
    Q1, Q3 = np.percentile(df['Price'], [25, 75])
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * (std + IQR)/2
    lower_limit = Q1 - 1.5 * IQR
    df.loc[(df['Price'] > upper_limit), 'Outlier'] = 1
    
    center = np.array([x_axis, y_axis]).reshape(1, -1)
    plot_outliers(df.loc[df['Outlier'] == 1], df.loc[df['Outlier'] == 0], center, df)
    
    # distances = cdist(center, df[['Quantity', 'Price']], 'seuclidean')
    # df['Distance'] = np.transpose(distances)
    # print(df)
    # outliers = df[df['Distance'] >= np.percentile(df['Distance'], 95)]
    # df.loc[df['Distance'] >= np.percentile(df['Distance'], 95), 'Outlier'] = 1
    # df = df.drop(columns='Distance')
    # inliers = df[df['Distance'] < np.percentile(df['Distance'], 95)]
    # print(outliers)
    # plot_outliers(outliers, inliers, center, df)
    
    return df

def main():
    data = pd.read_excel('../dataset/output.xlsx')
    data['Outlier'] = 0
    for x in range (20, 30):
        cluster = data[(data['Cluster'] == x)]
        print("Cluster: ", x)
        if(cluster.shape[0] > 0): 

            df = outliers(cluster)

            data[data['Cluster'] == x] = df

    #data.to_excel("../dataset/outliers.xlsx")

if __name__ == '__main__':
    main()