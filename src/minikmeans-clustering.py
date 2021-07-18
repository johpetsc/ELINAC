import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def generate_data():
    data = pd.read_excel('../dataset/dataset.xlsx')
    df = pd.DataFrame(columns=['Cluster', 'Text', 'Original', 'Quantity', 'Price'])
    df['Original']  = data['DESC']
    df['Quantity'] = data['QTD']
    df['Price'] = data['VAL']
    df['Text'] = df['Original'].str.upper()
    df['Text'] = df['Text'].replace('\W', ' ', regex=True)
    df['Text'] = df['Text'].replace('( LOT).*', '', regex=True)
    df['Text'] = df['Text'].replace('( LT).*', '', regex=True)
    df['Text'] = df['Text'].replace('ITEM', '', regex=True)
    df['Text'] = df['Text'].replace('^\d+\s+', '', regex=True).str.strip()
    df['Text'] = df['Text'].replace('   ', ' ', regex=True)
    df['Text'] = df['Text'].replace('  ', ' ', regex=True)
    df['Text'] = df['Text'].replace('^\d+\s+', '', regex=True).str.strip()
    df['Text'] = df['Text'].replace('^\d+\s+', '', regex=True).str.strip()
    print(df)

    return df

def K_means(X, df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    kmeans = MiniBatchKMeans(n_clusters=1564, random_state=0, verbose=1, batch_size=128)
    model = kmeans.fit(X)
    pred_classes = kmeans.predict(X)
    X = X.todense()

    embeddings = TSNE(n_components=2)
    Y = embeddings.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral, c=model.labels_.astype(float))
    plt.show()

    df['Cluster'] = pred_classes
    df.sort_values(by=['Cluster']).reset_index(drop=True)

    df.index.rename('id', inplace=True)
    df.to_excel("../dataset/output_1.xlsx")
    #pd.set_option('display.max_rows', None)
    #print(res.sort_values(by=['cluster']))
    

def main():
    df = generate_data()

    K_means(df['Text'], df)

if __name__ == '__main__':
    main()