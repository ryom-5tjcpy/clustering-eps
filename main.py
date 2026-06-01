import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_excel("sample_data.xlsx")
    print(df.head())

    x = df[['so', 'eps']]
    x = df[['s2', 'Ω', 'so', 'eps']]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = KMeans(n_clusters=5)
    model.fit(x_scaled)
    df['cluster'] = model.predict(x_scaled)
    print(df['cluster'])

    # Plot the clusters
    plt.scatter(df['so'], df['eps'], c=df['cluster'], cmap='viridis')
    plt.xlabel('so')
    plt.ylabel('eps')
    plt.title('K-Means Clustering')
    plt.savefig('kmeans_clustering.png')

if __name__ == "__main__":
    main()
