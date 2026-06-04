import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_excel("sample_data.xlsx")
    #print(df.head())
    n_clusters = 6

    x = df[['s2', 'Ω', 'so', 'eps']]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(x_scaled)
    df['cluster'] = model.predict(x_scaled)

    target = ['s2', 'Ω', 'so']
    max = [90, 220, 90]
    #colors = ['r', 'b', 'g', 'c', 'm', 'y']
    discrete_cmap = plt.get_cmap('viridis', 6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(n_clusters):
        df1 = df[df['cluster'] == i]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for j, col in enumerate(target):
            axes[j].scatter(df1[col], df1['eps'], color=discrete_cmap(i))
            axes[j].set_xlabel(col)
            axes[j].set_ylabel('eps' if i == 0 else '')
            axes[j].set_title(f'K-Means Clustering - {col} cluster {i}')
            axes[j].set_xlim(0, max[j])
            axes[j].set_ylim(0, 53000)
        fig.savefig(f'cluster{i}.png')



    # Plot the clusters
    for i, col in enumerate(target):
        axes[i].scatter(df[col], df['eps'], c=df['cluster'], cmap='gist_rainbow')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('eps' if i == 0 else '')
        axes[i].set_title(f'K-Means Clustering - {col}')
        axes[i].set_xlim(0, max[i])
        axes[i].set_ylim(0, 53000)
    fig.savefig('kmeans_clustering.png')

    print(df.head())
    df.to_csv('clustered_data.csv', index=False)

if __name__ == "__main__":
    main()
