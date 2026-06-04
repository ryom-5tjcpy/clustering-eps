import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_excel("sample_data.xlsx")
    #print(df.head())
    n_clusters = 6
    discrete_cmap = plt.get_cmap('gist_rainbow', 6)

    x = df[['s2', 'Ω', 'so', 'eps']]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(x_scaled)
    df['cluster'] = model.predict(x_scaled)

    for i in range(n_clusters):
        df1 = df[df['cluster'] == i]
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        ax.scatter(df1['i'], df1['j'], df1['k'], color=discrete_cmap(i))

        fig.savefig(f'dist{i}.png')

if __name__ == "__main__":
    main()
