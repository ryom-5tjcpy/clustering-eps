import pandas as pd
import plotly.express as px

def main():
    df = pd.read_csv("clustered_data.csv")

    fig = px.scatter_3d(
        df,
        x='i',
        y='j',
        z='k',
        color='eps',
        symbol='cluster',
        range_color=[1500, 52000]
    )
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            title_text='Cluster',
        )
    )
    fig.update_traces(marker=dict(cmin=1500, cmax=52000))

    fig.write_html("dist.html")

if __name__ == '__main__':
    main()