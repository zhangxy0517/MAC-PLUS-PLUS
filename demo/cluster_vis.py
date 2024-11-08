import chart_studio
import pandas as pd
import math
from colormap import rgb2hex

# fill your information here
chart_studio.tools.set_credentials_file(
    username='',
    api_key=''
)
df = pd.read_csv('result/cluster.csv')
df.head()
import chart_studio.plotly as py


def rgb2hex_list(df):
    list = []
    r_list = df['r']
    g_list = df['g']
    b_list = df['b']
    for i in range(0, len(df)):
        list.append(rgb2hex(r_list[i], g_list[i], b_list[i]))
    return list


scatter = dict(
    mode="markers",
    name="y",
    type="scatter3d",
    x=df['x'], y=df['y'], z=df['z'],
    marker=dict(size=2, color=rgb2hex_list(df))  # rgb->hex
)
clusters = dict(
    alphahull=50,
    name="y",
    opacity=0.1,
    type="mesh3d",
    x=df['x'], y=df['y'], z=df['z']
)
layout = dict(
    # font=dict(family="Courier New, monospace",
    #                          size=32,
    #                           color="RebeccaPurple"
    #                           ),
    #title=dict(text='3d point clustering',font=dict(size=20)),
    scene=dict(
        xaxis=dict(zeroline=True, title='X', titlefont=dict(size=32, family='Times New Roman', color='black'), tickfont=dict(size=20, family='Times New Roman', color='black'), tickvals=[0, math.pi, 2*math.pi], ticktext=['0','π','2π'], autorange=False),
        yaxis=dict(zeroline=True, title='Y', titlefont=dict(size=32, family='Times New Roman', color='black'), tickfont=dict(size=20, family='Times New Roman', color='black'), tickvals=[math.pi], ticktext=['π'], autorange=False),
        zaxis=dict(zeroline=True, title='Z', titlefont=dict(size=32, family='Times New Roman', color='black'), tickfont=dict(size=20, family='Times New Roman', color='black'), tickvals=[0, math.pi, 2*math.pi], ticktext=['0','π','2π'], autorange=False),
        xaxis_range=[0, 2*math.pi], yaxis_range=[0, 2*math.pi], zaxis_range=[0, 2*math.pi]
    )
)
fig = dict(data=[scatter, clusters], layout=layout)
# Use py.iplot() for IPython notebook
py.plot(fig, filename='3d point clustering.html')
