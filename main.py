import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data import load_data


UTILITY = "VITENS"

areas = []
if UTILITY == 'VITENS':
    areas = list(load_data.VITENS_KEYS)
    dbm = dict(
        forecast = pd.read_parquet(r'data\vitens\forecast.parquet'),
        hihi = pd.read_parquet(r'data\vitens\hihi.parquet'),
        lolo = pd.read_parquet(r'data\vitens\lolo.parquet')
    )
    dbm_areas = list(dbm['forecast'].columns)


# Set up the Streamlit app
st.set_page_config(page_title='HULO')#, layout='wide')
st.title('{}'.format(UTILITY))
st.sidebar.title('Controls')
st.columns(1)


def draw_dbm(area, data):
    # for key, value in dbm.items():
    #     value.index = pd.DatetimeIndex(value['timestamp'])
    #     value.index.tz_localize(None)
    #     value.drop(columns=['timestamp'], inplace=True)
    #     value.columns = [key]
    dbm_area = load_data.VITENS_KEYS[area]['dbm']
    # events = data.loc[(data>) &]
    fig = go.Figure([
        go.Scatter(
            x=data.index.to_numpy(), 
            y=data.to_numpy(), 
            name="Measured",
            line=dict(color='rgba(100,255,100,0.9)'),
            mode='lines'
        ),
        go.Scatter(
            x=dbm['forecast']['timestamp'].to_numpy(), 
            y=dbm['forecast'][dbm_area].to_numpy(), 
            name="DBM forecast",
            line=dict(color='rgba(100,150,255,0.8)'),
            mode='lines'
        ),
        go.Scatter(
            x=dbm['lolo']['timestamp'].to_numpy(), 
            y=dbm['lolo'][dbm_area].to_numpy(), 
            name="DBM low",
            line=dict(color='rgba(50,150,255,0.2)'),
            mode='lines'
        ),
        go.Scatter(
            x=dbm['hihi']['timestamp'].to_numpy(), 
            y=dbm['hihi'][dbm_area].to_numpy(), 
            name="DBM high",
            line=dict(color='rgba(50,150,255,0.2)'),
            mode='lines'
        )
    ])
    return fig

# Define the function to create the four plots
def create_plots(area):
    # Generate some random data
    data = pd.DataFrame(np.random.randn(num_points, 2), columns=['x', 'y'])

    # Create the four plots using Plotly
    # fig1 = px.line(dbm['forecast'], x='timestamp', y='{}'.format(area), title='DBM')
    fig2 = px.histogram(data, x='x', nbins=20, title='Plot 2')
    fig3 = px.box(data, x='x', y='y', title='Plot 3')
    fig4 = px.scatter(data, x='x', y='y', color='x', title='Plot 4')

    return fig2, fig3, fig4

with st.sidebar:
    # Add some widgets to the sidebar
    num_points = st.slider('Number of points', 10, 1000, 50)
    color_map = st.selectbox('Color map', ['viridis', 'plasma', 'inferno'])
    selected_area = st.selectbox(('Choose an area'), areas, index=3)
# Create the four plots
fig2, fig3, fig4 = create_plots('')

flow, pressure = load_data.load_data_utility(UTILITY, area=selected_area)
flow['waterbalance'] = load_data.apply_waterbalance_to_dataset(flow, load_data.calculate_waterbalance_vitens(selected_area))
fig1 = draw_dbm(selected_area, flow['waterbalance'])
# Display the plots in the app
st.plotly_chart(fig1, use_container_width =True)
st.plotly_chart(fig2, use_container_width =True)
st.plotly_chart(fig3, use_container_width =True)
st.plotly_chart(fig4, use_container_width =True)
