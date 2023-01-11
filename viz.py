#!/usr/bin/env python
# coding: utf-8


import numpy as np
import streamlit as st
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px

st.set_page_config(layout="wide")
st.title('Vizualizer')
st.columns(1)

#waveform = pd.DataFrame({"Amplitude": sound.values[0].T})
#st.line_chart(waveform)
sensor_of_interst='016_HG01VM001'

def detect_timeouts(df):
    dt = df.reset_index()['time']
    day = pd.Timedelta('5M')
    breaks = dt.diff() != day
    groups = breaks.cumsum()
    df['timeout'] = breaks.values.astype(int)

@st.cache
def load_data(sensor_of_interst='016_HG01VM001'):
    dataset = pd.read_excel('Ospel, Export Sensordata 2020.xlsx')
    dataset.set_index('DateTime', inplace=True)
    for col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col],errors='coerce')
    return dataset

def load_predictions(dataset):
    # Load sound into Praat
    df = np.load('predictions_016_HG01VM0012.npy', allow_pickle=True)
    perdictions_df = pd.DataFrame(df, columns=['time','prediction','var', 's2s', 'moments', 'msg'])
    perdictions_df.set_index('time', inplace=True)
    perdictions_df.index = pd.DatetimeIndex(perdictions_df.index)
    perdictions_df['prediction'] = [k[0] for k in perdictions_df['prediction'].values]
    perdictions_df['upper'] = perdictions_df['prediction']+perdictions_df['var']
    perdictions_df['lower'] = perdictions_df['prediction']-perdictions_df['var']
    perdictions_df['upper2'] = perdictions_df['prediction']+perdictions_df['prediction'].rolling(60).std()
    perdictions_df['lower2'] = perdictions_df['prediction']-perdictions_df['prediction'].rolling(60).std()
    perdictions_df['error'] = perdictions_df['prediction']-dataset.loc[perdictions_df.index, sensor_of_interst].values
    detect_timeouts(perdictions_df)
    return perdictions_df

def calculate_errors(df, dataset, window):
    sub_dataset = dataset.loc[perdictions_df.index]
    df.loc[:,'error'] = df['prediction']-sub_dataset.values
    df.loc[:,'error_abs'] = (df['prediction']-sub_dataset.values).abs()
    df.loc[:,'error_abs_{}'.format(window)] = (df['prediction']-sub_dataset.values).abs().rolling(window).mean()
    df.loc[:,'error_abs_flattend'] = (df.loc[:,'error']).abs()-df.loc[:,'error'].rolling(20*24).median().abs()
    df.loc[:,'pred_diff_{}'.format(window)] = df['prediction'].diff().abs().rolling(window).mean()
    df.loc[:,'measured_diff_{}'.format(window)] = sub_dataset.diff().abs().rolling(window).mean()
    df.loc[:,'error_diff'] = (df['prediction'].diff().abs() - sub_dataset.diff().abs()).abs()
    df.loc[:,'error_diff_{}'.format(window)] = (df.loc[:,'pred_diff_{}'.format(window)] - df.loc[:,'measured_diff_{}'.format(window)]).abs()

def group_events(df):
    dt = df.reset_index()['time']
    day = pd.Timedelta('5M')
    breaks = dt.diff() != day
    groups = breaks.cumsum()
    print(np.unique(groups.values))
    return pd.Series(groups.values, index=df.index)   

def detect_events(df, sub_dataset, level_error_abs=100, level_error_diff=60, error_abs=True, error_diff=True, and_=False, window=3):
    print(level_error_abs, level_error_diff)
    calculate_errors(df, sub_dataset, window)
    events_abs = df.loc[df['error_abs_{}'.format(window)]>level_error_abs]
    events_diff = df.loc[df['error_diff_{}'.format(window)]>level_error_diff]
    if error_abs and error_abs:
        events = pd.concat((events_abs, events_diff), axis=1)
        events = events.loc[events.index.drop_duplicates(keep = False)]
    elif error_diff:
        events = events_diff
    else:
        events = events_abs

    if and_:
        idx = events_abs.index.intersection(events_diff.index)
        events = events.loc[idx]
    
    events_groups = group_events(events)    
    df.loc[events_groups.index, 'events'] = events_groups.values
    return df

dataset = load_data(sensor_of_interst)
perdictions_df = load_predictions(dataset)
named_colorscales = px.colors.named_colorscales()

def draw_data(df, sub_dataset):
    # Plot with plotly
    data = [
        go.Scatter(x=pd.DatetimeIndex(df.index.values), y=df['prediction'], name='prediction'),
        go.Scatter(x=pd.DatetimeIndex(sub_dataset.index.values), y=sub_dataset.values, name='measured'),
    ]
    layout = go.Layout(
        yaxis=dict(title='CMH'),
        xaxis=dict(title='Time (s)'),
        height=400,
        width=1000,
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, height=800, width=1000, use_container_width =True)

def draw_errors_and_events(df):
    window = 3
    date_index = pd.DatetimeIndex(df.index.values)
    data = [
        go.Scatter(x=[date_index[0], date_index[-1]], y=[level_error_abs, level_error_abs]),
        go.Scatter(x=[date_index[0], date_index[-1]], y=[level_error_diff, level_error_diff]),
        go.Scatter(x=pd.DatetimeIndex(df.index.values), y=df['error_abs_{}'.format(window)].values, name='error_abs'),
        go.Scatter(x=pd.DatetimeIndex(df.index.values), y=df['error_diff'.format(window)].values, name='error_diff')
    ]
    
    layout = go.Layout(
        yaxis=dict(title='delta CMH'),
        xaxis=dict(title='Time (s)'),
        height=500,
        width=1000,
    )

    fig = go.Figure(data=data, layout=layout)
    for name, group in df.groupby('events'):
        if len(group)>min_event_length:
            print(name)
            fig.add_vrect(
                x0=pd.to_datetime(group.index.values[0]), x1=pd.to_datetime(group.index.values[-1]),
                fillcolor="LightSalmon", opacity=0.9,
                layer="below", line_width=1,
            )
    st.plotly_chart(fig, height=800, width=1000, use_container_width =True)

# Side Bar #######################################################
with st.sidebar:
    level_error_abs = st.slider('Abs Error', 0, 1000, 100)
    level_error_diff = st.slider('Diff error', 0, 1000, 50)

    min_event_length = st.slider('Miminmal event length', 0, 10, 1)

    # colours = st.selectbox(('Choose a colour pallete'), named_colorscales, index=default_ix)
    # dynamic_range = st.slider('Dynamic Range (dB)', 10, 100, 75)
    # window_length = st.slider('Window length (s)', 0.005, 0.05, 0.05)

# App ##################################################
# Load sound into Praat


draw_data(perdictions_df, dataset[sensor_of_interst])
events_df = detect_events(perdictions_df, dataset[sensor_of_interst], level_error_abs=level_error_abs, level_error_diff=level_error_diff, error_abs=True, error_diff=True, and_=False, window=3)
st.write("events detected: {}".format(len([name for name, group in events_df.groupby('events') if len(group)>min_event_length])))
draw_errors_and_events(events_df)
