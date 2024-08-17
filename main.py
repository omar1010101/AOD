import streamlit as st  
from datetime import date
from plotly import  graph_objs as go
import pandas as pd
st.set_page_config(page_title="File Uploader")

def plot_data(column,alarm=None,alert=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=df['value_created_at'], y=df[column], 
            mode='lines', 
            name=column
        ))
    fig.layout.update(title_text="Time series data", xaxis_rangeslider_visible=True)
    if alarm or alert :
        fig.add_hline(y=alert, line=dict(color="yellow" , width=2 , dash ="solid"))
        fig.add_hline(y=alarm, line=dict(color="Red" , width=2 , dash ="solid"))
    st.plotly_chart(fig)

    

st.title("Change Point detection And Forecasting Dashbord")

df = st.file_uploader(label = "Upload your dataset: ")

if df :
    df = pd.read_csv(df)
    df['value_created_at'] = pd.to_datetime(df['value_created_at'])
    st.write(df.head(5))
    metric = st.sidebar.text_input('Choose column to plot')
    
    if metric :
        if metric in  ('NGA','NGV') :
            alarm = st.sidebar.text_input('Treshold alarm')
            alert = st.sidebar.text_input('Treshold alert')
            plot_data(metric,alarm,alert)
        else :
            plot_data(metric)
        
        

# TODAY = date.today().strftime("%Y-%m-%d")

# period = st.slider("period of prediction: ",1,30*24)

forcasting, AOD = st.tabs(["Forcasting" , "AOD"])

with forcasting :
    st.header('Forcasting')
    date = ('Hour','Days' , 'months')
    selected_date = st.selectbox("Select Date : ",date)
    if selected_date == 'Hour' :
        period = st.slider("Period of prediction in hours: ",1,12)
    if selected_date == 'Days' :
        period = st.slider("Period of prediction in days: ",1,15)
    if selected_date == 'months' :
        period = st.slider("Period of prediction in months: ",1,5)
    
with AOD :
    st.header('AOD')
    
    
    