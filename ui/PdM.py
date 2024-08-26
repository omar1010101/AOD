import streamlit as st 
from streamlit_option_menu import option_menu
from datetime import date
from plotly import  graph_objs as go
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
# TODAY = date.today().strftime("%Y-%m-%d")
st.set_page_config(page_title="File Uploader", page_icon = "📚")
selected = option_menu (
    menu_title = None,
    options = ["Home", "Project" , "Contact"],
    icons = ["house", "book", "envelope"],
    menu_icon = "cast",
    default_index = 0,
    orientation= "horizontal"
)


path_model = "../model_last.keras"
WINDOW_SIZE = 200
delay = 2
mydata = pd.DataFrame()
test_size = 0.52
list_sensors = ['NGA']
list_metrics = ['mean','Median', 'STD', 'PPV','rms','cf','mf','sf','if','IQR','IDC','QCD','msra']
window_size = 100
cut_factor = .2


def run_model(df,periods ,metric):
    df['value_created_at'] = pd.to_datetime(df['value_created_at'])
    df.set_index('value_created_at', inplace=True)
    mydata = df[[metric]]
    mydata['set'] = 'actual'
    model1 = load_model(path_model)
    for period in range(periods) :
        if delay == 1 :
            prediction = model1.predict(mydata[metric].values[-WINDOW_SIZE * delay : ].reshape((-1,WINDOW_SIZE,1)))
        else :
            prediction = model1.predict(mydata[metric].values[-WINDOW_SIZE * delay : -WINDOW_SIZE * delay + WINDOW_SIZE].reshape((-1,WINDOW_SIZE,1)))
        last_timestamp = mydata.index[-1]
        new_timestamp = last_timestamp + pd.Timedelta(hours=1)
        new_row = pd.DataFrame({'value_created_at': [new_timestamp], metric: [prediction[0][0]]})
        new_row['set'] = 'predicted'
        new_row.set_index('value_created_at', inplace=True)
        mydata = pd.concat([mydata, new_row], ignore_index=False)
        
    return mydata

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

def calc_metric_stats(serie, metric) : 
    if metric == "mean": 
        return np.mean(serie)
    elif metric == 'STD':
        return np.std(serie)
    elif metric == 'PPV':
        return np.max(serie) - np.min(serie)
    elif metric == 'Median':
        return np.median(serie)
    elif metric == 'rms':
        return np.sqrt(np.mean(np.square(serie)))
    elif metric == 'cf':
        max_value = np.max(np.abs(serie))
        sum_values = np.sum(np.square(serie))
        return max_value / np.sqrt(sum_values)
    elif metric == 'mf':
        mu = np.mean(serie)
        max_value = np.max(serie)
        min_value = np.min(serie)
        return (max_value - min_value) / mu
    elif metric == 'sf':
        mu = np.mean(serie)
        sigma = np.std(serie)
        return sigma / mu
    elif metric == 'if':
        max_value = np.max(np.abs(serie))
        sum_values = np.sum(np.abs(serie))
        return max_value / sum_values
    elif metric == 'IQR':
        Q1 = np.quantile(serie, .25)
        Q3 = np.quantile(serie, .75)
        return Q3 - Q1
    elif metric == 'IDC':
        D1 = np.percentile(serie, 10)
        D9 = np.percentile(serie, 90)
        return D9 - D1
    elif metric == 'QCD':
        Q1 = np.quantile(serie, .25)
        Q3 = np.quantile(serie, .75)
        return (Q3 - Q1) / (Q3 + Q1)
    elif metric == 'msra' :
        valeur_abs = np.abs(serie)
        racine = np.sqrt(valeur_abs) 
        msra = ((1/len(serie)) * np.sum(racine)) ** 2
        return msra
    # elif metric == "MidRange" :
    #     return (np.min(x) + np.max(x)) / 2
    # elif metric == "MidHinge" :
    #     Q1 = np.quantile(serie, .25)
    #     Q3 = np.quantile(serie, .75)
    #     return (Q3 + Q1) / 2       

def Algorithm2(F, J, K, cut_factor, data_full):
    list_idx_cut = list()
    cut = int(np.round(cut_factor * K))
    if(cut > 0):
        D = F*J
        lpoints = list()
        for k in range(K):
            point = np.zeros(D)
            d = 0
            for i in range(F):
                for j in range(J):
                    point[d] = data_full[k][i][j]
                    d += 1
            lpoints.append(point)
            
        maxpoint = np.full(D, -np.inf)
        minpoint = np.full(D, np.inf)

        for k in range(K):
            for d in range(D):
                maxpoint[d] = max(lpoints[k][d], maxpoint[d])
                minpoint[d] = min(lpoints[k][d], minpoint[d])

        lpoints_normalized = list()
        for k in range(K):
            point = np.zeros(D)
            for d in range(D):
                point[d] = (lpoints[k][d] - minpoint[d])/(maxpoint[d] - minpoint[d])
            lpoints_normalized.append(point)

        centroid = np.zeros(D)
        for k in range(K):
            for d in range(D):
                centroid[d] += lpoints_normalized[k][d]/K

        list_dist_centroid = list()
        for k in range(K):
            list_dist_centroid.append(np.linalg.norm(lpoints_normalized[k]-centroid))

        order = np.argsort(list_dist_centroid)
        for c in range(cut):
            list_idx_cut.append(order[len(order)-c-1])
    return list_idx_cut

def create_dataframe(data_range, list_sensors, list_metrics):
    temp = {"min" : {} , "max" : {}}
    for idx_sensor in range(len(list_sensors)) :
        for metric in list_metrics : 
            temp["min"][(list_sensors[idx_sensor], metric)] = data_range[idx_sensor][0][list_metrics.index(metric)]
            temp["max"][(list_sensors[idx_sensor], metric)] = data_range[idx_sensor][1][list_metrics.index(metric)]
    # print(temp)
    return pd.DataFrame(temp)

def algorithme1(metric,list_sensors, list_metrics, df_train, window_size = 100, cut_factor = .2):
    df_train.reset_index(drop = True, inplace=True)
    F = len(list_sensors)
    J = len(list_metrics)
    K = int(len(df_train)//window_size)
    N = window_size
    data_full = np.zeros((K,F,J))

    for k in range(K):
        for i in range(F):
            data_sensor = df_train[list_sensors[i]].values
            #serie = data_sensor[df_train.shape[0]- (k+1)*N : df_train.shape[0] - k*N]
            serie = data_sensor[ k*N + df_train.shape[0]%100 : (k+1)*N + df_train.shape[0]%100]
            #print(f"{ k*N + df_train.shape[0]%100} : {(k+1)*N + df_train.shape[0]%100}")
            for j in range(J):
                # print(list_metrics[j])
                data_full[k][i][j] = calc_metric_stats(serie, list_metrics[j])
    print(f"shape of data full : {data_full.shape}" )   

    list_idx_cut = Algorithm2(F, J, K, cut_factor, data_full)
    print(f"shape of cut index : {list_idx_cut}")

    data_length = len(df_train[metric].values)
    fig = go.Figure()
    
    # Ajouter la trace des données
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train[metric].values, mode='lines', name=metric))
    
    # Ajouter les zones rouges pour les indices de coupure
    for idx in list_idx_cut:
        start = idx * 100
        end = (idx + 1) * 100
        if start < data_length:
            end = min(end, data_length)
            fig.add_shape(
                type='rect',
                x0=start,
                x1=end,
                y0=df_train[metric].min(),
                y1=df_train[metric].max(),
                fillcolor='red',
                opacity=0.3,
                line=dict(color='red', width=0)
            )
    
    # Mise à jour de la mise en page
    fig.update_layout(
        title='Visualisation des données avec zones marquées',
        xaxis_title='Index',
        yaxis_title=metric,
        xaxis=dict(range=[0, data_length]),
        yaxis=dict(range=[df_train[metric].min(), df_train[metric].max()])
    )
    
    # Afficher la figure
    st.plotly_chart(fig)
    
    data_range = list()
    for i in range(F):
        lmtr_min = list()
        lmtr_max = list()
        for j in range(J):
            list_metric_per_window = list()
            for k in range(K):
                if(k not in list_idx_cut):
                    list_metric_per_window.append(data_full[k][i][j])
            lmtr_min.append(np.min(list_metric_per_window))
            lmtr_max.append(np.max(list_metric_per_window))
        data_range.append((lmtr_min, lmtr_max))

    df_model = create_dataframe(data_range, list_sensors, list_metrics)
    return df_model

def test(list_sensors, list_metrics, current_window, df_model):
    current_window = current_window[current_window.NGA > .2]
    color = []
    F = len(list_sensors)
    J = len(list_metrics)
    report = dict()
    report['count_metrics_affects'] = 0
    report['count_metrics_not_affects'] = 0
    sum_error_rel = 0
    for name_sensor in list_sensors:
        serie = current_window[name_sensor].values
        for j in range(J):
            metric = calc_metric_stats(serie, list_metrics[j])
            v_min =  df_model.loc[name_sensor].loc[list_metrics[j]]['min']
            v_max =  df_model.loc[name_sensor].loc[list_metrics[j]]['max']
            if metric < v_min : 
                color.append("red")
                error_rel_normalized = np.abs((metric-v_min)/(v_max-v_min))
                sum_error_rel += error_rel_normalized
                report['count_metrics_affects'] = report['count_metrics_affects'] + 1
            elif metric > v_max :
                color.append("red")
                error_rel_normalized = np.abs((metric-v_max)/(v_max-v_min))
                sum_error_rel += error_rel_normalized
                report['count_metrics_affects'] = report['count_metrics_affects'] + 1
            else : 
                color.append("green")
                report['count_metrics_not_affects'] = report['count_metrics_not_affects'] + 1
    report['colors'] = color

    if report['count_metrics_affects'] > 1 :
        report['error'] = sum_error_rel * 100 / J*F
    
    return report

def test_data(test_data,window_size,mesure,df_model):
    reports = []
    K = int(len(test_data)//window_size)
    N = window_size

    test_data.reset_index(drop = True, inplace=True)

    for k in range(K) :  
        serie = test_data[ k*N : (k+1)*N ]
        reports.append(test(list_sensors, list_metrics, serie, df_model))

    return reports    
       
def AOD(df,metric,sensibility_alarm = 75,sensibility_alert = 65,alarm=None,alert=None):

    Treshold_alert = alert
    Treshold_alarm = alarm
    
    df_test = df.tail( int(test_size * df.shape[0] // window_size) * window_size)
    df_train = df.loc[~df.index.isin(df_test.index)]
    
    df_train['set'] = 'train'
    df_test['set'] = 'test'

    df_combined = pd.concat([df_train, df_test])

    fig = px.line(df_combined, x='value_created_at', y=metric, color='set', title='Séparation des ensembles d\'entraînement et de test')
    st.plotly_chart(fig)
    
    df_train = df_train[df_train[metric] > .2]
    
    df_model = algorithme1(metric,list_sensors, list_metrics, df_train, window_size = 100, cut_factor=.1)
    
    reports = test_data(df_test, 100, metric,df_model)

    colors = []
    for report in reports : 
        if report['count_metrics_affects'] >= round(len(list_metrics) *(sensibility_alarm/100)) : #rouge
            colors.extend(["red" for _ in range(100)])
        elif report['count_metrics_affects'] >= round(len(list_metrics) *(sensibility_alert/100)) : #orange
            colors.extend(["orange" for _ in range(100)])
        else : #green
            colors.extend(["green" for _ in range(100)])
            
    x = np.arange(len(df_test))
    y = df_test[metric]
    coeffs = np.polyfit(x, y, 1)
    trendline = np.polyval(coeffs, x)
    df_test['colors'] = colors

    fig = go.Figure()

    for i in range(df_test.shape[0] // 100):
        segment = df_test.iloc[i*100 : (i+1)*100]
        if "red" in segment['colors'].values :      
            fig.add_trace(go.Scatter(
                x=segment['value_created_at'], y=segment[metric], 
                mode='lines', 
                marker=dict(color="red"),
                name="red"
            ))
        elif "orange" in segment['colors'].values :      
            fig.add_trace(go.Scatter(
                x=segment['value_created_at'], y=segment[metric], 
                mode='lines', 
                marker=dict(color="orange"),
                name="orange"
            ))
        else :
            fig.add_trace(go.Scatter(
                x=segment['value_created_at'], y=segment[metric], 
                mode='lines', 
                marker=dict(color="green"),
                name="green"
            ))
        fig.add_trace(go.Scatter(
            x=df_test['value_created_at'], y=trendline, 
            mode='lines', 
            line=dict(color='black', width=2),
            name='Tendance'
        ))
    fig.update_xaxes(rangeslider_visible=True)
    if alarm or alert :
        fig.add_hline(y=alert, line=dict(color="yellow" , width=2 , dash ="solid"))
        fig.add_hline(y=alarm, line=dict(color="Red" , width=2 , dash ="solid"))
    fig.update_layout(title='test data',
                    xaxis_title='Date',
                    yaxis_title = metric)
    return fig




if selected == "Home" :
    st.title("Predictive Maintenance System")
    
    st.markdown("""
    *Welcome to our predictive maintenance platform.
    To get started, please enter the following information regarding the metric you want to monitor:*
    """)
    
    metric_name = st.text_input("Sensor Name")
    measurement_type = st.selectbox("Measurement Type", ["Temperature", "Vibration"])
    metric_alert = st.number_input("Alert Threshold", min_value=0.0, max_value=100.0, step=0.1)
    metric_alarm = st.number_input("Alarm Threshold", min_value=0.0, max_value=100.0, step=0.1)

    names = ['Forecasting', 'Change Point Detection']
    name = st.radio('Choose a method :', names, horizontal=True,index = 0)
    if st.button("Submit"):
        if not metric_name:
            st.error("Sensor Name is required.")
        elif metric_alert == 0.0 and metric_alarm == 0.0:
            st.error("Alert and Alarm Thresholds must be greater than zero.")
        else:
            st.session_state.metric_name = metric_name
            st.session_state.measurement_type = measurement_type
            st.session_state.metric_alert = metric_alert
            st.session_state.metric_alarm = metric_alarm
            st.session_state.name = name
            
            st.success("Data submitted successfully!")
    
    
elif selected == "Project" :
    if 'metric_name' in st.session_state:
                
        metric = st.session_state.metric_name
        alert = st.session_state.metric_alert
        alarm = st.session_state.metric_alarm
        name = st.session_state.name
        
        st.title("Change Point detection And Forecasting Dashbord")

        path = st.file_uploader(label = "Upload your dataset: ")    
        
        if path :
        
            df = pd.read_csv(path)
            df['value_created_at'] = pd.to_datetime(df['value_created_at'])
            st.write("**Data**")
            st.write(df.head(5))
    
            if metric :
                if metric in  ('NGA','NGV') :
                    # alarm = st.sidebar.text_input('Treshold alarm')
                    # alert = st.sidebar.text_input('Treshold alert')
                    plot_data(metric,alarm,alert)
                    
                    if name == "Forecasting" :
                        st.header("Forcasting")  
                    
                        period_names = ['1 Hour','1 Day' , '1 month']
                        period = st.radio('Period',period_names,horizontal=True)
                        if period == '1 Hour' :
                            st.write("**Period of prediction is 1 hours:**")
                            nga = run_model(df,1,metric)
                            st.write(nga[nga.set == "predicted"])
                            custom_colors = ['#EF553B', '#00CC96']
                            fig = go.Figure()
                            fig = px.line(nga, y=metric, color='set', color_discrete_sequence=custom_colors,title='.....')
                        elif period == '1 Day' :
                            st.write("**Period of prediction is 1 Day:**")
                            nga = run_model(df,24,metric)
                            st.write(nga[nga.set == "predicted"])
                            custom_colors = ['#EF553B', '#00CC96']
                            fig = go.Figure()
                            fig = px.line(nga, y=metric, color='set', color_discrete_sequence=custom_colors,title='.....')
                        elif period == '1 month' :
                            st.write("**Period of prediction is 1 Month:**")
                            nga = run_model(df,30*24,metric)
                            st.write(nga[nga.set == "predicted"])
                            custom_colors = ['#EF553B', '#00CC96']
                            fig = go.Figure()
                            fig = px.line(nga, y=metric, color='set', color_discrete_sequence=custom_colors,title='.....')
                        st.plotly_chart(fig)
                
                    elif name == "Change Point Detection":
                        st.header('AOD')
                        st.write("**choose sensibility Alarm (%):**")
                        sensibility_alarm = st.slider("sensibility Alarm", 0, 100)
                        
                        st.write("**choose sensibility Alert (%):**")
                        sensibility_alert = st.slider("sensibility Alert", 0, 100)
                        
                        if sensibility_alarm == 0 and sensibility_alert == 0 :
                            sensibility_alarm , sensibility_alert = 75, 65
                        if metric :
                            fig = AOD(df,metric,sensibility_alarm,sensibility_alert,alarm,alert)
                        else :
                            fig = AOD(df,metric,sensibility_alarm,sensibility_alert)
                        st.plotly_chart(fig)
                        
                else :
                    plot_data(metric)
        

    else:
        st.warning("No data available. Please submit the information in the Home section.")
elif selected == "Contact":
    st.title("Contact Us")
    
    # Contact Information
    st.write("### Contact Information")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.image("https://img.icons8.com/ios-filled/50/000000/name.png", width=25)  # Name icon
    with col2:
        st.write("**Society Name:**  OCP Maintenance Solutions")
    
    with col1:
        st.image("https://img.icons8.com/ios-filled/50/000000/marker.png", width=25)  # Location icon
    with col2:
        st.write("**Location:**  OCP SA, Route Jorf Lyoudi, Safi 46000")
    
    with col1:
        st.image("https://img.icons8.com/ios-filled/50/000000/phone.png", width=25)  # Phone icon
    with col2:
        st.write("**Phone:**  (+212) 664-762708")
    
    with col1:
        st.image("https://img.icons8.com/ios-filled/50/000000/email.png", width=25)  # Email icon
    with col2:
        st.write("**Email:**  contact@ocp-ms.com")
    
    with col1:
        st.image("https://img.icons8.com/ios-filled/50/000000/user-group-man-man.png", width=25)  # Team icon
    with col2:
        st.write("**Supervisor:**  Mr KARMOUD Mohamed")
    
    st.write("### Additional Information")
    
    st.write("""
    *This project involves change point detection using an unsupervised method for anomaly detection in time series based on statistical features for industrial predictive maintenance and forecasting using an LSTM model. 
    The project is being conducted by Mr. Abdelhamid CHEBEL and Mr. Omar ELKALKHA.*
    """)