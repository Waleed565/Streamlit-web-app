import streamlit as st
import pandas as pd 
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import date
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
# Add an image from an online resource
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg",  width=1000)

# Take input from the user of the app about the start and end date

# Sidebar
st.sidebar.header('Select the parameters from below')

# Sidebar inputs for the date range
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2024, 12, 31))

# Add ticker symbol list
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch data from user inputs using yfinance library
data = yf.download(ticker, start=start_date, end=end_date)

# Ensure the dataframe has a single level of columns (flatten columns if necessary)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns]

# Add Date as a column to the dataframe
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)

# Display the data
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# Plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")

# Rename columns to match the format you’re expecting
data.rename(columns={
    'Open': f'Open_{ticker}',
    'High': f'High_{ticker}',
    'Low': f'Low_{ticker}',
    'Close': f'Close_{ticker}',
    'Volume': f'Volume_{ticker}'
}, inplace=True)

fig = px.line(data, x='Date', y=f'Close_{ticker}', title=f'Closing price of {ticker}', width=1000, height=600)
st.plotly_chart(fig)

# Add a select box to choose the column for forecasting
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

# Subsetting the data
data1 = data[['Date', column]]
st.write("Selected Data")
st.write(data1)

# ADF test to check stationarity
st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "News"])

with pricing_data:
    st.header('Pricing Data')
    data2 = data
    data2['%_Change'] = data[f'Close_{ticker}'] / data[f'Close_{ticker}'].shift(1) - 1
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['%_Change'].mean() * 252*100
    stdev = np.std(data2['%_Change']) * np.sqrt(252) * 100
    st.write("Annual Return: ", annual_return, '%')
    st.write("Standard Deviation: ", stdev, '%')
    st.write('Risk Adjusted Return is:', annual_return /(stdev * 100))


from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
    st.write("Fundamental Data")
    key = '5MD078DOTSP4T5S8'
    fd = FundamentalData(key, output_format='pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    st.write(is1)
    st.subheader('Cash Flow Statement')
    cash_flow_statement = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow_statement.T[2:]
    cf.columns = list(cash_flow_statement.T.iloc[0])
    st.write(cf)
    

from stocknews import StockNews
with news:
    st.header('News of {ticker}') 
    sn = StockNews(ticker, save_news = False)
    df_news = sn.read_rss()
    for i in range (10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i]) 
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i]) 
        title_sentiment = df_news['title_sentiment'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')
        st.write('-----------------------------')    


# Model selection
models = [ 'Random Forest',  'Prophet']
selected_model = st.sidebar.selectbox('Select the model for forecasting', models)


if selected_model == 'Prophet':
 # Prepare the data for Prophet
    prophet_data = data[['Date', column]]
    prophet_data = prophet_data.rename(columns={'Date': 'ds', column: 'y'})

    # Create and fit the Prophet model
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)

    # Forecast the future values
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)

    # Plot the forecast
    fig = prophet_model.plot(forecast)
    plt.title('Forecast with Facebook Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)
elif selected_model == 'Random Forest':
    # Random Forest Model
    st.header('Random Forest Regression')

    # Splitting data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Feature engineering
    train_X, train_y = train_data['Date'], train_data[column]
    test_X, test_y = test_data['Date'], test_data[column]

    # Initialize and fit the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(train_X.values.reshape(-1, 1), train_y.values)

    # Predict the future values
    predictions = rf_model.predict(test_X.values.reshape(-1, 1))

    # Calculate mean squared error
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)

    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Combine training and testing data for plotting
    combined_data = pd.concat([train_data, test_data])

    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[column], mode='lines', name='Actual',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data["Date"], y=predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (Random Forest)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
    st.plotly_chart(fig)


    