import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta
from datetime import datetime
import plotly.graph_objects as go
import plotly.subplots as sp
from tensorflow import keras
import streamlit as st
import pandas_ta as pta


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load the datasets

# Get a list of available cryptocurrencies from the filenames in the directory
path_4h = "C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/binance_4h_coin_datasets/"
files = os.listdir(path_4h)
coins = [file.split('_')[0] for file in files if file.endswith(".csv")]


def app():
    # User select a cryptocurrency
    selected_coin = st.selectbox('Select a cryptocurrency', coins)

        ###indicators graphic function: 
    
    def plot_indicators(df, upper_cols, lower_cols):
    
        df = df.iloc[-30:]

        fig = sp.make_subplots(rows=2, cols=1)

        # Add a line for each column in upper_cols
        for col in upper_cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col], 
                    mode='lines',
                    name=col
                ),
                row=1,
                col=1
            )

        # Add a line for each column in lower_cols
        for col in lower_cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col], 
                    mode='lines',
                    name=col
                ),
                row=2,
                col=1
            )

        fig.update_layout(
            title=f'Indicators',
            xaxis=dict(
                title='Date',
            ),
            yaxis=dict(
                title='Value',
            )
        )

        return fig


        # Load the selected CSV file

    if st.button('Ask to AI'):

    # Load the selected CSV file
        coin_4h_path = path_4h + selected_coin + "_binance_4h.csv"
        coin_daily_path = "C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/binance_daily/" + selected_coin + "_binance_1d.csv"
        btc_mc_path='C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/Coingecko_market_cap/bitcoin_market_cap.csv'
        tether_mc_path= 'C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/Coingecko_tether_market_cap/tether_market_cap.csv'

        coin_4h_df = pd.read_csv(coin_4h_path)
        coin_daily_df = pd.read_csv(coin_daily_path)
        btc_mc= pd.read_csv(btc_mc_path)
        tether_mc= pd.read_csv(tether_mc_path)
        

        # Convert 'Time_Date' to datetime format
        coin_4h_df['Time_Date'] = pd.to_datetime(coin_4h_df['Time_Date'])
        coin_daily_df['Time_Date'] = pd.to_datetime(coin_daily_df['Time_Date'])
        btc_mc['Time_Date'] = pd.to_datetime(btc_mc['Time_Date'])
        tether_mc['Time_Date'] = pd.to_datetime(tether_mc['Time_Date'])

        last_date = coin_4h_df['Time_Date'].max()

        coin_4h_df['EMA_stoch_9_4h'] = pta.ema(coin_4h_df['STOCHRSIk_14_14_3_3'], length=9)
    

    # Merge the dataframes
        btc_df = pd.merge_asof(coin_4h_df, coin_daily_df[['Time_Date', 'scaled_price', 'RSI_daily']], on='Time_Date', direction='backward')
        btc_df = pd.merge_asof(btc_df, btc_mc, left_index=True, right_index=True, direction='backward')
        btc_df= pd.merge_asof(btc_df, tether_mc, left_index=True, right_index=True, direction='backward')

        btc_df_updated = btc_df.drop(['Time_Date', 'close', 'BBM_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'open_time','Time_Date_y', 'Time_Date_x'], axis=1)
        btc_df_updated.fillna(btc_df_updated.mean(), inplace=True)
              
        btc_df_updated['timestamp'] = pd.to_datetime(btc_df_updated['timestamp'])
        btc_df_updated['day'] = btc_df_updated['timestamp'].dt.day
        btc_df_updated['hour'] = btc_df_updated['timestamp'].dt.hour
        btc_df_updated.set_index('timestamp', inplace=True)  
        
        X = btc_df_updated.drop('price', axis=1)
        y = btc_df_updated['price']

    #    Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)  # Added shuffle=False because sequential

        n_steps = 30
        n_features = X_train.shape[1]

    # Apply MinMax Scaler for X
        scaler = MinMaxScaler(feature_range=(0.01, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Fit a separate scaler for y
        scaler_y = MinMaxScaler(feature_range=(0.01, 1))
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))  # reshape to 2D array because MinMaxScaler expects 2D input
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)) # Use the same scaler for y_test

        def create_sequences(X, y, time_steps=n_steps):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)


        X_train_reshaped, y_train_reshaped = create_sequences(X_train_scaled, y_train_scaled) 
        X_test_reshaped, y_test_reshaped = create_sequences(X_test_scaled, y_test_scaled) # Use y_test_scaled here

    # Continue with the model as usual
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(None, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
    

        model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit model
        model.fit(X_train_reshaped, y_train_reshaped, epochs=200, verbose=2)

    # Make predictions
        y_pred = model.predict(X_test_reshaped, verbose=0) 

    # Evaluate model
        mse = mean_squared_error(y_test_reshaped, y_pred) # Use y_test_reshaped here
        print('Test MSE: %.3f' % mse)
    
    # Calculate MAPE
        def mape(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
        mape_score = mape(y_test_reshaped, y_pred)

    # Make predictions
        y_pred = model.predict(X_test_reshaped, verbose=0)


    # Future prediction logic

            # Future prediction logic

        n_future = 5 * 24  # 5 days ahead
        input_sequence = X_train_scaled[-n_steps:]
        forecast = []

        for _ in range(n_future):
            input_sequence_reshaped = input_sequence.reshape((1, n_steps, n_features))
            next_step = model.predict(input_sequence_reshaped)[0] # Prediction from the model
            next_step_repeated = np.repeat(next_step, n_features).reshape(1, n_features)
            forecast.append(next_step[0]) # Append the original prediction, not the repeated one
            input_sequence = np.vstack((input_sequence[1:], next_step_repeated)) # Use the repeated prediction to maintain the shape

    # Rescale forecasted data
        forecast = np.array(forecast).reshape(-1, 1)
        forecast_rescaled = scaler_y.inverse_transform(forecast) 

    # Create dates range for original data and predictions
        dates = pd.date_range(start=coin_4h_df['Time_Date'].min(), periods=len(X_train) + len(X_test) + n_steps, freq='4H')
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=4), periods=n_future, freq='4H')

    # Get original prices
        original_prices = np.concatenate([y_train, y_test])

        forecast_df = pd.DataFrame(data={"Forecast": forecast_rescaled.flatten(), "Time_Date": forecast_dates})
        forecast_df.to_csv("forecast.csv", index=False)

        model.save("my_model.h5")


        def plot_forecast(actual, forecast):
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=actual.index,
                    y =actual,
                    mode='lines',
                    name='Actual'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast.index,
                    y=forecast,
                    mode='lines',
                    name='Forecast'
                )
            )

            fig.update_layout(
                title=f'{selected_coin} Price Forecast',
                xaxis=dict(
                    title='Date',
                ),
                yaxis=dict(
                    title='Price',
                )
            )

            return fig

    # Streamlit 

    # Load actual data
        actual_df = pd.read_csv(f"C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/binance_4h_coin_datasets/{selected_coin}_binance_4h.csv")  
        actual_df['Date'] = pd.to_datetime(actual_df['Time_Date'])
        actual_df.set_index('Date', inplace=True)

    # Load the forecasted data
        forecast_df = pd.read_csv("C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/forecast.csv")

    # Set the Time_Date column to datetime format and set it as the index
        forecast_df['Time_Date'] = pd.to_datetime(forecast_df['Time_Date'])
        forecast_df.set_index('Time_Date', inplace=True)
        update_date = sorted(forecast_df.index)[1]

    # Plot 'price' column from actual_df and 'Forecast' column from forecast_df
        fig = plot_forecast(actual_df['price'], forecast_df['Forecast'])
        st.plotly_chart(fig)
    
        st.write(f"Error Percentage (MAPE): {mape_score}%")
        
        # Get the indices where y_test_reshaped is zero
        #zero_indices = np.where(y_test_reshaped == 0)[0]
        
    # Second visualization: 'hl2', 'median_hl2_14' and 'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3', 'EMA_stoch_9_4h'
        st.plotly_chart(plot_indicators(btc_df_updated, ['hl2', 'median_hl2_14'], ['STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3', 'EMA_stoch_9_4h']))

    # Third visualization: 'ALMA_9_08_6', 'T3_10_0.7' and 'FISHERT_9_1', 'FISHERTs_9_1'
        st.plotly_chart(plot_indicators(btc_df_updated, ['ALMA_9_08_6', 'T3_10_0.7'], ['FISHERT_9_1', 'FISHERTs_9_1']))

    # Fourth visualization: 'EMA_price_9_4h', 'price' and 'SMI_Ergodic', 'SMI_Ergodic_Signal', 'VWMA_20_SMII'
        st.plotly_chart(plot_indicators(btc_df_updated, ['EMA_price_9_4h', 'price'], ['SMI_Ergodic', 'SMI_Ergodic_Signal', 'VWMA_20_SMII']))

    # Fifth visualization: 'BBL_20_2.0', 'BBU_20_2.0', 'price' and 'MACDh_12_26_9', 'MACDs_12_26_9'
        st.plotly_chart(plot_indicators(btc_df_updated, ['BBL_20_2.0', 'BBU_20_2.0', 'price'], ['MACDh_12_26_9', 'MACDs_12_26_9']))

   
        st.write(f"Last updated date: {update_date.strftime('%Y-%m-%d %H:%M')} hours")


if __name__== "__main__":
    app()