import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas_ta as pta
import streamlit as st
from PIL import Image, ImageDraw


# Daily data
merged_data_folder = "C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/binance_daily"

coin_symbols = [filename.split("_")[0] for filename in os.listdir(merged_data_folder) if filename.endswith(".csv")]

coin_data = {}
coin_rsi_values = {}

for symbol in coin_symbols:
    csv_path = os.path.join(merged_data_folder, f"{symbol}_binance_1d.csv")
    if os.path.isfile(csv_path):
        coin_df_daily = pd.read_csv(csv_path)
        coin_df_daily = coin_df_daily.rename(columns={
        'open_time': 'Time',
        'open': 'open',
        'high': 'high_max',
        'low': 'low_max',
        'volume': 'total_volume',
        'close' : 'Price'
        })

        # Check if the 'Time' column exists in the DataFrame
        if 'Time' in coin_df_daily.columns:
            coin_df_daily['Time'] = pd.to_datetime(coin_df_daily['Time'])
            coin_df_daily = coin_df_daily[coin_df_daily['Time'] != 0]
            coin_df_daily = coin_df_daily.dropna(subset=['Time'])

            if not coin_df_daily.empty:
                coin_df_daily['RSI'] = pta.rsi(coin_df_daily['Price'], length=14)
                coin_rsi_values[symbol] = coin_df_daily['RSI'].iloc[-1]  # Get the most recent RSI value
                 

                ### moving averages for MACD
                coin_df_daily['SMA_50'] = coin_df_daily['Price'].rolling(window=50).mean()
                coin_df_daily['SMA_100'] = coin_df_daily['Price'].rolling(window=100).mean()
                coin_df_daily['SMA_200'] = coin_df_daily['Price'].rolling(window=200).mean()

                # Calculate ADX
                coin_df_daily['ADX'] = pta.adx(coin_df_daily['high_max'], coin_df_daily['low_max'], coin_df_daily['Price'], length=14)['ADX_14']

            else:
                print(f"Empty DataFrame for {symbol}")
        else:
            print(f"Missing 'Time' column in DataFrame for {symbol}")

        coin_data[symbol] = coin_df_daily
        
def get_daily_rsi_values():  
                    return coin_rsi_values

### RSI

sorted_coin_rsi = sorted(coin_rsi_values.items(), key=lambda x: x[1])

# Top 20 coins with lowest RSI values
top_20_coins_lowest_rsi = sorted_coin_rsi[:20]

#goldencross detector

def cross_detection(sma50, sma200):
    if len(sma50) < 9 or len(sma200) < 9:
        return 'none', None

    golden_cross = False
    death_cross = False

    for i in range(1, 9):  # Check for the last 8 days
        if sma50[-i] > sma200[-i] and sma50[-(i + 1)] <= sma200[-(i + 1)]:
            if i >= 3:  # Check if the cross happened at least 3 days before
                golden_cross = True

        if sma50[-i] < sma200[-i] and sma50[-(i + 1)] >= sma200[-(i + 1)]:
            if i <= 5:  # Check if the cross happened at most 5 days before
                death_cross = True

    risk_alert = ''
    distance = abs(sma50[-1] - sma200[-1]) / ((sma50[-1] + sma200[-1]) / 2)
    if distance <= 0.02:
        risk_alert = 'risk'

    if golden_cross and not death_cross:
        return 'golden_cross', risk_alert
    elif death_cross and not golden_cross:
        return 'death_cross', risk_alert
    else:
        return 'none', risk_alert

# Streamlit interface

def app():
    st.title('Cryptocurrency Visualizations')

    coin_symbols = list(coin_data.keys())
    selected_coin = st.selectbox('Select a coin', coin_symbols)

    if selected_coin is not None:
        daily_data = coin_data[selected_coin]
        recent_daily_data = daily_data.iloc[-100:]  # Use only the last 100 days for visualization

    # Check if the necessary columns exist in the DataFrame
        required_columns = {'high_max', 'low_max', 'Price', 'RSI'}
        if required_columns.issubset(recent_daily_data.columns):
        # Price visualizations
            price_chart = go.Figure(go.Candlestick(x=recent_daily_data['Time'],
                                               open=recent_daily_data['open'],
                                               high=recent_daily_data['high_max'],
                                               low=recent_daily_data['low_max'],
                                               close=recent_daily_data['Price'],
                                               name=f'{selected_coin} Price'))

            price_chart.update_layout(title=f'{selected_coin} Candlestick Chart')
            st.plotly_chart(price_chart)

        # RSI-ADX graphic definitions
            def get_color_code_and_arrow(rsi_value, adx_value):
                if rsi_value <= 20 and adx_value < 20:
                    return 'green', 'triangle-up', None
                elif 21 <= rsi_value <= 40 and adx_value < 30:
                    return 'blue', 'triangle-up', None
                elif 41 <= rsi_value <= 50 and adx_value > 35:
                    return 'orange', 'triangle-up', None
                elif 50 <= rsi_value <= 69 and adx_value > 35:
                    return 'red', 'triangle-down', 'Peak point alert'
                elif rsi_value >= 70 and adx_value > 30:
                    return 'black', 'triangle-down', None
                else:
                    return 'grey', 'circle', 'Any signal'

            last_rsi = recent_daily_data['RSI'].iloc[-1]
            last_adx = recent_daily_data['ADX'].iloc[-1]

            rsi_color, arrow_symbol, message = get_color_code_and_arrow(last_rsi, last_adx)

            if message is not None:
                st.write(message)
          
            def create_rsi_adx_chart(df):
                fig = go.Figure()

             # Add RSI data
                fig.add_trace(go.Scatter(x=df['Time'], y=df['RSI'], name='RSI', mode='lines'))

            # Add ADX data with a secondary y-axis
                fig.add_trace(go.Scatter(x=df['Time'], y=df['ADX'], name='ADX', mode='lines',  line=dict(color='red')))

                last_rsi = df['RSI'].iloc[-1]
                last_adx = df['ADX'].iloc[-1]

                rsi_color, arrow_symbol, _ = get_color_code_and_arrow(last_rsi, last_adx)

                fig.add_trace(go.Scatter(x=[df['Time'].iloc[-1]], y=[df['RSI'].iloc[-1]], mode='markers', marker=dict(color=rsi_color, symbol=arrow_symbol, size=30)))

            # Update the layout to include a secondary y-axis
                fig.update_layout(
                    title="RSI and ADX Line Chart",
                    yaxis=dict(title="RSI"),
                    yaxis2=dict(title="ADX", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )

                return fig

            rsi_adx_chart = create_rsi_adx_chart(recent_daily_data)
            st.plotly_chart(rsi_adx_chart)

            if 15 <= last_adx <= 25:
                st.markdown('<p style="color:red; font-weight:bold;">Trend change signal detected!</p>', unsafe_allow_html=True)
            elif last_adx > 25:
                st.markdown('<p style="color:red; font-weight:bold;">Same trend!</p>', unsafe_allow_html=True)


            # Volume visualizations
                
            volume_chart = go.Figure(go.Bar(x=recent_daily_data['Time'], y=recent_daily_data['total_volume'], name='24h Volume'))
            volume_chart.update_layout(title=f'{selected_coin} 24h Volume Bar Chart') 
            st.plotly_chart(volume_chart)

            # Moving Averages visualizations
            ma_chart = go.Figure()

            # Plot the price data
            ma_chart.add_trace(go.Scatter(x=recent_daily_data['Time'], y=recent_daily_data['Price'], mode='lines', name='Price'))

            # Plot the 50-day moving average
            ma_chart.add_trace(go.Scatter(x=recent_daily_data['Time'], y=recent_daily_data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='green', width=2)))

            # Plot the 100-day moving average
            ma_chart.add_trace(go.Scatter(x=recent_daily_data['Time'], y=recent_daily_data['SMA_100'], mode='lines', name='SMA 100', line=dict(color='yellow', width=1)))

            # Plot the 200-day moving average
            ma_chart.add_trace(go.Scatter(x=recent_daily_data['Time'], y=recent_daily_data['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red', width=2)))

            ma_chart.update_layout(title=f'{selected_coin} Moving Averages')
            st.plotly_chart(ma_chart)

            # Goldencross and risk alerts       

            def display_cross_alert(cross_state, risk_alert, label):
                if cross_state == 'golden_cross':
                    st.markdown(f"<strong style='color:green'>{label}: Golden Cross Zone!</strong>", unsafe_allow_html=True)
                elif cross_state == 'death_cross':
                    st.markdown(f"<strong style='color:red'>{label}: Death Cross Zone!</strong>", unsafe_allow_html=True)
                elif risk_alert == 'risk':
                    st.markdown(f"<strong style='color:orange'>{label}: Cross risk detected!</strong>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<strong style='color:blue'>{label}: No significant cross detected.</strong>", unsafe_allow_html=True)


            cross_state, risk_alert = cross_detection(recent_daily_data['SMA_50'].values, recent_daily_data['SMA_200'].values)
            btc_recent_coin_df = coin_data['BTCUSDT'].iloc[-100:]
            btc_cross_state, btc_risk_alert = cross_detection(btc_recent_coin_df['SMA_50'].values, btc_recent_coin_df['SMA_200'].values)

        # Display the cross state and risk alert for the selected coin
            display_cross_alert(cross_state, risk_alert, f"{selected_coin} Cross Zone")

        # Display the cross state and risk alert for BTC
            display_cross_alert(btc_cross_state, btc_risk_alert, "BTC Cross Zone")

        
        else:
                st.warning("The selected coin's data is missing necessary columns for visualization.")
    else:
        st.warning("Please select a coin from the dropdown menu.")

if __name__ == "__main__":
    app()

           

               