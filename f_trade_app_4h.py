import os
import pandas as pd
import pandas_ta as pta
import streamlit as st
from f_trade_app_main import get_daily_rsi_values


def get_rsi_color_and_weight(rsi_value):
    if rsi_value < 20:
        return 'green', 'bold'
    elif 20 <= rsi_value < 45:
        return 'blue', 'bold'
    elif 45 <= rsi_value < 60:
        return 'orange', 'bold'
    else:
        return 'red', 'bold'


def get_adx_color_and_weight(adx_value):
    if adx_value < 20:
        return 'green', 'bold'
    elif 20 <= adx_value < 35:
        return 'blue', 'bold'
    elif 35 <= adx_value < 45:
        return 'orange', 'bold'
    else:
        return 'red', 'bold'
    
def get_rsi_adx_color(rsi_value, adx_value):
    if rsi_value < 20 and adx_value >= 30:
        return '<span style="font-size: 1.5em; font-weight: bold;">🚀</span>', 'green'
    elif rsi_value < 20 and 0 < adx_value <= 29:
        return '🍀⬆️', 'blue'
    elif 21 <= rsi_value < 45 and adx_value >= 30:
        return '🍀⬆️', 'blue'
    elif 21 <= rsi_value < 45 and 0 < adx_value <= 29:
        return '🌙⬇️', 'orange'
    elif 46 <= rsi_value < 60 and adx_value >= 25:
        return '🌙⬇️', 'orange'
    elif 46 <= rsi_value < 60 and 0 < adx_value < 25:
        return '🍀⬆️', 'blue'
    elif rsi_value > 60:
        return '🔥', 'red'
    else:
        return 'N/A', 'black'  # Default value
    
def get_sto_median_color(sto_upwards, median_cross, sto_k1, ema_sto_value_4h):
    if sto_upwards and median_cross:
        if sto_k1 < 21:
            if sto_k1 > ema_sto_value_4h:
                return '🚀🚀🚀', 'green'
            else:
                return '🚀', 'green'
        else:
            return '🍀⬆️', 'blue'
    elif sto_upwards and not median_cross:
        if sto_k1 < 21:
            return '🍀⬆️', 'blue'
        elif 21 <= sto_k1 < 80:
            return '🌙⬇️', 'orange'
        else:
            return '🔥', 'red'
    elif not sto_upwards and median_cross:
        if sto_k1 >= 80:
            return '🌙⬇️', 'orange'
        else:
            return '🍀⬆️', 'blue'
    else:  # not sto_upwards and not median_cross
        if sto_k1 > ema_sto_value_4h:
            return '🔥🔥', 'red'
        else:
            return '🔥', 'red'
        
# Tilson-Alma  fisher comparaisons

def get_Tilson_Fisher_color_and_icon(tilson_alma_condition,fisher_condition ):
    
    if tilson_alma_condition and fisher_condition:
        return '🚀', 'green'
    elif fisher_condition and not tilson_alma_condition:
        return '🍀⬆️', 'blue'
    elif tilson_alma_condition and not fisher_condition:
        return '🌙⬇️', 'orange'
    else:
        return '🔥', 'red'


def get_SMII_VWMA_color(price, EMA_price_9_4h, SMI_Ergodic, VWMA_20_SMII):
    if price > EMA_price_9_4h and SMI_Ergodic > VWMA_20_SMII:
        return 'green', '🚀'
    elif price > EMA_price_9_4h and SMI_Ergodic < VWMA_20_SMII:
        return 'blue', '🍀⬆️'
    elif price < EMA_price_9_4h and SMI_Ergodic > VWMA_20_SMII:
        return 'orange', '🌙⬇️'
    elif price < EMA_price_9_4h and SMI_Ergodic < VWMA_20_SMII:
        return 'red', '🔥'
    else:
        return 'N/A', 'N/A'

def app():

    daily_rsi_values = get_daily_rsi_values()
    daily_rsi_df = pd.DataFrame(list(daily_rsi_values.items()), columns=['Symbol', 'RSI Daily'])

    directory = "C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/binance_4h_coin_datasets/"
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    dataframes = {}
    

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(directory, csv_file))
        df['symbol'] = csv_file.split('_binance')[0]
        dataframes[csv_file] = df
        

    coin_screen_df = pd.DataFrame(columns=['Coin', 'RSI_daily', 'RSI', 'ADX', 'RSI/ADX', 'sto_ema_hl_median', 
                                           'Alma_Tilson_Fisher', 'Smii_Vwma_Ema','Total_points'])
    coin_symbol_style_df = pd.DataFrame(columns=['RSI_color', 'RSI_weight', 'ADX_color', 'ADX_weight', 'Symbol_color', 
                                                 'sto_ema_hl_median_color', 'Alma_Tilson_Fisher_color', 'Smii_Vwma_Ema_color', 
                                                 'daily_RSI_color', 'daily_RSI_weight'])

    color_points = {'green': 1, 'blue': 2, 'orange': 3, 'red': 4}

    for csv_file, coin_4h_data in dataframes.items():
        unique_symbols = coin_4h_data['symbol'].unique()
        for coin_symbol in unique_symbols:
            indicators_4h = coin_4h_data[coin_4h_data['symbol'] == coin_symbol]

            rsi_value_4h = indicators_4h['RSI_4h'].iloc[-1]
            adx_value_4h = indicators_4h['ADX_14'].iloc[-1]
            rsi_value_daily = daily_rsi_values.get(coin_symbol, None) #coin symbol is enough to find the good coin

            rsi_color, rsi_weight = get_rsi_color_and_weight(rsi_value_4h)
            adx_color, adx_weight = get_adx_color_and_weight(adx_value_4h)
            daily_rsi_color, daily_rsi_weight = get_rsi_color_and_weight(rsi_value_daily)

            icon, icon_color = get_rsi_adx_color(rsi_value_4h, adx_value_4h)

            
            ema_sto_value_4h = pta.ema(indicators_4h['STOCHRSIk_14_14_3_3'], length=9).iloc[-1]
            hl2_value_4h = indicators_4h['hl2'].iloc[-1]
            med_value_4h = indicators_4h['median_hl2_14'].iloc[-1] 
            sto_k1 = indicators_4h['STOCHRSIk_14_14_3_3'].iloc[-1]
            sto_d1 = indicators_4h['STOCHRSId_14_14_3_3'].iloc[-1]

            # Calculate Stochastic RSI %K and %D / in fact no need to know about trend line sto_k1>sto_d1 is enough
        
            sto_upwards=  sto_k1 >= sto_d1 #upwards trend
            #sto_downwards= sto_k2 > sto_d2

            median_cross= med_value_4h>hl2_value_4h

            sto_median_icon, sto_median_color = get_sto_median_color(sto_upwards, median_cross, sto_k1, ema_sto_value_4h)

            #Alma/Tilson - fisher conditions
        #
            tilson_alma_condition = indicators_4h['T3_10_0.7'].iloc[-1] > indicators_4h['ALMA_9_08_6'].iloc[-1]

            fisher_condition = indicators_4h['FISHERT_9_1'].iloc[-1] > indicators_4h['FISHERTs_9_1'].iloc[-1]

            Tilson_Fisher_icon, Tilson_Fisher_color = get_Tilson_Fisher_color_and_icon(tilson_alma_condition,fisher_condition )

            # SMII _VWMA / EMA 9 control

            price_4h= indicators_4h['price'].iloc[-1]
            ema9_4h= indicators_4h['EMA_price_9_4h'].iloc[-1]
            smi_4h=indicators_4h['SMI_Ergodic'].iloc[-1]
            vwma_4h= indicators_4h['VWMA_20_SMII'].iloc[-1]

            smii_color, smii_icon = get_SMII_VWMA_color(price_4h,ema9_4h, smi_4h, vwma_4h )  

            
            # Calculate the total points
            total_points = color_points[rsi_color] + color_points[adx_color]+color_points[sto_median_color]+color_points[Tilson_Fisher_color]+color_points[smii_color]


            # Add style information to coin_symbol_style_df
            coin_symbol_style_df = pd.concat(
                [
                    coin_symbol_style_df,
                    pd.DataFrame(
                        {
                            'RSI_color': [rsi_color],
                            'RSI_weight': [rsi_weight],
                            'ADX_color': [adx_color],
                            'ADX_weight': [adx_weight],
                            'Symbol_color': [icon_color],
                            'sto_ema_hl_median_color': [sto_median_color],  # stoch_median color,
                            'Tilson_Fisher_color': [Tilson_Fisher_color], #Alma_Tilson_Ficher color
                            'Smii_Vwma_Ema_color': [smii_color],
                            'daily_RSI_color': [daily_rsi_color],
                            'daily_RSI_weight': [daily_rsi_weight]
                                }
                                ),
                ],
                ignore_index=True,
                    )
                 # Add coin information to coin_screen_df

            coin_screen_df = pd.concat(
                [
                    coin_screen_df,
                    pd.DataFrame(
                        {
                            'Coin': [coin_symbol],
                            'RSI_daily': rsi_value_daily,
                            'RSI': [str(round(rsi_value_4h, 1)).rstrip('0').rstrip('.')],
                            'ADX': [str(round(adx_value_4h, 1)).rstrip('0').rstrip('.')],
                            'RSI/ADX': [icon],
                            'sto_ema_hl_median': [sto_median_icon],  # stoch_median symbol,
                            'Alma_Tilson_Fisher': [Tilson_Fisher_icon], # Alma_Tilson Ficher icon
                            'Smii_Vwma_Ema':[smii_icon],
                            'Total_points': [total_points]               
                         
                                 }
                                ),
                            ],
                            ignore_index=True, 
                        )


    # Sort the dataframe by total points
    coin_screen_df = coin_screen_df.sort_values(by='Total_points', ascending=True)

    def style_specific_cell(x):
        df = pd.DataFrame('', index=x.index, columns=x.columns)
        for i in range(len(coin_symbol_style_df)):
            df.loc[i, 'RSI'] = f"color:{coin_symbol_style_df.loc[i, 'RSI_color']}; font-weight:{coin_symbol_style_df.loc[i, 'RSI_weight']}"
            df.loc[i, 'ADX'] = f"color:{coin_symbol_style_df.loc[i, 'ADX_color']}; font-weight:{coin_symbol_style_df.loc[i, 'ADX_weight']}"
            df.loc[i, 'RSI/ADX'] = f"color:{coin_symbol_style_df.loc[i, 'Symbol_color']}"
            df.loc[i, 'sto_ema_hl_median'] = f"color:{coin_symbol_style_df.loc[i, 'sto_ema_hl_median_color']}"  # stoch_median column
            df.loc[i, 'Alma_Tilson_Fisher']= f"color:{coin_symbol_style_df.loc[i, 'Alma_Tilson_Fisher_color']}"
            df.loc[i, 'Smii_Vwma_Ema']= f"color:{coin_symbol_style_df.loc[i, 'Smii_Vwma_Ema_color']}"
            df.loc[i, 'RSI_daily'] = f"color:{coin_symbol_style_df.loc[i, 'daily_RSI_color']}; font-weight:{coin_symbol_style_df.loc[i, 'daily_RSI_weight']}"

        return df
            # Display the styled dataframe
   
    st.title('Cryptocurrency Indicators')

    st.write(coin_screen_df.style.apply(style_specific_cell, axis=None).to_html(), unsafe_allow_html=True)
    st.write(f"Color count: {color_points}")



if __name__=='__main__':
    app()



