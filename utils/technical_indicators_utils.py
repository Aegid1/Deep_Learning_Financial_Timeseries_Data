import os
import pandas as pd
import talib

def clean_technical_indicators(company_names:list[str], columns:list[str], path:str):
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df = pd.read_csv(company_path)

        mask_has_zero = df[columns].eq(0.0).any(axis=1)
        mask_has_null = df[columns].isna().any(axis=1)
        mask_remove = mask_has_zero | mask_has_null

        df_clean = df.loc[~mask_remove].copy()
        df_clean.to_csv(company_path, index=False)


def add_rsi(company_names:list[str], time_period:int, path:str):
    """
        Calculates the Relative Strength Index (RSI) for the given company stock data.

        Parameters:
        - company_name (str): The name of the company (used to locate the stock data).
        - time_period (int): The time period for RSI calculation (e.g., 14).
        - target_start_date (str): From which date on should the RSI be calculated from
        - warmup_period (int): Rows that are also taken into account for calculating the RSI
    """
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        rsi = talib.RSI(df_stock_data["Volume"], timeperiod=time_period) #time_period = 14
        df_stock_data["RSI"] = rsi
        df_stock_data.to_csv(company_path, index=False)


def add_moving_average_convergence_divergence(company_names:list[str], fast_period:int, slow_period:int, signalperiod:int, path:str):
    """
        Calculates the Moving Average Convergence Divergence (MACD) for the given company stock data.

        Parameters:
        - company_name (str): The name of the company (used to locate the stock data).
        - fast_period (int): Fast period for MACD (e.g., 12).
        - slow_period (int): Slow period for MACD (e.g., 26).
        - signalperiod (int): Signal period for MACD (e.g., 9).
    """
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        macd, macdsignal, macdhist = talib.MACD(df_stock_data["Open"], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signalperiod)
        df_stock_data["MACD"] = macd
        df_stock_data["MACDSignal"] = macdsignal
        df_stock_data["MACDHist"] = macdhist
        df_stock_data.to_csv(company_path, index=False)


def add_simple_moving_average(company_names:list[str], time_period:int, path:str):
    """
        Calculates the Simple Moving Average (SMA) for the given company stock data.

        Parameters:
        - company_name (str): The name of the company (used to locate the stock data).
        - time_period (int): The time period for SMA calculation (e.g., 20).
    """
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        sma = talib.SMA(df_stock_data["Open"], timeperiod=time_period) #time_period = 20
        df_stock_data["SMA"] = sma
        df_stock_data.to_csv(company_path, index=False)


def add_exponential_moving_average(company_names:list[str], time_period: int, path:str):
    """
        Calculates the Exponential Moving Average (EMA) for the given company stock data.

        Parameters:
        - company_name (str): The name of the company (used to locate the stock data).
        - time_period (int): The time period for EMA calculation (e.g., 20).
    """
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        ema = talib.EMA(df_stock_data["Open"], timeperiod=time_period)  # time_period = 20
        df_stock_data["EMA"] = ema
        df_stock_data.to_csv(company_path, index=False)


def add_bollinger_baender(company_names:list[str], time_period: int, path:str):
    """
        Calculates the Bollinger Bands for the given company stock data.

        Parameters:
        - company_name (str): The name of the company (used to locate the stock data).
        - time_period (int): The time period for Bollinger Bands calculation (e.g., 20).
    """
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        upperband, middleband, lowerband = talib.BBANDS(df_stock_data["Open"], timeperiod=time_period, nbdevup=2, nbdevdn=2) #20
        df_stock_data["UpperBand"] = upperband
        df_stock_data["MiddleBand"] = middleband
        df_stock_data["LowerBand"] = lowerband
        df_stock_data.to_csv(company_path, index=False)


def add_on_balance_volume(company_names:list[str], path:str):
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        obv = talib.OBV(df_stock_data["Open"], df_stock_data["Volume"])
        df_stock_data["OBV"] = obv
        df_stock_data.to_csv(company_path, index=False)


def add_volume_weighted_average_price(company_names:list[str], path:str):

    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        vwap_values = []
        for i in range(len(df_stock_data)):
            start_idx = max(0, i - 3)
            df_subset = df_stock_data.iloc[start_idx:i]
            vwap = (df_subset["Open"] * df_subset["Volume"]).sum() / df_subset["Volume"].sum()
            vwap_values.append(vwap)

        df_stock_data["VWAP"] = vwap_values
        df_stock_data.to_csv(company_path, index=False)


def add_volume_price_trend(company_names:list[str], path:str):
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        vpt_change = ((df_stock_data["Open"] - df_stock_data["Open"].shift(1)) / df_stock_data["Open"].shift(1)) * df_stock_data["Volume"]
        vpt = vpt_change.cumsum()
        df_stock_data["VPT"] = vpt
        df_stock_data.to_csv(company_path, index=False)


def add_stochastic_oscillator(company_names:list[str], path:str):
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)
        slowk, slowd = talib.STOCH(
            high=df_stock_data["High"],
            low=df_stock_data["Low"],
            close=df_stock_data["Close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        df_stock_data["STOCH_K"] = slowk
        df_stock_data["STOCH_D"] = slowd
        df_stock_data.to_csv(company_path, index=False)


def add_williams_r(company_names:list[str], time_period: int, path:str):
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)

        willr = talib.WILLR(
            high=df_stock_data["High"],
            low=df_stock_data["Low"],
            close=df_stock_data["Close"],
            timeperiod=time_period # 14
        )

        df_stock_data["WILL_R"] = willr
        df_stock_data.to_csv(company_path, index=False)


def add_adosc(company_names:list[str], path:str):
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)

        adosc = talib.ADOSC(
            high=df_stock_data["High"],
            low=df_stock_data["Low"],
            close=df_stock_data["Close"],
            volume=df_stock_data["Volume"],
            fastperiod=3,
            slowperiod=10
        )

        df_stock_data["ADOSC"] = adosc
        df_stock_data.to_csv(company_path, index=False)


def add_cci(company_names:list[str], time_period:int, path:str):
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)

        cci = talib.CCI(
            high=df_stock_data["High"],
            low=df_stock_data["Low"],
            close=df_stock_data["Close"],
            timeperiod=time_period
        )

        df_stock_data["CCI"] = cci
        df_stock_data.to_csv(company_path, index=False)


def add_all_technical_indicators(company_names:list[str], path:str):
    add_volume_price_trend(company_names, path)
    add_volume_weighted_average_price(company_names, path)
    add_on_balance_volume(company_names, path)
    add_bollinger_baender(company_names, 20, path)
    add_exponential_moving_average(company_names, 20, path)
    add_simple_moving_average(company_names, 20, path)
    add_moving_average_convergence_divergence(company_names, 14, 26, 9, path)
    add_rsi(company_names, 14, path)
    clean_technical_indicators(company_names, ["UpperBand", "MiddleBand", "LowerBand", "RSI", "SMA", "EMA", "MACD", "MACDSignal", "MACDHist", "OBV", "VWAP", "VPT"], path)


def add_ohcl_technical_indicators(company_names:list[str], path:str):
    add_stochastic_oscillator(company_names, path)
    add_williams_r(company_names, 14, path)
    add_adosc(company_names, path)
    add_cci(company_names, 14, path)
    add_volume_weighted_average_price(company_names, path)
    add_on_balance_volume(company_names, path)
    add_bollinger_baender(company_names, 20, path)
    add_exponential_moving_average(company_names, 20, path)
    add_moving_average_convergence_divergence(company_names, 14, 26, 9, path)
    add_rsi(company_names, 14, path)
    clean_technical_indicators(company_names, ["ADOSC", "CCI", "WILL_R", "STOCH_D", "STOCH_K", "UpperBand", "MiddleBand", "LowerBand", "RSI", "EMA", "MACD", "MACDSignal", "MACDHist", "OBV", "VWAP"], path)


# calculates technical indicators relative to the price/volatility at the given timestep
def add_relative_technical_indicators(company_names:list[str], path:str):
    for company in company_names:
        company_path = os.path.join(path, company + ".csv")
        df_stock_data = pd.read_csv(company_path)

        df_stock_data["VWAP_p"] = df_stock_data["VWAP"] / df_stock_data["Open"] # VWAP relative to OPEN price
        df_stock_data["EMA_p"] = df_stock_data["EMA"] / df_stock_data["Open"] # EMA relative to OPEN price

        df_stock_data.to_csv(company_path, index=False)


# add_relative_technical_indicators(
#     company_names=["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     path=os.path.join("..", "data", "stocks_test_data_benchmark")
# )

# add_moving_average_convergence_divergence(["BMW"], 14, 26, 9)

# clean_technical_indicators(
#     ["Siemens", "SAP"],
#     ["UpperBand", "MiddleBand", "LowerBand", "RSI", "SMA", "EMA", "MACD", "MACDSignal", "MACDHist", "OBV", "VWAP", "VPT"],
#     os.path.join("..", "data", "stocks_combined_news")
# )

# add_ohcl_technical_indicators(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     os.path.join("..", "data", "stocks_ohlc")
# )


# add_moving_average_convergence_divergence(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     14,
#     26,
#     9,
#     os.path.join("..", "data", "stocks_ohlc"),
#     False
# )
#
# add_exponential_moving_average(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     20,
#     os.path.join("..", "data", "stocks_ohlc"),
#     False
# )
#
# add_rsi(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     14,
#     os.path.join("..", "data", "stocks_ohlc"),
#     False
# )
#
# add_mom(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     10,
#     os.path.join("..", "data", "stocks_ohlc"),
#     False
# )
#
# add_volume_weighted_average_price(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     os.path.join("..", "data", "stocks_ohlc"),
#     False
# )
#
# clean_technical_indicators(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     ["CCI", "ADOSC", "WILL_R", "STOCH_K", "STOCH_D", "RSI", "EMA", "MACD", "VWAP"],
#     os.path.join("..", "data", "stocks_ohlc")
# )

# clean_technical_indicators(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     ["currency_rate_change", "oil_price_change"],
#     os.path.join("..", "data", "stocks_stationary_volume_updated")
# )

# add_all_technical_indicators(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     os.path.join("..", "data", "stocks_test_data_benchmark")
# )

# add_all_technical_indicators(["BMW"], os.path.join("..", "data", "stocks_stationary_volume"))

# clean_technical_indicators(["Adesso", "Adidas", "Airbus", "Allianz", "BASF", "BMW", "Bayer", "Continental", "Deutsche Bank", "Knorr-Bremse", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],["UpperBand", "MiddleBand", "LowerBand", "RSI", "SMA", "EMA", "MACD", "MACDSignal", "MACDHist", "OBV", "VWAP", "VPT"], os.path.join("..", "data", "stocks_stationary"))
#
# clean_technical_indicators(
#     ["Airbus", "Allianz", "BMW", "Deutsche Bank", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     ["UpperBand", "MiddleBand", "LowerBand", "RSI", "SMA", "EMA", "MACD", "MACDSignal", "MACDHist", "OBV", "VWAP", "VPT"],
#     os.path.join("..", "data", "stocks_test_data_benchmark")
# )

# clean_technical_indicators(
#     ["BMW"],
#     ["UpperBand", "MiddleBand", "LowerBand", "RSI", "SMA", "EMA", "MACD", "MACDSignal", "MACDHist", "OBV", "VWAP", "VPT", "currency_rate_change"],
#     os.path.join("..", "data", "stocks_stationary_volume")
# )

#clean_technical_indicators(["Continental"], ["UpperBand", "MiddleBand", "LowerBand", "RSI", "SMA", "EMA", "MACD", "MACDSignal", "MACDHist", "OBV", "VWAP", "VPT"])

# clean_technical_indicators(["Adidas", "BASF", "Bayer"], ["UpperBand", "MiddleBand", "LowerBand", "RSI", "SMA", "EMA", "MACD", "MACDSignal", "MACDHist", "OBV", "VWAP", "VPT"])

# add_moving_average_convergence_divergence(["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"], 14, 26, 9)
# add_volume_price_trend(["Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"])