import os
from datetime import timezone, timedelta, datetime
from enum import Enum
import numpy as np
import yaml
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from ecbdata import ecbdata
import pandas as pd
from stock_data_utils import combine_dataframe_with_stock_dataframe_on_datetime_column


class ECBMacroFactors(Enum):
    central_interest_rate = "FM.B.U2.EUR.4F.KR.MRR_FR.LEV"
    inflation_rate = "ICP.M.DE.N.000000.4.ANR"
    unemployment_rate = "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T"
    gdp = "MNA.Q.N.DE.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N"


class Currency(Enum):
    euro = "EUR"
    usd = "USD"


def add_ecb_macro_factor(initial_start_period: str, end_period: str, company_names, macro_factor: ECBMacroFactors, stock_dir:str):
    """
        Fetches and saves the central interest rate data from the ECB API.

        Parameters:
        - start_period (str): The start period in 'YYYY-MM-DD' format.
        - end_period (str): The end period in 'YYYY-MM-DD' format.
        - company_names (list): A list of company names.

        Data Processing:
        - Fetches data using the ECB API series identifier 'FM.B.U2.EUR.4F.KR.MRR_FR.LEV'.
        - Selects and renames columns for better clarity.
        - Appends the ecb data to the existing stock data of the provided companies (.csv files)
    """
    df = ecbdata.get_series(macro_factor.value, start=initial_start_period, end=end_period)
    df = df[["TIME_PERIOD", "OBS_VALUE"]]
    df = df.rename(columns={"TIME_PERIOD": "Datetime", "OBS_VALUE": macro_factor.name})
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert(timezone(timedelta(hours=1)))
    # if there is no ecb data existing for a given date in the stock data -> gather new ecb data until there is no missing ecb data
    # when counter exceeds the value 12, it seems to be that there is an error with fetching the correct data
    counter = 0
    start_period = initial_start_period

    while df["Datetime"].iloc[0] > pd.to_datetime(initial_start_period, utc=True):
        if counter > 12: raise "Error in fetching the ecb data"

        counter += 1
        start_period = pd.to_datetime(start_period) - pd.Timedelta(days=30)
        # string slicing cuts the time information, so that only the date is present
        df = ecbdata.get_series(macro_factor.value, start=str(start_period)[:10], end=end_period)
        df = df[["TIME_PERIOD", "OBS_VALUE"]]
        df = df.rename(columns={"TIME_PERIOD": "Datetime", "OBS_VALUE": macro_factor.name})
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert(timezone(timedelta(hours=1)))

    combine_dataframe_with_stock_dataframe_on_datetime_column(df, company_names, stock_dir)


def add_currency_rate(start_period: str, end_period: str, from_currency: Currency, to_currency: Currency, company_names: list[str], stock_dir:str):
    """
        Fetches and saves the ratio of a currency to another currency.

        Parameters:
        - start_period (str): The start period in 'YYYY-MM-DD' format.
        - end_period (str): The end period in 'YYYY-MM-DD' format.
        - from_currency (Currency): A currency that is used for the rate (nominator)
        - to_currency (Currency): Another currency that is used for the rate (denominator)
        - company_names (list): A list of company names.
    """
    config = yaml.safe_load(open("../openai_config.yaml"))
    api_key = config['KEYS']['alpha_vantage']
    fx = ForeignExchange(key=api_key, output_format='json')

    data, meta_data = fx.get_currency_exchange_daily(from_symbol=from_currency.value, to_symbol=to_currency.value, outputsize='full')

    start_date = datetime.strptime(start_period, '%Y-%m-%d')
    end_date = datetime.strptime(end_period, '%Y-%m-%d')

    filtered_data = []
    for date_str, values in data.items():
        date = datetime.strptime(date_str, '%Y-%m-%d')

        if start_date <= date <= end_date:
            open_value = values['1. open']
            filtered_data.append([date_str, open_value])

    df = pd.DataFrame(filtered_data, columns=["Date", "Open"])
    df = df.rename(columns={"Date": "Datetime", "Open": from_currency.value + "_" + to_currency.value})
    df = df.sort_values(by="Datetime", ascending=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert(timezone(timedelta(hours=1)))

    combine_dataframe_with_stock_dataframe_on_datetime_column(df, company_names, stock_dir)



#TODO: change such that it actually fetches the BRENT oil price (Europe)
def add_oil_prices(start_period: str, end_period: str, company_names: list[str], stock_dir:str):
    """
    Fetches and appends daily WTI oil price data to the central stock data df.

    Parameters:
    - start_period (str): The start period in 'YYYY-MM-DD' format.
    - end_period (str): The end period in 'YYYY-MM-DD' format.
    - company_names (list): A list of company names.

    """
    config = yaml.safe_load(open("../openai_config.yaml"))
    api_key = config['KEYS']['alpha_vantage']
    ts = TimeSeries(key=api_key, output_format='json')
    data, meta_data = ts.get_daily(symbol="WTI", outputsize='full')

    start_date = datetime.strptime(start_period, '%Y-%m-%d')
    end_date = datetime.strptime(end_period, '%Y-%m-%d')

    filtered_data = []
    for date_str, values in data.items():
        date = datetime.strptime(date_str, '%Y-%m-%d')

        if start_date <= date <= end_date:
            open_value = values['1. open']
            filtered_data.append([date_str, open_value])

    df = pd.DataFrame(filtered_data, columns=["Date", "Open"])
    df = df.rename(columns={"Date": "Datetime", "Open": "oil_price"})
    df = df.sort_values(by="Datetime", ascending=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert(timezone(timedelta(hours=1)))

    combine_dataframe_with_stock_dataframe_on_datetime_column(df, company_names, stock_dir)


# Note: assumes that all the stock data of the companies start with 8:00/9:00
def add_oil_price_changes(company_names: list[str], stocks_data_dir: str):
    for company in company_names:
        path = os.path.join(stocks_data_dir, company + ".csv")
        df_stock_data = pd.read_csv(path)

        current = df_stock_data["oil_price"].shift(5)[5:]
        previous = df_stock_data["oil_price"][:-5]
        oil_price_change = current / previous

        df_stock_data["oil_price_change"] = np.log(oil_price_change)
        df_stock_data.to_csv(path, index=False)


# Note: assumes that all the stock data of the companies start with 8:00/9:00
def add_currency_rate_changes(company_names: list[str], stocks_data_dir: str):
    for company in company_names:
        path = os.path.join(stocks_data_dir, company + ".csv")
        df_stock_data = pd.read_csv(path)

        current = df_stock_data["EUR_USD"].shift(5)[5:]
        previous = df_stock_data["EUR_USD"][:-5]
        oil_price_change = current / previous

        df_stock_data["currency_rate_change"] = np.log(oil_price_change)
        df_stock_data.to_csv(path, index=False)


def add_previous_values_for_every_macro_factor(companies: list[str], stock_dir:str):
    macro_factors = ["central_interest_rate", "gdp", "unemployment_rate", "inflation_rate"]
    for company in companies:
        path = os.path.join(stock_dir, f"{company}.csv")
        df = pd.read_csv(path)
        for factor in macro_factors:
            unique_vals = pd.Series(df[factor].unique())
            unique_vals_shifted = pd.Series(df[factor].unique()).shift(1)
            unique_vals_shifted.iloc[0] = unique_vals_shifted.iloc[1]

            mapping = dict(zip(unique_vals.values, unique_vals_shifted.values))
            df[f"{factor}_previous"] = df[factor].map(mapping)

        df.to_csv(path, index=False)


def add_all_existing_macrofactors(companies:list[str], start_period: str, end_period: str, stock_dir:str):
    add_oil_prices(start_period, end_period, companies, stock_dir)
    add_ecb_macro_factor(start_period, end_period, companies, ECBMacroFactors.central_interest_rate, stock_dir)
    add_ecb_macro_factor(start_period, end_period, companies, ECBMacroFactors.gdp, stock_dir)
    add_ecb_macro_factor(start_period, end_period, companies, ECBMacroFactors.unemployment_rate, stock_dir)
    add_ecb_macro_factor(start_period, end_period, companies, ECBMacroFactors.inflation_rate, stock_dir)
    add_currency_rate(start_period, end_period, Currency.euro, Currency.usd,companies, stock_dir)


#add_all_existing_macrofactors(["Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"], "2022-11-07", "2024-10-30")

# add_currency_rate(
#     "2025-09-03",
#     "2026-03-24",
#     Currency.euro,
#     Currency.usd,
#     ["BMW", "Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     os.path.join("..", "data", "stocks_ohlc")
# )

# add_currency_rate(
#     "2025-09-03",
#     "2026-03-24",
#     Currency.euro,
#     Currency.usd,
#     ["BMW"],
#     os.path.join("..", "data", "stocks_stationary_volume")
# )

# add_oil_prices(
#     "2023-10-13",
#     "2025-10-01",
#     ["BMW", "Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     os.path.join("..", "data", "stocks_ohlc")
# )

# add_oil_prices(
#     "2025-09-03",
#     "2026-03-24",
#     ["BMW"],
#     os.path.join("..", "data", "stocks_stationary_volume")
# )

# add_currency_rate_changes(
#     ["BMW", "Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     os.path.join("..", "data", "stocks_stationary_volume_updated")
# )

# add_currency_rate_changes(
#     ["BMW"],
#     os.path.join("..", "data", "stocks_stationary_volume")
# )

# add_oil_price_changes(
#     ["BMW", "Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"],
#     os.path.join("..", "data", "stocks_stationary_volume_updated")
# )