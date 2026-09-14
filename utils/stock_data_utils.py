import datetime
import os
import pickle
from zoneinfo import ZoneInfo
import numpy as np
from natsort import natsorted
from dateutil import parser
from datetime import timedelta, timezone
import pandas as pd
import yfinance as yf
from pandas import DataFrame

def add_stock_data_to_existing(ticker_symbol:str, start_date:str, end_date:str, retrieving_older_data:bool, path:str, columns):
    """
        Downloads hourly stock data for a given ticker symbol and saves it as weekly CSV files.

        This function uses the Yahoo Finance API to download stock data for a specific ticker symbol between the
        specified start and end dates. It then filters the data to include only specific times of the day (09:00, 11:00,
        13:00, 15:00, and 17:00), adds year and week information, and saves the data in CSV format for each week.

        Parameters:
        - ticker_symbol (str): The stock ticker symbol (e.g., "AAPL" for Apple).
        - start_date (str): The start date for the stock data (format: "YYYY-MM-DD"), INCLUSIVE
        - end_date (str): The end date for the stock data (format: "YYYY-MM-DD"), EXCLUSIVE
        - path (str): The path to the existing data to which the data will be added to
        - retrieving_older_data (bool): describes whether new or older data is going to be retrieved
        - columns (list[str]): Name of the columns of the data which should added
    """
    existing_df = pd.read_csv(path)
    data = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1h")
    data.index = data.index.tz_convert("Europe/Berlin")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    desired_times = ["09:00:00", "11:00:00", "13:00:00", "15:00:00", "17:00:00"]
    filtered_data = data[data.index.strftime('%H:%M:%S').isin(desired_times)]
    filtered_data['Year'] = filtered_data.index.year
    filtered_data['Week'] = filtered_data.index.isocalendar().week
    filtered_data['Datetime'] = filtered_data.index
    final_data = filtered_data[columns]

    # adds log return to the df
    previous_open = final_data["Open"].shift(1)
    previous_open.iloc[0] = existing_df["Open"].iloc[-1] # add previous open value of existing data
    current_open = final_data["Open"]
    stock_return = current_open / previous_open
    stock_log_return = np.log(stock_return)
    final_data["log_return"] = stock_log_return

    # adds log volume change to the df TODO check if this calculation makes sense
    previous_vol = final_data["Volume"].shift(1)
    previous_vol.iloc[0] = existing_df["Volume"].iloc[-1] # add previous open value of existing data
    current_vol = final_data["Volume"]
    volume_change_log = np.log(current_vol + 1) -  np.log(previous_vol + 1)
    final_data["log_volume_change"] = volume_change_log

    # time-mask for intraday
    final_data["time_mask"] = (pd.to_datetime(final_data["Datetime"]).dt.hour - 8) // 2
    # time-mask for inter day
    final_data["time_mask_days"] = pd.to_datetime(final_data["Datetime"]).dt.day

    if retrieving_older_data: # when older data is retrieved, it needs to be included before the newer data
        final_data = pd.concat([final_data, existing_df], ignore_index=True)
    else:
        final_data = pd.concat([existing_df, final_data], ignore_index=True)

    final_data.to_csv(path, index=False)
    print(f"Gespeichert: {path}")


def add_volume_per_hour_to_existing_stock_data(ticker_symbol:str, start:str, end:str, company_name:str):
    """
        Retrieves the volume data of the stocks and appends it to existing stock data.

        Parameters:
        - ticker_symbol (str): The stock ticker symbol (e.g., "AAPL" for Apple).
        - start_date (str): The start date for the volume data (format: "YYYY-MM-DD").
        - end_date (str): The end date for the volume data (format: "YYYY-MM-DD").
        - company_name (str): The name of the company (used to organize the saved files).
    """
    # this part extracts all the relevant .csv files that need to be adjusted with the volume counts
    input_dir = os.path.join('..', 'data', 'stocks')
    company_path = os.path.join(input_dir, company_name)
    csv_files = natsorted([f for f in os.listdir(company_path) if f.endswith('.csv')])
    starting_file = csv_files.index(f"{company_name}_{start}.csv")
    ending_file = csv_files.index(f"{company_name}_{end}.csv")
    csv_files = csv_files[starting_file:ending_file+1]

    for file in csv_files:
        df = pd.read_csv(os.path.join(company_path, file), parse_dates=['Datetime'])
        start_date = parser.isoparse(str(df["Datetime"].iloc[0]))
        end_date = parser.isoparse(str(df["Datetime"].iloc[-1]))
        end_date = end_date + timedelta(days=1)

        start_date_formatted = start_date.strftime('%Y-%m-%d')
        end_date_formatted = end_date.strftime('%Y-%m-%d')

        data = yf.download(
            ticker_symbol,
            start=start_date_formatted,
            end=end_date_formatted,
            interval="1h"
        )

        data.index = data.index.tz_convert("Europe/Berlin")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        desired_times = ["09:00:00", "11:00:00", "13:00:00", "15:00:00", "17:00:00"]
        filtered_data = data[data.index.strftime('%H:%M:%S').isin(desired_times)]
        filtered_data = filtered_data.reset_index()
        final_data = filtered_data[['Datetime', 'Volume']]

        df = df.merge(final_data[["Datetime", "Volume"]], on="Datetime", how="left")
        df.to_csv(os.path.join(company_path, file), index=False)
        print(f"Gespeichert: {file}")


def combine_dataframe_with_stock_dataframe_on_datetime_column(df: DataFrame, company_names: list[str], stocks_data_dir:str):
    """
        Combines a dataframe with the central stock_data dataframe on the column "Datetime".
        Columns of the provided df are appended on the stock_data dataframe.
        The provided df needs to have a column named "Datetime" that needs to be of dtype datetime.

        Parameters:
        - df (DataFrame): dataframe which contains the columns that will be appended
        - company_names (list): A list of company names.
    """

    df = df.sort_values("Datetime").reset_index(drop=True)
    currency_cols = [c for c in df.columns if c != "Datetime"]

    for company in company_names:
        file_path = os.path.join(stocks_data_dir, company + ".csv")
        df_stock_data = pd.read_csv(file_path)
        df_stock_data["Datetime"] = pd.to_datetime(df_stock_data["Datetime"], utc=True).dt.tz_convert(timezone(timedelta(hours=1)))
        df_stock_data = df_stock_data.sort_values("Datetime").reset_index(drop=True)

        mapped_currency = pd.merge_asof(df_stock_data[["Datetime"]], df, on="Datetime", direction="backward")
        for col in currency_cols:
            if col in df_stock_data.columns:
                df_stock_data[col] = mapped_currency[col].combine_first(df_stock_data[col])
            else:
                df_stock_data[col] = mapped_currency[col]

        # add the data were no data is
        max_stock_date = df_stock_data["Datetime"].max()
        future_currency_data = df[df["Datetime"] > max_stock_date]

        df_new = pd.concat([df_stock_data, future_currency_data], ignore_index=True)
        df_new = df_new.sort_values("Datetime").reset_index(drop=True)
        df_new.to_csv(file_path, index=False)


def delete_columns_of_central_stock_data(company_names: list[str], columns: list[str], path: str):
    for company in company_names:
        df_stock_data = pd.read_csv(os.path.join(path, company + ".csv"))
        df_stock_data = df_stock_data.drop(columns, axis=1)
        df_stock_data.to_csv(os.path.join(path, company + ".csv"))


def create_pickle_samples_all_data(company_names:list[str], sample_size:int, stocks_data_dir:str, output_dir:str):
    """
            Creates training samples as pickle files with a specified sequence size.
            So every pickle file contains rows equal to the specified sample_size.
            Every pickle file contains unique rows and thus unique data, there is no data overlap over multiple pickles.

            Parameters:
            - company_names (list): A list of company names.
            - sample_size (int): The amount of rows included in a single pickle file.
            - stocks_data_dir (str): The path to the stock data from which the pickle files are created from.
            - output_dir (str): The path where the pickle samples should be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    for company in company_names:
        output_dir_with_company = os.path.join(output_dir, company)
        df_stock_data = pd.read_csv(os.path.join(stocks_data_dir, company + ".csv"))
        df_stock_data['Datetime'] = pd.to_datetime(df_stock_data['Datetime'])

        print(f"Creating samples for {company} with the columns: {df_stock_data.columns}")

        # every pickle file contains rows equal to the provided sample_size
        for i in range(0, len(df_stock_data) - sample_size, sample_size):
            if i + sample_size <= len(df_stock_data):
                sample = df_stock_data.iloc[i:i + sample_size+1]

                # stock_features = sample['log_return'][:-1].values
                # stock_target = sample['log_return'].iloc[-1]
                datetime_end = sample['Datetime'].iloc[-1]

                # features = { 'stock_prices': stock_features }
                features, targets = { }, { }

                exclude_cols_lower = {'open', 'datetime'}
                for col in df_stock_data.columns:
                    if col.lower() in exclude_cols_lower: continue
                    arr = sample[col].iloc[:-1].values
                    target = sample[col].iloc[-1]

                    key = col.lower().strip()
                    features[key], targets[key] = arr, target

                item = {
                    "company_name": company,
                    'features': features,
                    'targets': targets,
                    'datetime': datetime_end
                }

                os.makedirs(output_dir_with_company, exist_ok=True)
                output_filename = os.path.join(output_dir_with_company, f'{company}_{datetime_end.strftime("%Y-%m-%d_%H-%M-%S")}_size_{sample_size}.pkl')
                with open(output_filename, 'wb') as f:
                    pickle.dump(item, f)


def fix_timezones_of_stock_data(company_names: list[str], path):
    fixed_tz = datetime.timezone(datetime.timedelta(hours=1)) # be aware that this is wrong for german timezone -> winter/summer change +01/+02
    for company in company_names:
        df_stock_data = pd.read_csv(os.path.join(path, company + ".csv"))
        df_stock_data["Datetime"] = pd.to_datetime(df_stock_data["Datetime"], utc=True)
        # df_stock_data["Datetime"] = df_stock_data["Datetime"].dt.tz_convert(ZoneInfo("Europe/Berlin"))
        df_stock_data["Datetime"] = df_stock_data["Datetime"].dt.tz_convert(fixed_tz)
        df_stock_data.to_csv(os.path.join(path, company + ".csv"), index=False)


def filter_df_time_interval_based(date_col, start_date, end_date, path):
    """
    Only keeps the rows that are in the provided time interval
    """
    df_filtered = pd.read_csv(path)
    temp_datetime_col = pd.to_datetime(df_filtered[date_col], utc=True)

    start = pd.to_datetime(start_date, utc=True)
    ende = pd.to_datetime(end_date, utc=True)
    mask = (temp_datetime_col >= start) & (temp_datetime_col <= ende)
    df_filtered = df_filtered.loc[mask]
    df_filtered.to_csv(path, index=False)


# filter_df_time_interval_based(
#     "Datetime",
#     "2026-03-25 09:00:00+01:00",
#     "2026-08-10 17:00:00+02:00",
#     os.path.join("..", "data", "stocks_test_data_benchmark", "Volkswagen.csv")
# )

tickers = ["BMW.DE", "AIR.DE", "ALV.DE", "DBK.DE", "DTE.DE", "MBG.DE", "P911.DE", "SAP.DE", "SIE.DE", "SHL.DE", "VOW3.DE"]
company_names = ["BMW", "Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"]

# for ticker, company_name in zip(tickers, company_names):
#     add_stock_data_to_existing(
#         ticker,
#         "2026-03-25",
#         "2026-08-11",
#         False,
#         os.path.join("..", "data", "stocks_test_data_benchmark", f"{company_name}.csv"),
#         ['Datetime', 'Open', 'Volume', 'Year', 'Week']
#     )

# add_stock_data_to_existing(
#     "AIR.DE",
#     "2025-10-02",
#     "2026-03-25",
#     False,
#     os.path.join("..", "data", "stocks_ohlc", "Airbus.csv"),
#     ['Datetime', 'Open', "High", "Low", "Close", 'Volume', 'Year', 'Week']
# )

# create_pickle_samples_all_data (
#         ["BMW", "Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "Siemens", "SAP", "Siemens Healthineers", "Volkswagen"],
#     23,
#     os.path.join("..", "data", "stocks_stationary_volume"),
#     os.path.join("..", "data", "samples", "train_samples", "stationary_volume_new"),
# )