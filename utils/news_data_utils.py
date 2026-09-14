import os
import re
import uuid
from datetime import date, timedelta, datetime
import time
from zoneinfo import ZoneInfo
import pandas as pd
import pytz
import requests
import yaml
from dateutil.relativedelta import relativedelta
from openai import OpenAI
from bs4 import BeautifulSoup

config = yaml.safe_load(open("openai_config.yaml"))
client = OpenAI(api_key=config['KEYS']['openai'])

def store_all_relevant_articles_from_news_api(start_date: str, end_date: str, up_to_page_number: int, company_name: str, is_company_given: bool):
    """
        Can only be used when start_date and end_date are of the same month
    Args:
        start_date:
        end_date:
        up_to_page_number:
        company_name:
        is_company_given:
    """
    start_time = time.time()
    dates = create_monthly_intervals(start_date, end_date)

    # if news of a specific company are desired, when this is added more precise results are given
    topic = f"{company_name} Unternehmen" if is_company_given else company_name

    # iterates over all possible dates in the range to retrieve news
    for date_tuple in dates:
        articles_list = []
        days = get_open_days(date_tuple[0], date_tuple[1])
        if len(days) < 2: continue

        # loads the existing data
        _, month, year = date_tuple[0].split("/")
        path = f"../data/news/raw_news/{company_name}/{company_name}_sorted_{month}_{year}.csv"
        if os.path.exists(path):
            existing_df = pd.read_csv(path, header=None, names=['UUID', 'Date', 'Text'])
        else:
            existing_df = pd.DataFrame([], columns=['UUID', 'Date', 'Text'])

        print(f"DAY: {days[0]} to {days[1]}")
        current_page = 1
        response = get_articles(topic, current_page, days[0], days[1])

        # iterates over all existing pages
        while response.get("count") != 0:
            articles = response.get("news")
            print("PAGE: " + str(current_page))
            if articles: # if there are articles
                for article in articles:
                    content = article.get("text")
                    content = re.sub(r'\s+', ' ', content).strip()  # news texts contain unnecessary newlines

                    if not check_if_article_is_relevant(content, company_name): continue # sort unrelevant out

                    date = article.get("date")
                    try:
                        if "08:00:00" in date or "07:00:00" in date:  # the used news api sometimes doesn't provide the actual date and uses 08:00:00 or 07:00:00 as a placeholder
                            date = get_actual_date_of_article(article.get("url"), date)
                        converted_date = pd.to_datetime(date, format='%a, %d %b %Y %H:%M:%S %Z')

                    except Exception: # when exception is thrown it's not important to correctly resolve it
                        continue

                    articles_list.append((str(uuid.uuid4()), converted_date, content))

            current_page += 1
            if current_page > up_to_page_number: break
            response = get_articles(topic, current_page, days[0], days[1])

        # append the new data to the existing data
        new_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        articles_df = combined_df.drop_duplicates(subset='Text')
        # articles_sorted = articles_df.sort_values(by='Date')
        articles_df.to_csv(path, index=False, header=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Die Funktion hat {duration:.2f} Sekunden gebraucht.")


def check_if_article_is_relevant(content: str, keyword: str):
    if keyword.lower() not in content.lower(): return False # check whether the content contains the topic
    return True


def get_articles(topic: str, page_number: int, start_date: str, end_date: str):
    """
        Get articles from the News API based on the provided topic and date range.

        Parameters:
            topic (str): The topic to search for.
            page_number (int): The page number to retrieve.
            start_date (str): The start date for the search (DD/MM/YYYY).
            end_date (str): The end date for the search (DD/MM/YYYY).

        Returns:
            dict: The response from the News API as a JSON object.
    """
    url = "https://newsnow.p.rapidapi.com/newsv2"
    headers = {
        "x-rapidapi-key": config["KEYS"]["rapid-api"],
        "x-rapidapi-host": "newsnow.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    #maybe test with variable amount of pages
    payload = {
        "query": topic,
        "time_bounded": True,
        "from_date": start_date,
        "to_date": end_date,
        "location": "de",
        "language": "de",
        "page": page_number
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()


def get_open_days(start_date_str, end_date_str):
    """
    Determines the list of weekdays (Monday to Friday) that are not holidays between two given dates.

    This function calculates the weekdays (excluding holidays) between a start and end date,
    considering both fixed holidays (e.g., New Year's Day, Christmas) and specific holiday dates.

    Parameters:
    - start_date_str (str): The start date in the format "DD/MM/YYYY".
    - end_date_str (str): The end date in the format "DD/MM/YYYY".

    Returns:
    - open_days (list of str): A list of strings representing the weekdays (in "DD/MM/YYYY" format) that are not holidays.
    """
    start_date = datetime.strptime(start_date_str, "%d/%m/%Y").date()
    end_date = datetime.strptime(end_date_str, "%d/%m/%Y").date()

    holidays = [ # always check for correct holidays
        date(start_date.year, 1, 1),  # Neujahrstag
        date(start_date.year, 4, 3),  # Karfreitag
        date(start_date.year, 4, 6),  # Ostermontag
        date(start_date.year, 5, 1),  # Tag der Arbeit
        date(start_date.year, 5, 14),  # Christi Himmelfahrt
        date(start_date.year, 5, 25),  # Pfingstmontag
        date(start_date.year, 10, 3),  # Tag der Deutschen Einheit
        date(start_date.year, 12, 25),  # Erster Weihnachtstag
        date(start_date.year, 12, 26)  # Zweiter Weihnachtstag
    ]

    open_days = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 6 and current_date not in holidays:
            open_days.append(current_date.strftime("%d/%m/%Y"))
        current_date += timedelta(days=1)

    return open_days


def get_actual_date_of_article(url:str, placeholder_date:str):
    """
        Extracts the actual publication date of an article from a given URL.

        This function tries to extract the publication date from various HTML elements, such as meta tags and time tags.
        If no date is found in these tags, it attempts to infer the date from the text content by extracting the time
        and combining it with a placeholder date. The final result is returned in a standardized UTC format.

        Parameters:
        - url (str): The URL of the article from which the publication date is to be extracted.
        - placeholder_date (str): A fallback date (in ISO format) used when the actual publication time cannot be determined.

        Returns:
        - str: The publication date in the format "Day, DD Mon YYYY HH:MM:SS GMT" in UTC time zone. If the date cannot be found or parsed, the original date string is returned.

        Raises:
        - ValueError: If no valid timestamp can be found or parsed from the article's content.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        raise ValueError("No timestamp found")

    date = None

    # try to extract the date from <meta> tag
    meta_date = soup.find('meta', {'property': 'article:published_time'})
    if meta_date:
        date = meta_date.get('content')

    if not date:
        meta_date_fallback = soup.find('meta', {'name': 'date'}) or soup.find('meta', {'property': 'datePublished'})
        if meta_date_fallback:
            date = meta_date_fallback.get('content')

    #try to extract the date from <time> tag
    if not date:
        time_tag = soup.find('time')
        if time_tag:
            date = time_tag.get('datetime')

    if not date:
        text = soup.get_text()
        time_match = re.search(r'\b(\d{2}:\d{2}(?::\d{2})?)\b', text)  #searches in text for time format 'HH:MM' oder 'HH:MM:SS'
        if time_match:
            time = time_match.group(0)
            placeholder_date_obj = datetime.fromisoformat(placeholder_date)
            placeholder_date_iso = placeholder_date_obj.strftime("%Y-%m-%d")
            date = f"{placeholder_date_iso} {time}"  # Format: 'YYYY-MM-DD HH:MM[:SS]'

    if not date or not re.search(r'\d{2}:\d{2}(:\d{2})?', date):
        #date = placeholder_date
        raise ValueError("No timestamp found")

    try:
        date_obj = datetime.fromisoformat(date)
        # transform in UTC, if necessary
        date_utc = date_obj.astimezone(pytz.UTC)
        formatted_date = date_utc.strftime("%a, %d %b %Y %H:%M:%S GMT")
        return formatted_date
    except ValueError:
        return date


def merge_articles(company_name:str, month:str, year:str):
    """
        The idea is, to merge general news articles with company specific news, because
        general news can have different meanings to different companies.
        Merges news articles for a specific company and month into a single dataset.

        This function reads two CSV files containing news articles: one for general economic news and one for the specific
        company. It merges the articles, cleans up the dates, filters the articles by the specified time period, and then
        saves the merged dataset into a new CSV file.

        Parameters:
        - company_name (str): The name of the company whose articles are to be merged.
        - month (str): The month (in "MM" format) for which the articles are being processed.
        - year (str): The year (in "YYYY" format) for which the articles are being processed.

        Workflow:
        1. Reads the economic news and company-specific news articles for the given month and year from CSV files.
        2. Merges the two datasets into a single DataFrame.
        3. Converts the date column to a datetime format, ensuring no time zone information is retained.
        4. Filters the articles to include only those within the specified time period (from the 1st to the last day of the month).
        5. Sorts the articles by date and resets the index.
        6. Saves the resulting dataset as a CSV file named after the company and the specified month/year.
    """
    df1 = pd.read_csv(f"../data/news/raw_news/Wirtschaft/Wirtschaft_sorted_{month}_{year}.csv", header=None, names=["ID", "Datum", "Text"])
    df2 = pd.read_csv(f"../data/news/raw_news/{company_name}/{company_name}_sorted_{month}_{year}.csv", header=None, names=["ID", "Datum", "Text"])

    merged_df = pd.concat([df1, df2], ignore_index=True)

    merged_df['Datum'] = pd.to_datetime(merged_df['Datum'], errors='coerce')
    merged_df['Datum'] = merged_df['Datum'].dt.tz_localize(None)

    #delete all articles that are not relevant to the given time period
    start_date = pd.Timestamp(year=int(year), month=int(month), day=1)
    if start_date.month == 12:  # December -> January as end date
        end_date = pd.Timestamp(year=start_date.year + 1, month=1, day=1) - pd.Timedelta(days=1)
    else:  # any other month
        end_date = pd.Timestamp(year=start_date.year, month=start_date.month + 1, day=1) - pd.Timedelta(days=1)
    merged_df = merged_df[(merged_df['Datum'] >= start_date) & (merged_df['Datum'] <= end_date)]

    merged_df = merged_df.sort_values(by='Datum').reset_index(drop=True)
    merged_df.to_csv(f"../data/news/news_merged/{company_name}/{company_name}_merged_{month}_{year}.csv", index=False, header=False)


def create_monthly_intervals(start_date: str, end_date: str):
    """
        Creates a list of monthly intervals between the given start and end dates.

        This function generates a series of monthly intervals between the specified start and end dates. Each interval is
        represented as a tuple with a start date and an end date.

        Parameters:
        - start_date (str): The start date of the range in "DD/MM/YYYY" format.
        - end_date (str): The end date of the range in "DD/MM/YYYY" format.

        Returns:
        - list: A list of tuples where each tuple contains the start and end dates of a month in the format "DD/MM/YYYY".

        Example:
        - If the start date is "01/01/2023" and the end date is "31/03/2023", the function will return:
          [("01/01/2023", "31/01/2023"), ("01/02/2023", "28/02/2023"), ("01/03/2023", "31/03/2023")]
        """
    start_date = datetime.strptime(start_date, "%d/%m/%Y")
    end_date = datetime.strptime(end_date, "%d/%m/%Y")

    dates = []
    current_start = end_date
    while current_start > start_date:
        current_end = current_start
        current_start = current_start - relativedelta(days=1)
        dates.append((current_start.strftime("%d/%m/%Y"), current_end.strftime("%d/%m/%Y")))

    return dates


def add_sentiments_to_combined_data(company_names: list[str], input_dir:str, output_dir:str):
    """
        Creates the sentiment features from the sentiments data and adds them to the already available data
    Args:
        company_names (list[str]): A list of company names
        input_dir (str): The input directory where the news sentiments data is located
        output_dir (str): The output directory to which the sentiment features will be added to
    """
    for company in company_names:
        print(company)
        company_path = os.path.join(input_dir, company)
        csv_files = sorted([f for f in os.listdir(company_path) if f.endswith('.csv')])

        # load already available data in a df
        combined_df = pd.read_csv(os.path.join(output_dir, f"{company}.csv"), parse_dates=['Datetime'])
        combined_df["Datetime"] = pd.to_datetime(combined_df["Datetime"], utc=True).dt.tz_convert(ZoneInfo("Europe/Berlin"))
        combined_df = combined_df.reset_index(drop=False).rename(columns={'index': 'idx'})
        # drop the columns as they will be added again with the new values -> efficiency?
        combined_df = combined_df.drop(columns=['long_term_impact', "short_term_impact", "short_term_std", "long_term_std", "count"], errors="ignore")

        combined_df["short_term_impact"], combined_df["long_term_impact"] = 0.0, 0.0 # initialize with zeros
        combined_sorted = combined_df.rename(columns={'Datetime': 'merge_key'}).sort_values('merge_key').reset_index(drop=True)
        sentiments = pd.DataFrame()

        # iterate over all files containing the sentiments and add them to the sentiments df
        for file in csv_files:
            file_path = os.path.join(company_path, file)
            df = pd.read_csv(file_path, parse_dates=['date'])

            df['short_term_sentiment'] = pd.to_numeric(df.get('short_term_sentiment', 0.0), errors='coerce').fillna(0.0)
            df['long_term_sentiment']  = pd.to_numeric(df.get('long_term_sentiment', 0.0), errors='coerce').fillna(0.0)
            df['relevancy_factor']     = pd.to_numeric(df.get('relevancy_factor', 0.0), errors='coerce').fillna(0.0)

            df["short_term_impact"] = df["short_term_sentiment"] * df["relevancy_factor"]
            df["long_term_impact"] = df["long_term_sentiment"] * df["relevancy_factor"]
            df['merge_key'] = df['date'].apply(compute_merge_key)

            # sentiments_new = df[['merge_key', 'short_term_impact', 'long_term_impact']]
            sentiments_new = df[['merge_key', 'short_term_impact', 'long_term_impact']].copy()
            sentiments_new[['short_term_impact', 'long_term_impact']] = sentiments_new[['short_term_impact', 'long_term_impact']].clip(lower=-10, upper=10)

            # sentiments_new = sentiments_new.loc[ # maybe clip the values to be between [-10, 10] instead
            #     (sentiments_new['short_term_impact'].between(-10, 10)) &
            #     (sentiments_new['long_term_impact'].between(-10, 10))
            # ]
            sentiments_new = sentiments_new.sort_values('merge_key').reset_index(drop=True)
            sentiments_new["merge_key"] = pd.to_datetime(sentiments_new["merge_key"], utc=True).dt.tz_convert(ZoneInfo("Europe/Berlin"))
            sentiments = pd.concat([sentiments, sentiments_new], ignore_index=True)

        # add the sentiments data to the already available data
        sentiments = sentiments.sort_values('merge_key').reset_index(drop=True)
        agg = sentiments.groupby('merge_key').agg(
            short_term_impact=('short_term_impact', 'mean'),
            short_term_std=('short_term_impact', 'std'),
            long_term_impact=('long_term_impact', 'mean'),
            long_term_std=('long_term_impact', 'std'),
            count=('short_term_impact', 'count')
        )

        result = pd.merge_asof(
            combined_sorted,
            agg,
            on='merge_key',
            direction='forward',
            tolerance=pd.Timedelta(hours=2)
        ).fillna(0.0)

        result = result.reset_index()  # idx wieder als Spalte
        result = result.drop(columns=['index', "idx", "short_term_impact_x", "long_term_impact_x"])
        #result = result.drop(columns=['index', "idx"])
        result = result.rename(columns={'merge_key': 'Datetime', "short_term_impact_y": "short_term_impact", "long_term_impact_y": "long_term_impact"})
        result.to_csv(os.path.join(output_dir, f"{company}.csv"), index=False)


def get_raw_news_as_single_df(path:str):
    file_names = os.listdir(path)
    combined_df = pd.DataFrame()
    for file in file_names:
        df = pd.read_csv(os.path.join(path, file), header=None)
        df.columns =  ["document_id", "datetime", "sentiment"]
        combined_df = pd.concat([combined_df, df], ignore_index=True).reset_index(drop=True)

    combined_df.columns = ["document_id", "datetime", "sentiment"]
    return combined_df


def get_sentiments_from_combined_data(company_names: list[str]):
    for company in company_names:
        print(company)
        df_raw_news = get_raw_news_as_single_df(os.path.join('..', 'data', "news", 'raw_news', company))
        file_names = os.listdir(os.path.join('..', 'data', "news", "sentiments_combined_with_economy", company))
        # iterate through all .csv files with sentiments
        for file in file_names:
            df = pd.read_csv(os.path.join('..', 'data', "news", "sentiments_combined_with_economy", company, file))
            mask = df['document_id'].isin(df_raw_news['document_id'])
            df_filtered = df.loc[mask].copy()
            df_filtered = df_filtered[["document_id", "date", "short_term_sentiment", "long_term_sentiment", "relevancy_factor"]]
            df_filtered.to_csv(os.path.join('..', 'data', "news", "sentiments", company, file), index=False)


def add_national_economy_sentiments(company_names: list[str]):
    input_dir = os.path.join('..', 'data', "news", 'sentiments', "Wirtschaft")
    output_dir = os.path.join('..', 'data', 'stocks')
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])

    for company in company_names:
        combined_df = pd.read_csv(os.path.join(output_dir, f"{company}.csv"), parse_dates=['Datetime'])
        combined_df["Datetime"] = pd.to_datetime(combined_df["Datetime"], utc=True).dt.tz_convert(ZoneInfo("Europe/Berlin"))
        combined_df = combined_df.reset_index(drop=False).rename(columns={'index': 'idx'})
        combined_df["sentiment"] = 0.0  # initialize with zeros
        combined_sorted = combined_df.rename(columns={'Datetime': 'merge_key'}).sort_values('merge_key').reset_index(drop=True)
        sentiments = pd.DataFrame()

        for file in csv_files:  # iterate over all files containing the sentiments
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path, parse_dates=['date'])

            df['sentiment'] = pd.to_numeric(df.get('sentiment', 0.0), errors='coerce').fillna(0.0)
            df['relevancy_factor'] = pd.to_numeric(df.get('relevancy_factor', 0.0), errors='coerce').fillna(0.0)

            #TODO: determine whether sentiment explains enough or if sentiment with relevancy explains more
            df["economy_impact"] = df["sentiment"] * df["relevancy_factor"]
            df['merge_key'] = df['date'].apply(compute_merge_key)

            sentiments_new = df[['merge_key', 'sentiment', 'economy_impact']]

            sentiments_new = sentiments_new.sort_values('merge_key').reset_index(drop=True)
            sentiments_new["merge_key"] = pd.to_datetime(sentiments_new["merge_key"], utc=True).dt.tz_convert(ZoneInfo("Europe/Berlin"))
            sentiments = pd.concat([sentiments, sentiments_new], ignore_index=True)

        sentiments = sentiments.sort_values('merge_key').reset_index(drop=True)
        agg = sentiments.groupby('merge_key')[['sentiment', 'economy_impact']].mean()

        result = pd.merge_asof(
            combined_sorted,
            agg,
            on='merge_key',
            direction='forward',
            tolerance=pd.Timedelta(hours=2)
        ).fillna(0.0)

        result = result.reset_index()  # idx wieder als Spalte
        result = result.drop(columns=['index', "idx", "sentiment_x"])
        result = result.rename(columns={'merge_key': 'Datetime', "sentiment_y": "sentiment"})
        result.to_csv(os.path.join(output_dir, f"{company}.csv"), index=False)


def add_time_zone_to_sentiments(companies:list[str]):
    input_dir = os.path.join('..', 'data', "news", 'sentiments_combined_with_economy')
    for company in companies:
        path = os.path.join(input_dir, company)
        sentiments = os.listdir(path)
        for file in sentiments:
            df = pd.read_csv(os.path.join(path, file))
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df["date"] = df["date"].dt.tz_convert(ZoneInfo("Europe/Berlin"))
            df.to_csv(os.path.join(path, file), index=False)


def compute_merge_key(ts):
    if ts.hour >= 17:
        # sentiments after 17: merge_key = 17:00 of the same day
        return ts.replace(hour=17, minute=0, second=0, microsecond=0)
    elif ts.hour < 9:
        # sentiments before 9: merge_key = 17:00 of the previous day
        return (ts - pd.Timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
    elif 9 <= ts.hour < 11:
        return ts.replace(hour=9, minute=0, second=0, microsecond=0)
    elif 11 <= ts.hour < 13:
        return ts.replace(hour=11, minute=0, second=0, microsecond=0)
    elif 13 <= ts.hour < 15:
        return ts.replace(hour=13, minute=0, second=0, microsecond=0)
    elif 15 <= ts.hour < 17:
        return ts.replace(hour=15, minute=0, second=0, microsecond=0)
    return None

#add_national_economy_sentiments(["Adesso", "Adidas", "BASF", "Bayer", "Biontech", "Continental", "Knorr-Bremse"])

#add_national_economy_sentiments(["BMW", "Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"])

#add_time_zone_to_sentiments(["SAP", "Siemens", "Airbus"])
#
# add_sentiments_to_combined_data(
#      ["Siemens", "SAP"],
#     input_dir=os.path.join('..', 'data', "news", 'sentiments_combined_with_economy'),
#     output_dir=os.path.join('..', 'data', 'stocks_combined_news')
# )

#add_sentiments_to_combined_data(
#     ["BMW"],
#)

#get_sentiments_from_combined_data(["BMW", "Airbus", "Allianz", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Volkswagen"])

# store_all_relevant_articles_from_news_api(
#     start_date="01/04/2025",
#     end_date="01/05/2025",
#     up_to_page_number=2,
#     company_name="Airbus",
#     is_company_given=True
# )

# response = get_articles(
#     "Airbus",
#     1,
#     "01/10/2025",
#     "02/10/2025"
# )

# print(response)