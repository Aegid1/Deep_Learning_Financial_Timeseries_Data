import json
import re
import time
import uuid
from datetime import datetime, date, timedelta
import pandas as pd
import requests
import yaml

config = yaml.safe_load(open("openai_config.yaml"))
def get_tweets_by_hashtag_and_date(company_name:str, start_date:str, end_date:str):
    url = "https://twitter-api-v1-1-enterprise.p.rapidapi.com/base/apitools/search"
    rapid_api_key = config["KEYS"]["rapid-api"]
    twitter_api_key = config["KEYS"]["twitter_api_v1_1"]
    querystring = {
        "any": company_name,
        "words": company_name,
        "until": start_date,
        "since": end_date,
        "tag": company_name,
        "resFormat": "json",
        "apiKey": twitter_api_key
    }

    headers = {
        "x-rapidapi-key": rapid_api_key,
        "x-rapidapi-host": "twitter-api-v1-1-enterprise.p.rapidapi.com"
    }

    return requests.get(url, headers=headers, params=querystring)


def extract_results_from_twitter_api(api_result):
    return api_result.get("data").get("data").get("search_by_raw_query").get("search_timeline").get("timeline").get("instructions")[0].get("entries")


def get_open_days_twitter(start_date_str:str, end_date_str:str):
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        holidays = [
            date(start_date.year, 1, 1),  # Neujahrstag
            date(start_date.year, 4, 7),  # Karfreitag
            date(start_date.year, 4, 10),  # Ostermontag
            date(start_date.year, 5, 1),  # Tag der Arbeit
            date(start_date.year, 5, 18),  # Christi Himmelfahrt
            date(start_date.year, 5, 29),  # Pfingstmontag
            date(start_date.year, 10, 3),  # Tag der Deutschen Einheit
            date(start_date.year, 12, 25),  # Erster Weihnachtstag
            date(start_date.year, 12, 26)  # Zweiter Weihnachtstag
        ]

        if start_date.year != end_date.year:
            holidays.extend([
                date(end_date.year, 1, 1),
                date(end_date.year, 4, 7),
                date(end_date.year, 4, 10),
                date(end_date.year, 5, 1),
                date(end_date.year, 5, 18),
                date(end_date.year, 5, 29),
                date(end_date.year, 10, 3),
                date(end_date.year, 12, 25),
                date(end_date.year, 12, 26)
            ])

        open_days = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5 and current_date not in holidays:
                open_days.append(current_date.strftime("%Y/%m/%d"))
            current_date += timedelta(days=1)

        return open_days


def store_all_relevant_articles_from_news_api(start_date: str, end_date:str, company_name:str):
    start_time = time.time()
    articles_list = []
    days = get_open_days_twitter(start_date, end_date)
    print(days)
    for i in range(len(days)-1):
        print(f"DAY: {days[i]} to {days[i+1]}")
        response = get_tweets_by_hashtag_and_date(company_name, days[i], days[i+1]).content.decode("utf-8")
        response = json.loads(response)
        posts = extract_results_from_twitter_api(response)
        for post in posts:
            if post:
                content = post.get("content").get("itemContent").get("tweet_results").get("result").get("legacy").get("full_text")
                date = post.get("content").get("itemContent").get("tweet_results").get("result").get("legacy").get("created_at") #time format: Tue Oct 31 23:13:56 +0000 2023
                content = re.sub(r'\s+', ' ', content).strip()  # news texts contain unnecessary newlines
                print(content)
                print(date)
                converted_date = pd.to_datetime(date, format='%a %b %d %H:%M:%S %z %Y')
                articles_list.append((str(uuid.uuid4()), converted_date, content))

        i+=2

    articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
    articles_df = articles_df.drop_duplicates(subset='Text')
    articles_sorted = articles_df.sort_values(by='Date')
    articles_sorted.to_csv(f"data/twitter_posts/{company_name}/{company_name}_sorted.csv", index=False, header=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Die Funktion hat {duration:.2f} Sekunden gebraucht.")