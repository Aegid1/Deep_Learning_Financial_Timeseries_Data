import csv
import json
import os
import time
from datetime import datetime
import pandas as pd
import yaml
from openai import OpenAI

config = yaml.safe_load(open("openai_config.yaml"))
client = OpenAI(api_key=config['KEYS']['openai'])

def create_one_prompt_for_batch(document_id: str, text: str, file_name: str, company_name: str):
    """
        Creates the prompt to analyze the sentiment of a document with respect to its short-term
        and long-term impact on the stock price of the company. Then adds the prompt to a batch file
        that is going to be processed with the OpenAI batch api.

        Parameters:
            document_id (str): The id of the document that gets analyzed
            text (str): The text of the document
            file_name (str): The name of the batch file that contains the prompts
            company_name (str): The name of the company
    """
    text = truncate_to_char_limit(text, 5000)
    if company_name == "Wirtschaft":
        prompt = (
                f"Analysiere die Stimmung des folgenden wirtschaftlichen artikels "
                f"und bewerten Sie es auf einer Skala von -10 bis +10, wobei -10 eine sehr negative Stimmung, 0 eine neutrale Stimmung und "
                f"+10 für eine sehr positive Stimmung steht. Finden Sie heraus, welche Relevanz der Artikel für die Kurse an der Börse hat. "
                "und bestimmen Sie einen Relevanzfaktor zwischen 1 und 0. 1 steht für eine hohe Relevanz und 0 für eine geringe Relevanz. "
                "Antworte nur mit einem Tupel im Format (Stimmung, Relevanzfaktor). "
                "richtiger output als Beispiel: (9, 0.85)\n"
                "richtiger output als weiteres Beispiel: (-3, 0.7)\n\n"
                "falscher output als Beispiel: To analyze the sentiment of the given economic news article ...\n"
                "falscher output als weiteres Beispiel: ([-5, 0.8]) ...\n\n"
                "Der Artikel:\n" + text
        )
    else:
        prompt = (
                f"Think step by step. Analyze the sentiment of the following economic news article about {company_name}"
                f"based on its short-term and long-term impact on the stock price of {company_name}."
                "Rate each on a scale from -10 to +10, where -10 indicates a very negative impact, 0 indicates a neutral impact, "
                f"and +10 indicates a very positive impact. Figure out which relevancy the article has to the stock price of {company_name}"
                "and determine a relevancy_factor between 1 and 0. 1 indicates a high relevancy and 0 indicates a low relevancy."
                "Respond only with a triple in the format (short-term sentiment, long-term sentiment, relevancy_factor). "
                "correct example output1: (9, 2, 0.85)\n"
                "correct example output2: (-7, -3, 0.7)\n\n"
                "wrong example output1: To analyze the sentiment of the given economic news article ...\n"
                "wrong example output2:  ([-5, 7, 0.8]) ...\n\n"
                "Article:\n" + text
        )
    request = {
        "custom_id": document_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini-2024-07-18",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial analyst specializing in sentiment analysis of economic news articles"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 15
        }
    }
    return add_to_batch(request, file_name, company_name)


def create_all_batches(company_name:str, batch_size:int, news_path:str):
    """
        Create batches of prompts from news articles for processing.

        This endpoint generates JSONL files containing batches of prompts from
        news article data for a specified company. The batches are organized by
        year, week, and batch number to allow systematic processing of large datasets.

        Parameters:
        - company_name (str): company of which the news are being processed
        - batch_size (int): maximum size of a batch

        Processing Steps:
        1. Identify the folder containing merged news data for the specified company.
        2. Iterate through the news files and process each line to extract document
           IDs, publication dates, and text.
        3. Group articles into batches of size 10, organized by ISO week and year.
        4. Create JSONL batch files with filenames indicating the company name, year,
           week, and batch number (e.g., `batch_Siemens_2024_W48_1.jsonl`).

        Batch Filename Format:
        - The batch filename follows the pattern:
          `batch_<company_name>_<year>_W<week>_<batch_number>.jsonl`.

        Example:
        If the `BatchMetadata` contains a company name `Siemens`, the generated
        batches look like:
        - `batch_Siemens_2024_W48_1.jsonl`
        - `batch_Siemens_2024_W48_2.jsonl`
        - ...
    """
    current_year, current_week = None, None

    batch_number = 1
    # iterate over all retrieved news: one line equals one news article
    for filename in os.listdir(news_path):
        file_path = os.path.join(news_path, filename)
        count_of_lines = 0
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                count_of_lines += 1
                # when the maximum number of prompts in a batch is exceeded create a new batch
                if count_of_lines > batch_size:
                    batch_number += 1
                    count_of_lines = 0

                document_id, date, text = line.split(",", 2)
                date = datetime.fromisoformat(date)
                year, week, _ = date.isocalendar()

                if current_year != year or current_week != week:
                    current_year, current_week = year, week
                    count_of_lines += 1

                # name of the batch looks like this afterwards: batch_Siemens_2025_W43_1.jsonl
                batch_filename = f"batch_{company_name}_{current_year}_W{current_week}_{batch_number}.jsonl"
                create_one_prompt_for_batch(document_id, text, batch_filename, company_name)


def send_batches_to_openai_api(company_name:str, just_send:bool):
    """
        Sends all batches of prompts for a provided company to the OpenAI service.

        Parameters:
            company_name (str): The name of the company.

    """
    # iterate through all batches of a given company and send them to openai
    folder_path = os.path.join("..", "data", "news", "batches", company_name)
    batch_id = None
    print("Total amount of batch files: ", len(os.listdir(folder_path)))
    counter = 0
    for filename in os.listdir(folder_path):
        print(counter)
        if not just_send:
            if batch_id:
                check_batch_status(batch_id)

        file_path = os.path.join(folder_path, filename)
        batch_id = send_batch(file_path, company_name)
        counter+=1


def retrieve_all_batches(company_name:str, is_general_news:bool):
    """
    Retrieve the content of batches and deletes the batch files, where the ids of the batches are stored for a given company

    Parameters:
            company_name (str): The name of the company.

    Returns:
        None
    """
    batch_ids = get_batch_ids(company_name)
    for batch_id in batch_ids:
        retrieve_batch_results(batch_id, company_name, is_general_news)


def send_batch(batch_file_name: str, company_name: str):
    """
        Send a batch of prompts and store the id of the sent batch into a file for later extraction of the results.

        Parameters:
            batch_file_name (str): The name of the batch file.
            company_name (str): The name of the company
    """
    batch_input_file = client.files.create(
        file=open(batch_file_name, "rb"),
        purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

    with open(os.path.join("..", "data", "news", "openai_batch_ids", company_name + "_batch_ids"), 'a') as file:
        file.write('\n' + batch.id)

    return batch.id


def retrieve_batch_results(batch_id: str, company_name: str, is_general_news: bool):
    """
        Retrieve the results of a sent batch.

        Parameters:
            batch_id (str): The ID of the batch.
            company_name (str): The name of the company
    """
    batch = client.batches.retrieve(batch_id)
    input_file_name = get_input_file_name_from_batch(batch)
    print(input_file_name)

    if batch.status != "completed":
        return "the batch is currently not ready and in the state: " + batch.status

    output_file = client.files.content(batch.output_file_id)
    content = output_file.content
    content_str = content.decode('utf-8')
    lines = content_str.splitlines()
    result_list = [json.loads(line) for line in lines]

    save_batch_results_in_csv_file(result_list, company_name, is_general_news)
    delete_batch_file(input_file_name, company_name)


def check_batch_status(batch_id: str):
    """
        Check the status of a batch.

        Parameters:
            batch_id (str): The ID of the batch.
    """
    batch_status = client.batches.retrieve(batch_id).status
    epoch = 0
    while batch_status != "completed":
        print("EPOCH: " + str(epoch))
        print(batch_status)
        time.sleep(10)
        batch_status = client.batches.retrieve(batch_id).status
        epoch += 1


def delete_batch_ids_file(batch_name):
    """
        Delete a batch file which contains the ids of the batches.

        Parameters:
            batch_name (str): The name of the batch
    """

    try:
        os.remove(batch_name)
        print(f"File '{batch_name}' deleted successfully.")
    except OSError as e:
        print(f"Error deleting file '{batch_name}': {e}")


def get_batch_ids(company_name:str):
    """
        Get batch IDs from a file.

        Parameters:
            batch_type (str): The type of document (SUMMARY or KEYWORDS).

        Returns:
            list: A list of batch IDs.
    """
    batch_ids = []

    with open(os.path.join("..", "data", "news", "openai_batch_ids", company_name + "_batch_ids"), "r") as file:
        for line in file:
            batch_id = line.strip()
            if batch_id:
                batch_ids.append(batch_id)
    return batch_ids


def add_to_batch(request: dict, file_name: str, company_name: str):
    """
            Add a request to a batch file in data/batches/[company_name].

            Parameters:
                request (dict): The request to add.
                company_name (str): The name of the company
    """
    json_str = json.dumps(request, ensure_ascii=False)
    with open(os.path.join("..", "data", "news", "batches", company_name, file_name), 'a',  encoding='utf-8') as file:
        file.write(json_str + '\n')


def save_batch_results_in_csv_file(results, company_name: str, is_general_news: bool):
    """
        Saves batch sentiment analysis results to a CSV file.

        This function processes the batch of results containing sentiment analysis data for different documents. It
        extracts the relevant information (document ID, sentiment analysis, and date), organizes it into a structured
        DataFrame, and saves the results as a CSV file.

        Parameters:
        - results (list): A list of result dictionaries, each containing the sentiment analysis data for a document.
        - company_name (str): The name of the company to be used in the output CSV file's path and filename.

        Workflow:
        1. Iterates through the provided results to extract document IDs, sentiment analysis responses, and their corresponding
           dates.
        2. Extracts short-term sentiment, long-term sentiment, and relevancy from the response, handling any issues with
           incorrect formatting.
        3. Retrieves the date of the document based on the document ID and company name.
        4. Creates a DataFrame containing the extracted data.
        5. Converts the date column to a proper datetime format.
        6. Determines the minimum and maximum date in the dataset for naming the CSV file.
        7. Saves the DataFrame as a CSV file with the name corresponding to the date range.

        Example:
        - The resulting CSV file will be saved in the "data/sentiments_news/{company_name}/{min_date}_to_{max_date}.csv" format,
          where `{min_date}` and `{max_date}` represent the date range of the sentiment analysis results.
        """
    data = []
    # Iterate through all results
    for result in results:
        document_id = result.get("custom_id")
        response = result.get("response").get("body").get("choices")[0].get("message").get(
            "content")  # gets the actual result

        # Get the determined long-term and short-term sentiment of the result
        try:
            open_brackets = response.count("(")
            close_brackets = response.count(")")
            if open_brackets + close_brackets != 2:
                response = response.replace(")", "")
                response = response.replace("(", "")
                response = response.replace("[", "")
                response = response.replace("]", "")
                response_splitted = response.split(",")
                if is_general_news:
                    sentiment,  relevancy = response_splitted[0], response_splitted[1]
                else:
                    short_term_sentiment, long_term_sentiment, relevancy = response_splitted[0], response_splitted[1], response_splitted[2]
            else:
                if is_general_news:
                    sentiment, relevancy = eval(response)
                else:
                    short_term_sentiment, long_term_sentiment, relevancy = eval(response)
        except (SyntaxError, ValueError, IndexError) as e:
            print(f"Fehler beim Verarbeiten von document_id {document_id} mit dem response: {response}: {e}")
            continue

        # Search for the date of the document
        date = get_date_of_document(document_id, company_name)
        if is_general_news:
            data.append({
                "document_id": document_id,
                "date": date,
                "sentiment": sentiment,
                "relevancy_factor": relevancy
            })
        else:
            data.append({
                "document_id": document_id,
                "date": date,
                "short_term_sentiment": short_term_sentiment,
                "long_term_sentiment": long_term_sentiment,
                "relevancy_factor": relevancy
            })

    if is_general_news:
        df = pd.DataFrame(data, columns=["document_id", "date", "sentiment", "relevancy_factor"])
        path = os.path.join("..", "data", "news", "sentiments", company_name)
    else:
        df = pd.DataFrame(data, columns=["document_id", "date", "short_term_sentiment", "long_term_sentiment", "relevancy_factor"])
        path = os.path.join("..", "data", "news", "sentiments_combined_with_economy", company_name)

    df['date'] = pd.to_datetime(df['date'])
    # Determine min and max date
    min_date = str(df["date"].min().date())
    max_date = str(df["date"].max().date())
    df['date'] = df['date'].dt.strftime("%Y-%m-%d %H:%M:%S")
    csv_filename = os.path.join(path, f"{min_date}_to_{max_date}.csv")

    # Save DataFrame as CSV
    df.to_csv(csv_filename, index=False)


def truncate_to_char_limit(text, char_limit):
    """
        Truncates the input text to a specified character limit.

        This function checks if the length of the text exceeds the given character limit and truncates the text
        accordingly. If the length is within the limit, it returns the original text.

        Parameters:
        - text (str): The text to be truncated.
        - char_limit (int): The maximum allowed character length.

        Returns:
        - str: The truncated text if the length exceeds the limit, or the original text if not.
    """
    if len(text) > char_limit:
        return text[:char_limit]
    return text


def delete_batch_file(filename: str, company_name:str):
    """
        Deletes a specified batch file from the filesystem.

        This function attempts to delete a batch file from the 'batches' directory. If the file exists, it is removed.
        Otherwise, an error message is displayed.

        Parameters:
        - filename (str): The name of the batch file to be deleted.
        - company_name (str): The name of the company (used to locate the file within the company's directory).

        Returns:
        - None: This function does not return any value.
    """
    file_path = os.path.join("..", "data", "news", "batches", f"{company_name}", f"{filename}")
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Die Datei {file_path} wurde gelöscht.")
        else:
            print(f"Die Datei {file_path} existiert nicht.")
    except Exception as e:
        print(f"Fehler beim Löschen der Datei {file_path}: {e}")


def get_date_of_document(document_id, company_name):
    """
        Retrieves the publication date of a document from its associated CSV file.

        This function searches through the company's merged news CSV files to find a document with the specified
        `document_id` and extracts its publication date.

        Parameters:
        - document_id (str): The ID of the document to retrieve the date for.
        - company_name (str): The name of the company (used to locate the directory containing the CSV files).

        Returns:
        - datetime: The publication date of the document.
    """
    directory = os.path.join("..", "data", "news", "news_merged", company_name) #TODO maybe revert to news_merged later
    for filename in os.listdir(directory):
        #if filename.endswith(".csv") and "merged" in filename:
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    uuid = row[0]
                    if uuid == document_id:
                        date_str = row[1]

                        #date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        date = datetime.fromisoformat(date_str)
                        return date


def get_input_file_name_from_batch(batch):
    """
        Retrieves the input file name from a batch object.

        This function uses the `input_file_id` from the provided batch object to retrieve the associated file's name
        from the client. If the file name is found, it is returned. Otherwise, a message is printed indicating that
        the name could not be found.

        Parameters:
        - batch: The batch object containing the input file ID.

        Returns:
        - str: The name of the input file, or None if the name cannot be found.
    """
    input_file_id = batch.input_file_id
    input_file = client.files.retrieve(input_file_id)
    input_file_name = input_file.filename

    if input_file_name:
        return input_file_name
    else:
        print("Der Name der Input-Datei konnte nicht gefunden werden.")
        return None


def cancel_all_running_batches(company_name:str):
    batch_ids = get_batch_ids(company_name)
    for batch_id in batch_ids:
        batch_status = client.batches.retrieve(batch_id).status
        if batch_status == "in_progress":
            batch = client.batches.cancel(batch_id)
            print(f"Cancelling batch: {batch.id} ....")
            input_file_name = get_input_file_name_from_batch(batch)
            delete_batch_file(input_file_name, company_name)
        else:
            print(f"batch: {batch_id} was cancelled or completed")


#cancel_all_running_batches("Wirtschaft")
#create_all_batches("Siemens", 40, os.path.join("..", "data", "news", "news_merged", "Siemens"))
#send_batches_to_openai_api("Siemens", True)
#retrieve_all_batches("Siemens", False)