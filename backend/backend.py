#!/usr/bin/env python3

# ==============================================================================
# backend.py
# Take user input, process and attempt to detect probability that the article is fake news.
#
# Written by: Honggyo Suh <honggyo.suh@student.unsw.edu.au>
# Date: 2023-11-02
# For TWEETTRUTH fake news detection system
# ==============================================================================


import joblib
import subprocess
import pandas as pd
import json
import requests
import openai
import textstat
import nltk
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


# Our system class contains model, scaler, API key, and statistic results from analysis
class Backend_system:
    def __init__(self):
        # Load pre trained model
        self.loaded_model = joblib.load("logistic_regression.pkl")
        # Load pre trained model
        self.loaded_scaler = joblib.load("standard_scaler.pkl")
        # API key to OpenAI chatbot is stored in separate txt file.
        with open("API_key.txt", "r") as f:
            openai.api_key = f.readline().rstrip()
        # Average score for fake news is stored in separate txt file.
        with open("LIWC_average_fake.txt", "r") as f:
            self.averages = {
                line.rstrip().split(":")[0]: line.rstrip().split(":")[1]
                for line in f.readlines()
            }


# Process given text with LIWC software, will only work while software is working
def LIWC_process(sentence):
    LIWC_analysis_result = None

    cmd_to_execute = [
        "LIWC-22-cli",
        "--mode",
        "wc",
        "--input",
        "console",
        "--console-text",
        sentence,
        "--output",
        "console",
    ]

    # CLI access to the software
    try:
        results = (
            subprocess.check_output(cmd_to_execute, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
            .splitlines()
        )
        # Only capture the result
        LIWC_analysis_result = results[7]
    except Exception as e:
        print(f"Error LIWC processing: {sentence}. Error: {e}")

    # Process data into the format we want, we keep the data as Pandas dataframe
    LIWC_analysis_result = json.loads(LIWC_analysis_result)
    columns = [key for key, _ in LIWC_analysis_result.items()]
    values = [value for _, value in LIWC_analysis_result.items()]
    pd_dataframe = pd.DataFrame(
        {column: [value] for column, value in zip(columns, values)}
    )

    return pd_dataframe


# LIWC analysis will be scaled with pre-trained standard scaler
def scale_result(pd_dataframe, loaded_scaler):
    columns_to_scale = None

    # Load the list of sorted columns used for training
    with open("columns_to_scale", "r") as f:
        columns_to_scale = [line.rstrip() for line in f.readlines()]

    pd_dataframe[columns_to_scale] = loaded_scaler.transform(
        pd_dataframe[columns_to_scale]
    )

    return pd_dataframe


# Model tries to detect probability if the given sentence is fake news
def model_prediction(pd_dataframe, loaded_model):
    columns_to_predict = []

    # Load the list of sorted columns used for training
    with open("columns_to_predict", "r") as f:
        columns_to_predict = [line.rstrip() for line in f.readlines()]

    probability = loaded_model.predict_proba(pd_dataframe[columns_to_predict])
    prediction = loaded_model.predict(pd_dataframe[columns_to_predict])

    return (prediction[0], probability[0])


# Generate short query to be used with Google fact check tools API
def query_generation(sentence):
    prompt = [
        {
            "role": "system",
            "content": """Pretend you are a query builder who can extract keyword from the given article and suggest appropriate very short query.
        This query will be given to Google fact check tools API to retrieve related source information.
        Please format your response as below so that can easily be used.

        Format:
        Query: "Your suggestion here"
        Explanation: "Your explanation here"
        """,
        },
        {
            "role": "user",
            "content": f"""Sentence: {sentence}""",
        },
        {
            "role": "assistant",
            "content": "For example, when the input is 'When Obama was sworn into office, he DID NOT use the Holy Bible, but instead the Kuran (Their equivalency to our Bible, but very different beliefs).', this can be summarised into 'Barack Obama bible kuran'",
        },
    ]

    # Use API cahtbot to generate query
    answer = ask_GPT(prompt)
    # Extract query from the generated response
    if answer != "":
        query = extract_query_from_content(answer)
        return query.strip('"')
    else:
        return ""


# Helper function to extract query from the generated response
def extract_query_from_content(content):
    parts = content.split("\n")

    for part in parts:
        if part.startswith("Query:"):
            return part.replace("Query:", "").strip()


# Check with Google fact check tool API if we can find any related sources
def google_fact_check_tool(query):
    # Define the API endpoint and parameters
    endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    results_list = []

    params = {
        "query": query,
        "languageCode": "en-US",
        "pageSize": 10,
        "key": "AIzaSyCiPY5hrNpKHCZ1d-htnrhvQ_EjOFbBi0E",
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        results_list.append(data)
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    return results_list


# Generate explanation by summing up all related information
def explaination_generation(sentence, prediction, probability, resource, analysis):
    prompt = None
    # Check if we could find any reliable source, and result if any
    rating = extract_textual_rating(resource)

    # We found the reliable source claiming that this is fake news
    if rating == "False" or "Distorts the Facts":
        prompt = [
            {
                "role": "system",
                "content": f"""
                You will evaluate the truthworthy of given article provided by an user based on following features. 
                LIWC analysis result, fake news dataset search result from Google fact check tool API.
                Prioritise features as following order, search result from Google fact check tool API, LIWC analysis result.
                Assume your user could be a person who does not have any related background knowledge, and provide very short explanation including reasoning and assumptions with easy languages.
                Please format your response for readability.

                Format:
                Fact check result from Google database: {rating}
                Explanation: "Your explanation here"
                """,
            },
            {
                "role": "user",
                "content": f"""Article: {sentence}, resource: {resource}, analysis: {analysis}""",
            },
        ]
    # We found the reliable source claiming that this is true news
    elif rating == "True" or "Half True" or "Exagerated":
        prompt = [
            {
                "role": "system",
                "content": f""" 
                You will evaluate the truthworthy of given article provided by an user based on following features. 
                Fake news dataset search result from Google fact check tool API.
                Assume your user could be a person who does not have any related background knowledge, and provide very short explanation including reasoning and assumptions with easy languages.
                Please format your response as below for readability.

                Format:
                Fact check result from Google database: {rating}
                Explanation: "Your explanation here"
                """,
            },
            {
                "role": "user",
                "content": f"""Article: {sentence}, resource: {resource}""",
            },
        ]
    # We could not find the reliable source so that tries to provide best guess about the given sentence using our model
    else:
        if probability[prediction] > 0.7:
            prompt = [
                {
                    "role": "system",
                    "content": f"""
                    You will evaluate the truthworthy of given article provided by an user based on following features. 
                    LIWC analysis result, Probability and prediction result from LIWC based classification model.
                    This sentence was not found on Google fact check tool API.
                    Prioritise features as following order, LIWC analysis result, and Probability and prediction result.
                    Assume your user could be a person who does not have any related background knowledge, and provide very short explanation including reasoning and assumptions with easy languages.
                    Please format your response as below for readability.

                    Format:
                    Fact check result from Google database: No result
                    Model prediction result: probability of {"fake" if prediction == 0 else "true"}: {probability[prediction]:.3f}
                    Explanation: "Your explanation here"
                    """,
                },
                {
                    "role": "user",
                    "content": f"""Article: {sentence}, probability of {"fake" if prediction == 0 else "true"}: {probability[prediction]:.3f}, analysis: {analysis}""",
                },
            ]
        else:
            prompt = [
                {
                    "role": "system",
                    "content": """
                    You will evaluate the truthworthy of given article provided by an user based on following features. 
                    LIWC analysis result, Probability and prediction result from LIWC based classification model.
                    This sentence was not found on Google fact check tool API, and our classification model is not certain with their prediction result.
                    Assume your user could be a person who does not have any related background knowledge, and provide very short explanation including reasoning and assumptions with easy languages.
                    Please format your response as below for readability.

                    Format:
                    Fact check result from Google database: No result
                    Explanation: "Your explanation here"
                    """,
                },
                {
                    "role": "user",
                    "content": f"""Article: {sentence}, analysis: {analysis}""",
                },
            ]

    return ask_GPT(prompt)


# Use OpenAI API to generate response
def ask_GPT(prompt):
    response = None

    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt)
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(f"OpenAI API error occured, error: {e}")

    return "Language model errror"


# LIWC analysis comparing current sentence with statistics from training dataset
def LIWC_analysis(pd_dataframe, system):
    analysis_result = ""

    # For each feature from LIWC analysis using software
    for column in pd_dataframe:
        # Check what is the average value for this feature
        average = system.averages.get(column)
        value = pd_dataframe[column].iloc[0]
        average = float(average)
        value = float(value)
        abs_diff = abs(average - value)

        # Skip if they are extreme values
        if abs_diff == 0 or value == 0:
            continue

        # Analytic score check
        if column == "Analytic":
            if value > 95:
                analysis_result += "The sentence has extremely high analytic score, fake news might try to mimic legitimate analytic articles\n"
            if value < (average / 2):
                analysis_result += "The sentence has low analytic score, fake news might try to appeal to emotion\n"
        # Authentic score check
        elif column == "Authentic":
            if value < 50:
                analysis_result += "The sentence has low authentic score, fake news might hide their true intentions\n"
        # Tone score check
        elif column == "Tone":
            if value < 50:
                analysis_result += "The sentence has low tone score, fake news might have more negative or anxious tone\n"
        # Features related to level of reasoning
        elif column in [
            "insight",
            "cause",
            "cogproc",
            "discrep",
            "cognition",
            "tentat",
            "certitude",
        ]:
            if average > value:
                analysis_result += "The sentence has fewer cognitive words than average, fake news might use fewer cognitive words as they might not provide logical or rational arguments.\n"
        # Features related to emotional language
        elif column in [
            "Affect",
            "emo_neg",
            "emo_sad",
            "emo_anx",
            "emo_anger",
            "emotion",
            "tone_neg",
        ]:
            if average < value:
                analysis_result += "The sentence has more emotional words than average, fake news stories often use emotionally charged language to provoke reactions.\n"
        # Features related to use of personal pronouns
        elif column in ["i", "we", "you", "shehe", "they"]:
            if average > value:
                analysis_result += "The sentence has more personal pronouns than average, an overuse of personal pronouns might indicate a subjective or biased perspective.\n"
        # Features related to time orientation
        elif column in ["focuspresent"]:
            if average > value:
                analysis_result += "The sentence is more present focused than average, fake news might be more present focused, emphasising immediate events or emotions rather than providing historical context.\n"
        # Features related to perception
        elif column in ["see", "hear", "feel", "Perception", "visual"]:
            if average > value:
                analysis_result += "The sentence has more sensory languages than average, the use of words related to seeing or hearing might be indicative of claims without evidence.\n"
        # Features related to body and health
        elif column in ["physical", "health", "illness", "mental"]:
            if average > value:
                analysis_result += "The sentence has more body and health language than average, an over-emphasis on health related terms might indicate health related hoaxes or myths.\n"
        # Features related to motion verbs and narrative
        elif column in ["motion"]:
            if average > value:
                analysis_result += "The sentence has more motion verbs than average, excessive use of motion verbs might indicate a narrative being constructed.\n"
        # Features related to fear and power language
        elif column in ["achieve", "power", "Drives"]:
            if average > value:
                analysis_result += "The sentence has more achievement and power language than average, fake news might appeal to readers aspirations or fears related to power and achievements.\n"

    return analysis_result


def sentiment_analysis(pd_dataframe, sentence):
    # Caculate sentiment analysis scores with Textblob library, add them to the dataframe
    pd_dataframe["Polarity_txb"] = float(TextBlob(sentence).sentiment.polarity)
    pd_dataframe["Subjectivity_txb"] = float(TextBlob(sentence).sentiment.subjectivity)

    # Caculate sentiment analysis scores with Vader library, add them to the dataframe
    analyzer = SentimentIntensityAnalyzer()
    pd_dataframe["vader_pos"] = float(analyzer.polarity_scores(sentence)["pos"])
    pd_dataframe["vader_neu"] = float(analyzer.polarity_scores(sentence)["neu"])
    pd_dataframe["vader_neg"] = float(analyzer.polarity_scores(sentence)["neg"])
    pd_dataframe["vader_comp"] = float(analyzer.polarity_scores(sentence)["compound"])

    # Tokenise/clean statements into tokens, count the common words each in fake and true dataset
    nltk.download("punkt")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    # Common words will only be extracted from training dataset to prevent target exposure
    pd_dataframe["clean_text"] = sentence.lower().replace(r"[^\w\s]+", " ")
    pd_dataframe["tokens"] = pd_dataframe["clean_text"].apply(word_tokenize)
    pd_dataframe["filtered_tokens"] = pd_dataframe["tokens"].apply(
        lambda tokens: [w for w in tokens if not w in stop_words]
    )

    only_common_in_fake_with_prob = []
    with open("common_in_fake.txt") as f:
        only_common_in_fake_with_prob = [
            (line.rstrip().split()[0], float(line.rstrip().split()[1]))
            for line in f.readlines()
        ]

    only_common_in_true_with_prob = []
    with open("common_in_true.txt") as f:
        only_common_in_true_with_prob = [
            (line.rstrip().split()[0], float(line.rstrip().split()[1]))
            for line in f.readlines()
        ]

    # Add common words score/existance to the dataframe
    for column in only_common_in_fake_with_prob:
        pd_dataframe[column[0]] = [
            column[1] if column[0] in text else 0
            for text in pd_dataframe["filtered_tokens"]
        ]
    pd_dataframe["common_words_score_fake"] = [
        sum(
            [
                tuple[1]
                for tuple in only_common_in_fake_with_prob
                if tuple[0] in pd_dataframe["filtered_tokens"]
            ]
        )
    ]

    for column in only_common_in_true_with_prob:
        pd_dataframe[column[0]] = [
            column[1] if column[0] in text else 0
            for text in pd_dataframe["filtered_tokens"]
        ]
    pd_dataframe["common_words_score_true"] = [
        sum([tuple[1] for tuple in only_common_in_true_with_prob if tuple[0] in text])
        for text in pd_dataframe["filtered_tokens"]
    ]

    pd_dataframe["flesch_reading_ease"] = textstat.flesch_reading_ease(sentence)
    pd_dataframe["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(sentence)
    pd_dataframe["gunning_fog"] = textstat.gunning_fog(sentence)
    pd_dataframe["smog_index"] = textstat.smog_index(sentence)
    pd_dataframe["ari"] = textstat.automated_readability_index(sentence)
    pd_dataframe["coleman_liau_index"] = textstat.coleman_liau_index(sentence)
    pd_dataframe["linsear_write"] = textstat.linsear_write_formula(sentence)
    pd_dataframe["dale_chall"] = textstat.dale_chall_readability_score(sentence)

    return pd_dataframe


def detect(input):
    # System initialisation
    system = Backend_system()
    # Log all errors while processing
    sys.stderr = open("error_log.txt", "w")
    # Process article with LIWC software
    pd_dataframe = LIWC_process(input)
    print("LIWC processed")
    print(pd_dataframe.iloc[0])
    # Statistical analysis of LIWC result
    analysis = LIWC_analysis(pd_dataframe, system)
    print("Analysis conducted")
    print(analysis)
    # Sentiment analysis
    pd_dataframe = sentiment_analysis(pd_dataframe, input)
    print(pd_dataframe.iloc[0])
    # Scale LIWC result
    pd_dataframe = scale_result(pd_dataframe, system.loaded_scaler)
    print("Scale conducted")
    print(pd_dataframe.iloc[0])
    # Predict fake news with pre-trained model
    prediction, probability = model_prediction(pd_dataframe, system.loaded_model)
    print("Prediction conducted")
    print(prediction, probability)
    print(prediction == 1)
    print(probability[prediction])
    # Generate query with OpenAI API
    query = query_generation(input)
    print("Query generated")
    print(query)
    # Search related source with Google API
    resource = google_fact_check_tool(query)
    print("Source retrieved")
    print(resource)
    # Generate user friendly explanation
    explanation = explaination_generation(
        input, prediction, probability, resource, analysis
    )
    print("Explaination generated")

    # Close stderr redirection
    sys.stderr.close()
    sys.stderr = sys.__stderr__

    return explanation


# To extract rating from related source
def extract_textual_rating(data):
    textual_ratings = []

    for item in data:
        claims = item.get("claims", [])

        for claim in claims:
            claim_reviews = claim.get("claimReview", [])

            for review in claim_reviews:
                rating = review.get("textualRating", None)
                if rating:
                    textual_ratings.append(rating)

    return textual_ratings[0] if len(textual_ratings) != 0 else None


if __name__ == "__main__":
    pass
