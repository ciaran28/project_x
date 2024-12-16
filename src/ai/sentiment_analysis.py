from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd


def analyze_conversations(sentiment_analyzer, conversations) -> list:
    """
    Analyze the sentiment of the conversations using the sentiment_analyzer pipeline.

    args:
        sentiment_analyzer: sentiment analysis pipeline
        conversations: list of conversations to analyze

    return:
        results: list of dictionaries containing the conversation, sentiment, and score
    """

    results = []
    for index, conversation in enumerate(conversations):
        conversation_chunks = [
            conversation[i : i + 512] for i in range(0, len(conversation), 512)
        ]
        sentiment_scores = {"LABEL_0": 0, "LABEL_1": 0, "LABEL_2": 0}
        total_length = sum(len(chunk) for chunk in conversation_chunks)

        for chunk in conversation_chunks:
            sentiment = sentiment_analyzer(chunk)[0]
            score = sentiment["score"]
            label = sentiment["label"]
            weight = len(chunk) / total_length
            sentiment_scores[label] += score * weight

        # Determine the dominant label by highest weighted score
        dominant_label = max(sentiment_scores, key=sentiment_scores.get)

        results.append(
            {
                "conversation": conversation,
                "sentiment": dominant_label,
                "score": sentiment_scores,
            }
        )

    return results


def run_sentiment_analysis(
    delta_df: pd.DataFrame, file_content_array: list
) -> pd.DataFrame:
    """
    Run sentiment analysis on the conversations in the delta_df dataframe.
    The sentiment analysis is done using the cardiffnlp/twitter-roberta-base-sentiment model.
    The sentiment analysis results are stored in the delta_df dataframe.

    args:
        delta_df: pandas dataframe containing the conversations
        file_content_array: list of conversations

    return:
        delta_df: pandas dataframe containing the conversations with sentiment analysis results
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    sentiment_analyzer = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )
    sentiment_results = analyze_conversations(
        sentiment_analyzer=sentiment_analyzer, conversations=file_content_array
    )
    # Label mapping for sentiment analysis
    # 0: negative, 1: neutral, 2: positive

    for index, row in delta_df.iterrows():
        # get the sentiment result from the results array
        sentiment_result = sentiment_results[index]

        if sentiment_result["sentiment"] == "LABEL_0":
            row["sentiment"] = "negative"
        elif sentiment_result["sentiment"] == "LABEL_1":
            row["sentiment"] = "neutral"
        else:
            row["sentiment"] = "positive"

        row["score"] = sentiment_result["score"]

    return delta_df
