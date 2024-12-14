from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def analyze_conversations(
        sentiment_analyzer,
        conversations
    ):

    results = []
    for index, conversation in enumerate(conversations):
        conversation_chunks = [conversation[i:i+512] for i in range(0, len(conversation), 512)]
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
        
        results.append({
            "conversation": conversation,
            "sentiment": dominant_label,
            "score": sentiment_scores
            })

    return results

def run_sentiment_analysis(
        delta_df,
        file_content_array: list) -> None:
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    sentiment_results = analyze_conversations(
        sentiment_analyzer=sentiment_analyzer,
        conversations=file_content_array)
    # Label mapping for sentiment analysis
    # 0: negative, 1: neutral, 2: positive

    for index, row in delta_df.iterrows():
        # get the sentiment result from the results array
        sentiment_result = sentiment_results[index]

        if sentiment_result['sentiment'] == 'LABEL_0':
            row['sentiment'] = 'negative'
        elif sentiment_result['sentiment'] == 'LABEL_1':
            row['sentiment'] = 'neutral'
        else:
            row['sentiment'] = 'positive'

        row['score'] = sentiment_result['score']
    
    return delta_df
