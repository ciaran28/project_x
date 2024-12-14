import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from data.transform import read_delta_table

def create_distribution_plot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='sentiment', palette='Set2')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

def create_sentiment_based_word_cloud(df: pd.DataFrame, sentiment: str) -> None:
    # Filter the DataFrame to include only rows with the specified sentiment
    filtered_df = df[df['sentiment'] == sentiment]
    
    # Join the content of the filtered file_content column into a single string
    text = " ".join(filtered_df['file_content'])
    
    # Generate a word cloud for the specified sentiment
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Plot and display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Common Words in Transcripts with {sentiment.capitalize()} Sentiment')
    plt.show()


def plot_sentiment_length_relationship(df: pd.DataFrame) -> None:
    df['content_length'] = df['file_content'].apply(len)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='sentiment', y='content_length', palette='Set2')
    plt.title('Relationship Between Sentiment and Content Length')
    plt.xlabel('Sentiment')
    plt.ylabel('Content Length')
    plt.show()


def run_bi():
    df = read_delta_table('data/delta')
    plot_sentiment_length_relationship(df)

