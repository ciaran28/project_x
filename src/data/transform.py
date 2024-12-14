import os
import json
import re
import ipdb
import pandas as pd
from deltalake.writer import write_deltalake
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from deltalake import DeltaTable
from nltk.stem import WordNetLemmatizer


def data_cleaning(sent):
    """
    This function cleans sentences with the help of regex. URLs, hashtags, and newlines are removed.
    :param sent: Sentence to be cleaned
    :return: Returns cleaned sentence after Lemmatization
    """
    
    texts = sent.lower()
    texts = re.sub(r'http\S+', '', texts)  # Remove URLs
    texts = re.sub(r'@[A-Za-z0-9]+', '', texts)  # Remove user mentions
    texts = re.sub(r'#[A-Za-z0-9]+', '', texts)  # Remove hashtags
    texts = re.sub(r'\n', '', texts)  # Remove newlines
    texts = re.sub(r'[^a-zA-Z0-9\s:.]', '', texts)  # Remove non-alphabetic characters except ':' and '.'
    words = word_tokenize(texts)
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [ lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

def file_content_to_array(
        delta_table_path: str  = 'data/sentiment_analysis/delta'
        ) -> list:
    
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load the Delta table
    delta_table = DeltaTable(delta_table_path)
    
    # Convert the Delta table to a pandas DataFrame
    df = delta_table.to_pandas()

    # Extract all rows under the "file_content" column into an array of strings
    file_content_array = df['file_content'].astype(str).tolist()

    # Apply data cleaning to each string in the array
    cleaned_file_content_array = [data_cleaning(content) for content in file_content_array]

    # Replace file_content_array with the cleaned version
    file_content_array = cleaned_file_content_array
    return file_content_array



def flag_identical_files(data_folder: str) -> None:
    """
    Compares all files in a directory and flags any that are identical.

    :param data_folder: Path to the folder containing the transcript text files.
    """
    file_hashes = {}
    duplicates = []
    #ipdb.set_trace()

    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, 'r') as file:
                file_content = file.read()
                file_hash = hash(file_content)

                if file_hash in file_hashes:
                    duplicates.append((file_path, file_hashes[file_hash]))
                else:
                    file_hashes[file_hash] = file_path

    if duplicates:
        print("Identical files found:")
        for duplicate_pair in duplicates:
            print(f"File 1: {duplicate_pair[0]} is identical to File 2: {duplicate_pair[1]}")
    else:
        print("No identical files found.")



def create_delta_table(data_folder: str, output_folder: str = "data/sentiment_analysis/delta") -> None:
    """
    Creates a Delta table with columns 'file_name' and 'file_content' from transcript files.

    :param data_folder: Path to the folder containing the transcript text files.
    :param output_folder: Path to the folder where the Delta table will be saved.
    """
    files_data = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, 'r') as file:
                file_content = file.read()
                # Initialize the additional columns with empty strings
                file_entry = {
                    "file_name": filename,
                    "file_content": file_content,
                    "sentiment": '',  # Column added for sentiment analysis later
                    "actions": json.dumps({  # Serialize to JSON string
                        "actionItems": [
                            {
                                "actionItem": '',
                                "dueDate": '',
                                "notes": '',
                                "owner": '',
                                "status": ''
                            }
                        ]
                    })  # Simulated struct for actions
                }
                files_data.append(file_entry)

    df = pd.DataFrame(files_data)

    # Ensure the DataFrame has the correct dtype for 'actions' as a nested structure
    #df['actions'] = df['actions'].astype(str)  # Convert to string format if needed

    write_deltalake(output_folder, df, mode='overwrite')
    df_created = read_delta_table(output_folder)



def read_delta_table(output_folder: str = "data/sentiment_analysis/delta") -> pd.DataFrame:
    """
    Reads the Delta table from the given folder.
    """
    df = DeltaTable(output_folder).to_pandas()
    return df


def process_transcripts(data_folder: str) -> None:
    """
    Processes all transcript text files in the given folder by checking for identical files.

    :param data_folder: Path to the folder containing the transcript text files.
    """
    flag_identical_files(data_folder)
    create_delta_table(data_folder)


