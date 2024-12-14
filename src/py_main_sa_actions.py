import ipdb
import asyncio
from ai.sentiment_analysis import run_sentiment_analysis
from common.actions import run_retrieve_actions
from data.transform import process_transcripts, file_content_to_array, write_deltalake, read_delta_table


def run(
        data_folder: str = 'data/sentiment_analysis',
        delta_folder: str = 'data/sentiment_analysis/delta'
) -> None:
    """
        Run the sentiment analysis pipeline

        1. Process the transcripts
        2. Perform sentiment analysis
        3. Update the delta table with the sentiment results

    """

    # Step 1: Process the transcripts
    process_transcripts(data_folder=data_folder)

    delta_df = read_delta_table(delta_folder)

    # Step 2: Perform sentiment analysis
    file_contents = file_content_to_array(delta_table_path=delta_folder)
    delta_df = run_sentiment_analysis(
        delta_df=delta_df,
        file_content_array=file_contents
        )
    
    delta_df = asyncio.run(
        run_retrieve_actions(
            plugin_folder='ai',
            delta_df=delta_df
            )
        )
    
    print("Update the Delta Table")
    write_deltalake(delta_folder, delta_df, mode='overwrite')


if __name__ == "__main__":
    run()