import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ast
from rich import print
from rich.syntax import Syntax


def granualar_view_sentiment_and_actions(
        num_conversations: int,
        df_delta: pd.DataFrame
):
    for i in range(num_conversations):
        print(f"[bold cyan]Conversation no.:[/bold cyan] {i+1}\n")
        print(f"[bold cyan]File Content:[/bold cyan] {df_delta.file_content[i]}\n")

        action_str = df_delta.actions[i]
        #print(f"[bold cyan]Raw Action String:[/bold cyan] {action_str}\n")

        try:
            # Step 1: Safely convert the string to a Python dictionary using ast.literal_eval
            python_dict = ast.literal_eval(action_str)

            # Step 2: Convert the Python dictionary to a JSON string
            formatted_data = json.dumps(python_dict, indent=4)

            # Step 3: Use Rich to pretty-print the JSON string
            syntax = Syntax(formatted_data, "json", background_color="black", line_numbers=True)
            print(syntax)
            print(f"[bold green]Predicted Sentiment: {df_delta.sentiment[i]}.[/bold green]\n")

        except (SyntaxError, ValueError) as e:
            print(f"[bold red]Parsing Error:[/bold red] {e}")
            print(f"[bold red]Failed to parse the action string at index {i}. Please validate the string.[/bold red]\n")
        except Exception as e:
            print(f"[bold red]Unexpected Error:[/bold red] {e}")
            
def display_features_by_count(
        dataset: pd.DataFrame,
        feature_name: str,
        title: str,
        xlabel: str) :
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Calculate proportions
    marital_status_counts = dataset[feature_name].value_counts()
    marital_status_proportions = marital_status_counts / marital_status_counts.sum()

    # Create the count plot
    sns.barplot(
        x=marital_status_counts.index, 
        y=marital_status_counts.values, 
        color="blue", 
        alpha=0.6, 
        edgecolor="black"
    )

    # Overlay proportions as text
    for i, proportion in enumerate(marital_status_proportions):
        plt.text(i, marital_status_counts.values[i], f'{proportion:.2%}', ha='center', va='bottom', fontsize=10, color='black')

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()