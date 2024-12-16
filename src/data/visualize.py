import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ast
from rich import print
from rich.syntax import Syntax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def display_correration_matrix(
        dataset: pd.DataFrame
        ) -> None:
    """
    Display a correlation matrix of a dataset.

    Args:
        dataset : pd.DataFrame
        A pandas DataFrame containing the data.

    Returns:
    None
    """

# Compute the correlation matrix
    corr_matrix = dataset.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 12))  # Increase the figure size for readability

    # Create a heatmap with annotations and rotation
    sns.heatmap(
        corr_matrix,
        annot=True,        # Display correlation values
        cmap='coolwarm',   # Choose a visually appealing colormap
        fmt=".2f",         # Limit correlation values to 2 decimal places
        linewidths=0.5     # Add space between cells
    )

    # Rotate x and y-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.title("Correlation Matrix Heatmap", fontsize=16)
    plt.tight_layout()  # Ensure everything fits within the figure
    plt.show()

def display_feature_importance(
        model,
        X: pd.DataFrame
        ) -> None:
    """
    Display feature importance of a model.

    Args:
    model : object
        A trained machine learning model.
    X : pd.DataFrame
        A pandas DataFrame containing the features used to train the model.

    Returns:
    None
    """

    # Plot feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance in XGBoost")
    plt.show()
    
def display_column_count_hist(
        source_dataset: pd.DataFrame,
        column_name: str,
) -> None:
    sns.histplot(source_dataset['Age'], kde=True, stat="density", color="blue", alpha=0.6, edgecolor="black")
    sns.kdeplot(source_dataset['Age'], color="red", linewidth=2)

    plt.title(f"{column_name} Distribution")
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.show()

def get_granualar_view_sentiment_and_actions(
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