from sklearn.metrics import classification_report, accuracy_score
from deltalake import DeltaTable
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def run_evaluation_metrics(
        df_delta_path: str = 'data/sentiment_analsis/delta',
        df_ground_truth_path: str = 'data/sentiment_analysis/ground_truth/ground_truth.csv'
        ) -> None:
        """
        Produce evaluation metrics on the sentiment analysis model
        
        args:
                df_delta_path: str: path to the delta table
                df_ground_truth_path: str: path to the ground truth csv file

        :return: None
        """
        
        df_delta = DeltaTable(df_delta_path).to_pandas()

        df_ground_truth = pd.read_csv(df_ground_truth_path)
        df_merged = pd.merge(df_ground_truth, df_delta, on='file_name')

        # Extracting ground truth and predictions
        y_true = df_merged['ground_truth'].tolist()
        y_pred = df_merged['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).tolist()

        #from ipdb import set_trace; set_trace()

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Create a seaborn styled plot for accuracy in the form of a gauge
        plt.figure(figsize=(8, 4))
        sns.set_style("whitegrid")

        # Plotting a filled gauge-like appearance
        plt.barh(0, accuracy, height=0.3, color='mediumseagreen')
        plt.barh(0, 1 - accuracy, left=accuracy, height=0.3, color='lightgray')

        # Set the limits and labels
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.title('Model Accuracy', fontsize=16)
        plt.xlabel('Accuracy')
        plt.yticks([])  # Remove y-ticks for cleanliness
        plt.xticks(ticks=[0, 0.25, 0.5, 0.75, 1], labels=['0%', '25%', '50%', '75%', '100%'])

        # Add text annotation for accuracy percentage
        plt.text(accuracy / 2, 0, f'{accuracy:.2%}', ha='center', va='center', fontsize=12, color='white')

        # Display the plot
        plt.tight_layout()
        plt.show()

        # Visualize the Classification Report using a Heatmap
        class_report = classification_report(y_true, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Use a neutral color palette
        cmap = sns.light_palette("slategray", as_cmap=True)

        sns.heatmap(
                class_report_df.iloc[:-1, :-1],
                annot=True,
                fmt='.2f',
                cmap=cmap,
                linewidths=0.5,
                linecolor='gray',
                cbar_kws={'label': 'Score'}
        )

        plt.title('Classification Report', fontsize=16, color='dimgray')
        plt.xlabel('Metrics', fontsize=12, color='dimgray')
        plt.ylabel('Classes', fontsize=12, color='dimgray')
        plt.xticks(rotation=45, color='gray')
        plt.yticks(rotation=0, color='gray')

        # Adjusting the plot aesthetics for a cleaner view
        plt.tight_layout()
        # Display the plot
        plt.show()

