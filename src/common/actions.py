import os 
from typing import Annotated
from dotenv import load_dotenv, dotenv_values
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings)
from data.transform import read_delta_table
from ai.sk_plugins.light_plugin import LightPlugin

def get_cleansed_actions(result) -> dict:
    """
        Parse a JSON string into a dictionary
    """
    actions_str = result.value[0].items[0].text
    cleansed_actions_str = actions_str.replace('\n', '').strip('```json').strip('```')
    import json
    # Step 2: Parse the cleaned string into a JSON object
    cleansed_actions = json.loads(cleansed_actions_str)
    return cleansed_actions


async def run_retrieve_actions(
        plugin_folder: str,
        delta_df
):
    #Load secrets from .env file
    load_dotenv()
    config = dict(dotenv_values(".env"))

    # Instantiate the Semantic Kernel
    kernel = Kernel()
    chat_completion = AzureChatCompletion(
        deployment_name=config['DEPLOYMENT_NAME'],
        api_key=config['API_KEY'],
        endpoint=config['ENDPOINT'],
    )
    kernel.add_service(chat_completion)

    # Instantiate Semantic Kernel Plugins
    plugin_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        plugin_folder
    )
    
    sk_plugins = kernel.add_plugin(plugin_name="sk_plugins", parent_directory=plugin_path)

    
    
    for index, row in delta_df.iterrows():
        print(f"Conversation no.: {index + 1}")
        import re
        conversation = re.sub(r'\n', '', row['file_content'])

        # Throttling at 150 words - use summarizer plugin
        # Throttling will occur at much higher document lengths and volume
        # content trigger warnings require try except
        # Significant Prompt Engineering required. 
        
        # To do: If number of words in the conversation is less than 150, summarize the conversation
        try:
            result = await kernel.invoke(
                sk_plugins["RetrieveActions"],
                conversation=conversation
                )
            
            actions = get_cleansed_actions(
                result=result
            )
            print(actions)
            row['actions'] = str(actions)
        except Exception as e:
            try:
                print()
                print("Request too large. Summarizing and trying again")
                result = await kernel.invoke(
                    sk_plugins["SummarizeConversation"],
                    conversation=conversation
                    )
                
                summarized_conversation = result.value[0].items[0].text
                print(summarized_conversation)
                
                print(f"summary length: {len(summarized_conversation.split())}")

                if len(summarized_conversation.split()) <= 150:
                    result = await kernel.invoke(
                        sk_plugins["RetrieveActions"],
                        conversation=summarized_conversation
                        )
                    
                    actions = get_cleansed_actions(
                        result=result
                    )
                    print(actions, "\n")
                    row['actions'] = str(actions)
            except Exception as e:
                print(f"An error occurred during kernel.invoke: {e}")
    
    print("Actions Retrieved")
    return delta_df









    # light_plugin = kernel.add_plugin(
    #     LightPlugin(),
    #     plugin_name="LightPlugin",
    # )

    # result = await kernel.invoke(light_plugin["get_state"])
    # print(f"The light is: {result}")

    # print("Changing the light's state...")

    # # Kernel Arguments are passed in as kwargs.
    # result = await kernel.invoke(light_plugin["change_state"], new_state=True)
    # print(f"The light is: {result}")

