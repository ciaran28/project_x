import os 
import ipdb
from dotenv import load_dotenv, dotenv_values
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)


async def run_summarize(
        plugin_folder: str,
        conversation: str
):
    load_dotenv()
    config = dotenv_values(".env") 

    kernel = Kernel()
    chat_completion = AzureChatCompletion(
        deployment_name="",
        api_key="",
        endpoint="",
    )
    kernel.add_service(chat_completion)


    plugin_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        plugin_folder
    )

    plugin = kernel.add_plugin(plugin_name="sample_plugins", parent_directory=plugin_path)

    result = await kernel.invoke(
        plugin["SummarizeConversation"],
        user_message=conversation
        )
    
    result = result.value[0].inner_content.choices[0].message.content
    return result
