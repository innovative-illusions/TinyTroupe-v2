import os
import logging
import configparser
import rich # for rich console output
import rich.jupyter

# add current path to sys.path
import sys
sys.path.append('.')
from tinytroupe import utils # now we can import our utils

# AI disclaimers
print(\
"""
!!!!
DISCLAIMER: TinyTroupe relies on Artificial Intelligence (AI) models to generate content. 
The AI models are not perfect and may produce inappropriate or inacurate results. 
For any serious or consequential use, please review the generated content before using it.
!!!!
""")


###########################################################################
# Default parameter values
###########################################################################
# We'll use various configuration elements below
config = utils.read_config_file()
utils.pretty_print_config(config)
utils.start_logger(config)

default = {}
default["embedding_model"] = config["OpenAI"].get("EMBEDDING_MODEL", "text-embedding-3-small")
default["max_content_display_length"] = config["OpenAI"].getint("MAX_CONTENT_DISPLAY_LENGTH", 1024)
if config["OpenAI"].get("API_TYPE") == "azure":
    default["azure_embedding_model_api_version"] = config["OpenAI"].get("AZURE_EMBEDDING_MODEL_API_VERSION", "2023-05-15")


## LLaMa-Index configs ########################################################
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding

if config["OpenAI"].get("API_TYPE") == "azure":
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
else:
    from llama_index.embeddings.openai import OpenAIEmbedding

import logging # Ensure logging is imported
from llama_index.core import Settings, Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.openai import OpenAIEmbedding # Corrected import
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding # Corrected import
# Attempt to import the enum for validation, handle if it doesn't exist in older versions
try:
    from llama_index.embeddings.openai.base import OpenAIEmbeddingModelType
except ImportError:
    OpenAIEmbeddingModelType = None # Placeholder if enum isn't available

logger = logging.getLogger("tinytroupe") # Get logger instance

# this will be cached locally by llama-index, in a OS-dependend location

##Settings.embed_model = HuggingFaceEmbedding(
##    model_name="BAAI/bge-small-en-v1.5"
##)

# Determine the model name to use for llama-index initialization
llama_index_model_name = default["embedding_model"]
fallback_model = "text-embedding-3-small" # Define a fallback

if OpenAIEmbeddingModelType: # Check if enum was imported successfully
    try:
        # Attempt to validate the model name against the enum
        OpenAIEmbeddingModelType(default["embedding_model"])
        # If successful, use the configured model name
        llama_index_model_name = default["embedding_model"]
    except ValueError:
        # If validation fails, log a warning and set the fallback model name
        logger.warning(
            f"Configured embedding model '{default['embedding_model']}' is not a standard OpenAI model recognized by llama-index. "
            f"Falling back to '{fallback_model}' for llama-index internal initialization."
        )
        llama_index_model_name = fallback_model
else:
    # If enum couldn't be imported, skip validation and use configured name directly (older llama-index behavior)
    logger.info("Could not import OpenAIEmbeddingModelType for validation. Using configured embedding model directly for llama-index.")
    llama_index_model_name = default["embedding_model"]


# Initialize llama-index embedding model
if config["OpenAI"].get("API_TYPE") == "azure":
    # For Azure, use the validated/fallback model name for 'model' parameter for llama-index compatibility,
    # but use the original configured name for 'deployment_name'.
    llamaindex_openai_embed_model = AzureOpenAIEmbedding(
        model=llama_index_model_name,
        deployment_name=default["embedding_model"], # Use original name for deployment
        api_version=default.get("azure_embedding_model_api_version", "2023-05-15"), # Use .get for safety
        embed_batch_size=10
    )
else:
    # For standard OpenAI, use the validated/fallback model name
    llamaindex_openai_embed_model = OpenAIEmbedding(
        model=llama_index_model_name,
        embed_batch_size=10
    )

Settings.embed_model = llamaindex_openai_embed_model

###########################################################################
# Fixes and tweaks
###########################################################################

# fix an issue in the rich library: we don't want margins in Jupyter!
rich.jupyter.JUPYTER_HTML_FORMAT = \
    utils.inject_html_css_style_prefix(rich.jupyter.JUPYTER_HTML_FORMAT, "margin:0px;")


