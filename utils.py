from langchain_openai import ChatOpenAI, AzureChatOpenAI
import yaml

def init_openrouter_chat_model(
    model_name: str, temperature: float, api_key: str = None, **kwargs
):
    """
    Initializes a chat model from OpenAI or OpenRouter.

    Args:
        model_identifier: String in the format "provider:model_name"
                          e.g., "openai:gpt-4o-mini"
                          e.g., "openrouter:anthropic/claude-3-opus-20240229"
        temperature: The sampling temperature.
        api_key: The API key for the specified provider.
        **kwargs: Additional arguments for the Chat model constructor.

    Returns:
        An instance of ChatOpenAI configured for the specified provider.
    """
    
    if not api_key: 
        lab_endpoints = yaml.safe_load(open('secrets.yaml'))
        api_key = lab_endpoints['openrouter']["API_KEY"]
        
    model_kwargs = {
        "reasoning": {"max_tokens": 1000},
        # "output_version" is not a standard or OpenRouter parameter and should be removed.
    }

    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        extra_body=model_kwargs,
        **kwargs,
    )

def init_azure_model(
    model_name: str="gpt-4o-sh-1", temperature: float=0.5, api_key: str="", **kwargs
):
    """
    Initializes a chat model from OpenAI or OpenRouter.

    Args:
        model_identifier: String in the format "provider:model_name"
                          e.g., "openai:gpt-4o-mini"
                          e.g., "openrouter:anthropic/claude-3-opus-20240229"
        temperature: The sampling temperature.
        api_key: The API key for the specified provider.
        **kwargs: Additional arguments for the Chat model constructor.

    Returns:
        An instance of ChatOpenAI configured for the specified provider.
    """

    lab_endpoints = yaml.safe_load(open('secrets.yaml'))
    config = lab_endpoints[model_name]
    endpoint = config["API_ENDPOINT"]
    api_key = config["API_KEY"]

    return AzureChatOpenAI(azure_deployment=model_name, 
            temperature=temperature, 
            api_version="2025-01-01-preview",
            azure_endpoint=endpoint,
            api_key=api_key)

def format_dict_str(_dict):
    # print(_dict)
    return "[" + ", ".join(f"{k}: {_dict[k]}" for k in sorted(_dict)) + "]"