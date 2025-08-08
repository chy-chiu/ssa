from langchain_openai import ChatOpenAI

def init_openrouter_chat_model(
    model_name: str, temperature: float, api_key: str, **kwargs
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

    return ChatOpenAI(
        model_name=model_name,  # e.g., "anthropic/claude-3-opus-20240229"
        temperature=temperature,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        **kwargs,
    )
