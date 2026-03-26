# %%
# 
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import yaml
from openai import OpenAI
from langchain_core.messages import convert_to_openai_messages, HumanMessage
from pydantic import BaseModel
from typing import Dict, Any
import requests
import json


def init_openrouter_chat_model(
    model_name: str, temperature: float, api_key: str = None, secrets_path: str = None, **kwargs
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
    secrets_path = secrets_path or "ssa/assets/secrets.yaml"

    if not api_key:
        lab_endpoints = yaml.safe_load(open(secrets_path))
        api_key = lab_endpoints["openrouter"]["API_KEY"]

    model_kwargs = {
        # "reasoning": {"max_tokens": 1000, "effort": "minimal"},
        "reasoning": {"effort": "minimal", "enabled": False},
        "provider": {"sort": "latency"},
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
    model_name: str = "gpt-5-cc", temperature: float = 0.5, api_key: str = "", secrets_path: str = None, **kwargs
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
    secrets_path = secrets_path or "ssa/assets/secrets.yaml"
    lab_endpoints = yaml.safe_load(open(secrets_path))
    config = lab_endpoints[model_name]
    endpoint = config["API_ENDPOINT"]
    api_key = config["API_KEY"]

    model_kwargs = {
        "reasoning": {"effort": "high"},
    }

    return AzureChatOpenAI(
        azure_deployment=model_name,
        temperature=temperature,
        api_version="2025-01-01-preview",
        azure_endpoint=endpoint,
        api_key=api_key,
        reasoning={"effort": "high"},
    )


class LangChainResponse(BaseModel):

    content: str
    response_metadata: Dict[str, Any]


class OpenRouterClient:
    """Because langchain sucks"""

    def __init__(
        self,
        model_name,
        temperature=0.5,
        base_url="https://openrouter.ai/api/v1/chat/completions",
        secrets_path=None,
        api_key=None,
        effort="low",
    ):

        secrets_path = secrets_path or "ssa/assets/secrets.yaml"
        lab_endpoints = yaml.safe_load(open(secrets_path))

        if not api_key:
            api_key = lab_endpoints["openrouter"]["API_KEY"]

        self.model_name = model_name
        self.temperature = temperature
        self.effort = effort
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.base_url = base_url
        self.api_key = api_key

    def invoke(self, messages):

        if self.model_name == "deepseek/deepseek-chat-v3.1":
            reasoning = {"enabled": False}
        elif self.model_name == "openai/gpt-5":
            reasoning = {"effort": "minimal"}
        reasoning = {"effort": self.effort}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": convert_to_openai_messages(messages),
            "reasoning": reasoning,
        }

        response = requests.post(self.base_url, headers=headers, data=json.dumps(payload)).json()

        message = response["choices"][0]["message"]
        content = message["content"]
        llm_reasoning = message.get("reasoning")
        token_usage = response["usage"]

        return LangChainResponse(
            content=content, response_metadata=dict(token_usage=token_usage, llm_reasoning=llm_reasoning)
        )


class OpenAIClient:
    """Because langchain sucks"""

    def __init__(
        self,
        model_name="gpt-5-cc",
        temperature=0.5,
        secrets_path=None,
        api_key=None,
        effort="low",
    ):

        secrets_path = secrets_path or "ssa/assets/secrets.yaml"
        lab_endpoints = yaml.safe_load(open(secrets_path))

        if not api_key:
            base_url = lab_endpoints["gpt-5-cc"]["API_ENDPOINT"]
            api_key = lab_endpoints["gpt-5-cc"]["API_KEY"]

        self.model_name = model_name
        self.temperature = temperature
        self.effort = effort
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def invoke(self, messages):

        reasoning = {"effort": self.effort}

        response = self.client.chat.completions.create(
            model=self.model_name, messages=convert_to_openai_messages(messages), reasoning_effort=self.effort
        )
        message = response.choices[0].message
        content = message.content
        llm_reasoning = message.model_dump().get("reasoning")
        token_usage = response.usage.model_dump()

        return LangChainResponse(
            content=content, response_metadata=dict(token_usage=token_usage, llm_reasoning=llm_reasoning)
        )


def format_dict_str(_dict):
    # print(_dict)
    return "[" + ", ".join(f"{k}: {_dict[k]}" for k in sorted(_dict)) + "]"


def generate_gini_table(data_tuples):
    """
    Generate LaTeX table from tuples of (ratio, market_limit, mean, std)
    
    Args:
        data_tuples: List of tuples (ratio, market_limit, mean, std)
    
    Returns:
        str: LaTeX table string
    """
    import numpy as np
    from collections import defaultdict
    
    # Organize data by market_limit and ratio
    table_data = defaultdict(dict)
    ratios = set()
    market_limits = set()
    
    for ratio, market_limit, mean, std in data_tuples:
        table_data[market_limit][ratio] = (mean, std)
        ratios.add(ratio)
        market_limits.add(market_limit)
    
    # Sort ratios and market_limits
    sorted_ratios = sorted(ratios)
    sorted_market_limits = sorted(market_limits)
    
    # Generate LaTeX table
    latex = "\\begin{table}[h]\n\\centering\n"
    
    # Table header
    num_cols = len(sorted_ratios) + 1  # +1 for market_limit column
    latex += f"\\begin{{tabular}}{{{'c' * num_cols}}}\n"
    latex += "\\hline\n"
    
    # Column headers
    header = "Market Limit"
    for ratio in sorted_ratios:
        header += f" & {ratio}"
    latex += header + " \\\\\n\\hline\n"
    
    # Data rows
    for market_limit in sorted_market_limits:
        row = f"{market_limit}"
        for ratio in sorted_ratios:
            if ratio in table_data[market_limit]:
                mean, std = table_data[market_limit][ratio]
                row += f" & ${mean:.2f} \\pm {std:.2f}$"
            else:
                row += " & --"  # Missing data
        latex += row + " \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Gini coefficient statistics across ratios and market limits}\n"
    latex += "\\label{tab:gini_ratios_markets}\n"
    latex += "\\end{table}"
    
    return latex


# # %%
# model = init_azure_model(secrets_path='assets/secrets.yaml')
# model.invoke('test')
# # %%
