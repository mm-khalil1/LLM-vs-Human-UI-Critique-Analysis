import os
import traceback
from typing import Any, Dict
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic

MODEL_REGISTRY = {
    "openai": ("gpt5", "OPENAI_API_KEY"),
    "google": ("gemini-3-pro", "GOOGLE_API_KEY"),
    "anthropic": ("claude-4", "ANTHROPIC_API_KEY"),
    "deepseek": ("deepseek-reasoner", "DEEPSEEK_API_KEY"),
    "alibaba": ("qwen3-vl", "DASHSCOPE_API_KEY"),
}

# --- GEMINI ---
def setup_gemini(*, model="", system_message:str, temperature=1.0, max_tokens=4096, provider="google"):
    
    if provider not in MODEL_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}")

    model_short, env_key = MODEL_REGISTRY[provider]

    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"Missing environment variable: {env_key}")
    
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_message,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    cfg = {
        "provider": "google",
        "model_version": model,
        "model_short": model_short,
        "rpm": 150,
        "config": config,
    }
    return client, cfg

    # import google.generativeai as genai
    # api_key = os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise ValueError("Set GOOGLE_API_KEY")
    # genai.configure(api_key=api_key)

    # safety_settings = [
    #     {"category": "HARM_CATEGORY_DANGEROUS",           "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_HARASSMENT",          "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_HATE_SPEECH",         "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",   "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",   "threshold": "BLOCK_NONE"},
    # ]
    # generation_config = {
    #     "temperature": float(temperature),
    #     "max_output_tokens": int(max_tokens),
    # }
    # model_obj = genai.GenerativeModel(
    #     model_name=model,
    #     safety_settings=safety_settings,
    #     system_instruction=system_message,
    #     generation_config=generation_config,
    # )
    # cfg = {
    #     "provider": "google",
    #     "model_version": model,
    #     "model_short": "gemini2_5pro",
    #     "rpm": 150,
    # }
    # return model_obj, cfg

def query_gemini_model(client, prompt, base64_target, query_args, samples_df):
    def build_gemini_contents(prompt, base64_target, samples_df, image_format="jpeg"):
        contents = [prompt]

        if samples_df is not None:
            for row in samples_df.itertuples(index=False):
                contents.append(types.Part.from_bytes(data=row.base64_screen, mime_type=f"image/{image_format}"))

        # Append the target image
        contents.append(types.Part.from_bytes(data=base64_target, mime_type=f"image/{image_format}"))

        return contents
    
    contents = build_gemini_contents(prompt, base64_target, samples_df, query_args["image_format"])

    try:
        response = client.models.generate_content(
            model=query_args["model_version"],
            config=query_args["config"],
            contents=contents
            )
        if response.text is None:
            print("⚠️ Request Terminated.", response.candidates[0].finish_reason)
            return None
        return response.text
    except Exception:
        raise

# --- ANTHROPIC ---
def setup_anthropic(*, model="", system_message:str, temperature=1.0, max_tokens=5000, provider="anthropic"):
    if provider not in MODEL_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}")

    model_short, env_key = MODEL_REGISTRY[provider]

    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"Missing environment variable: {env_key}")

    client = Anthropic(api_key=api_key)
    cfg = {
        "provider": provider,
        "model_version": model,
        "model_short": model_short,
        "rpm": 2000,
    }
    return client, cfg

def query_claude_model(
    client,
    prompt,
    base64_target,
    query_args,
    samples_df=None,
    assistant=None,
):
    """
    Sends a prompt and images to the specified large language model and returns its response.

    Args:
        client: Anthropic client instance.
        model_name: Model name string.
        prompt: Prompt text.
        base64_target: Base64 string of the target screen image.
        examples_df: Optional DataFrame with "base64_screen" column for few-shot samples.
        max_tokens: Max tokens for response.
        temperature: Sampling temperature.
        system: Optional system message.
        assistant: Optional assistant message.
        image_media_type: MIME type for images ("png" or "jpeg").
    """
    content_array = []

    # Add prompt text
    content_array.append({'type': 'text', 'text': prompt})

    # Add few-shot sample images if provided
    if samples_df is not None:
        for row in samples_df.itertuples(index=False):
            content_array.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': f"image/{query_args["image_format"]}",
                    'data': row.base64_screen
                    }})

    # Add target screen image
    content_array.append({
        'type': 'image',
        'source': {
            'type': 'base64',
            'media_type': "image/" + query_args["image_format"],
            'data': base64_target
        }})

    messages = [{'role': 'user', 'content': content_array}]
    if assistant is not None:
        messages.append({'role': 'assistant', 'content': assistant})

    try:
        response = client.messages.create(
            model=query_args["model_version"],
            max_tokens=query_args["max_tokens"],
            temperature=query_args["temperature"],
            system=query_args["system"],
            messages=messages,
        )
        return response.content[0].text

    except Exception:
        raise

# --- OPENAI ---
def setup_openai(
        *, 
        provider = "openai",
        model="gpt-5-2025-08-07", 
        system_message:str, 
        temperature=1.0, 
        max_tokens=4000, 
        base_url="https://api.openai.com/v1/"
        ):
    if provider not in MODEL_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}")

    model_short, env_key = MODEL_REGISTRY[provider]

    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"Missing environment variable: {env_key}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    cfg = {
        "provider": provider,
        "model_version": model,
        "model_short": model_short,
        "rpm": 5000,
    }
    return client, cfg

def query_openai_model(
    client: OpenAI,
    prompt: str,
    base64_target: str,
    query_args: dict,
    samples_df=None,
):
    """Sends a prompt and images to the specified OpenAI model and returns its response."""
    content_array = []

    # Add the prompt text
    content_array.append({"type": "text", "text": prompt})

    # Add few-shot sample images if provided
    if samples_df is not None:
        for row in samples_df.itertuples(index=False):
            content_array.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{query_args["image_format"]};base64,{row.base64_screen}"}
            })

    # Add target screen image
    if base64_target:
        content_array.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{query_args["image_format"]};base64,{base64_target}"}
        })

    try:
        response = client.chat.completions.create(
            model=query_args["model_version"],
            max_completion_tokens=query_args["max_tokens"],
            temperature=query_args["temperature"],
            messages=[
                {"role": "system", "content": query_args["system"]},
                {"role": "user", "content": content_array}
            ]
        )
        return response.choices[0].message.content
    except Exception:
        raise

PROVIDERS = {
    "openai": {
        "env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/",
        "model_short": "gpt5",
        "model_version": "gpt-5-2025-08-07",
        "client_type": "openai",
        "rpm": 5000,
    },
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/",
        "model_short": "deepseek3_2",
        "model_version": "deepseek-reasoner",
        "client_type": "openai",
        "rpm": 2000,
    },
    "qwen": {
        "env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model_short": "qwen3_vl",
        "model_version": "qwen-vl-max",
        "client_type": "openai",
        "rpm": 600,
    },
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "model_short": "claude4",
        "model_version": "claude-sonnet-4-20250514",
        "client_type": "anthropic",
        "rpm": 2000,
    },
    "google": {
        "env": "GOOGLE_API_KEY",
        "model_short": "gemini2_5pro",
        "model_version": "gemini-2.5-pro",
        "client_type": "google",
        "rpm": 150,
    },
}

def setup_client(
        provider,
        *,
        system_message="",
        temperature=1.0,
        max_tokens=4000,
    ):
    
    info = PROVIDERS[provider]

    api_key = os.getenv(info["env"])
    if not api_key:
        raise ValueError(f"Missing env: {info["env"]}")

    # ------------------------- OpenAI-compatible -------------------------
    if info["client_type"] == "openai":
        client = OpenAI(api_key=api_key, base_url=info["base_url"])
        return client, info

    # ------------------------- Anthropic -------------------------
    if info["client_type"] == "anthropic":
        client = Anthropic(api_key=api_key)
        return client, info

    # ------------------------- Google Gemini -------------------------
    if info["client_type"] == "google":
        client = genai.Client(api_key=api_key)
        cfg = {
            **info,
            "config": types.GenerateContentConfig(
                system_instruction=system_message,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        }
        return client, cfg

    raise ValueError("Unknown provider type")
