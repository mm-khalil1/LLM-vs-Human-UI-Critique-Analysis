import os
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_MODEL_PRESETS = {
    "openai": {
        "model_short": "openrouter_openai_gpt5_2",
        "model_version": "openai/gpt-5.2",
    },
    "claude": {
        "model_short": "openrouter_claude_sonnet_4_5",
        "model_version": "anthropic/claude-sonnet-4.5",
    },
    "gemini": {
        "model_short": "openrouter_gemini_2_5_pro",
        "model_version": "google/gemini-2.5-pro",
    },
    "deepseek": {
        "model_short": "openrouter_deepseek_r1_0528",
        "model_version": "deepseek/deepseek-r1-0528",
    },
    "qwen": {
        "model_short": "openrouter_qwen3_vl_32b",
        "model_version": "qwen/qwen3-vl-32b-instruct",
    },
}

# Every model used in a run, keyed by model_short: the name used in batch folders and result files.
MODELS = {
    "gpt5": {"provider": "openai", "model_version": "gpt-5-2025-08-07"},
    "gpt5_2": {"provider": "openai", "model_version": "gpt-5.2-2025-12-11"},
    "deepseek3_2": {"provider": "deepseek", "model_version": "deepseek-reasoner"},
    "qwen3_vl": {"provider": "alibaba", "model_version": "qwen-vl-max"},
    "qwen3_5": {"provider": "alibaba", "model_version": "qwen3.5-plus"},
    "claude4": {"provider": "anthropic", "model_version": "claude-sonnet-4-20250514"},
    "claude4_7": {"provider": "anthropic", "model_version": "claude-opus-4-7"},
    "gemini2_5pro": {"provider": "google", "model_version": "gemini-2.5-pro"},
    "gemini3_1pro": {"provider": "google", "model_version": "gemini-3.1-pro-preview"},
    "gemini_3_7_flash": {"provider": "google", "model_version": "gemini-3.7-flash"},
}

# Model used when a caller names only the provider.
DEFAULT_MODELS = {
    "openai": "gpt5_2",
    "deepseek": "deepseek3_2",
    "alibaba": "qwen3_5",
    "anthropic": "claude4_7",
    "google": "gemini_3_7_flash",
}

PROVIDERS = {
    "openai": {
        "env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/",
        "client_type": "openai",
        "rpm": 5000,
    },
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/",
        "client_type": "openai",
        "rpm": 2000,
    },
    "alibaba": {
        "env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "client_type": "openai",
        "rpm": 600,
    },
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
        "rpm": 2000,
    },
    "google": {
        "env": "GOOGLE_API_KEY",
        "client_type": "google",
        "rpm": 150,
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        **OPENROUTER_MODEL_PRESETS["openai"],
        "client_type": "openai",
        "rpm": 60,
        "default_headers": {
            "HTTP-Referer": "OPENROUTER_HTTP_REFERER",
            "X-OpenRouter-Title": "OPENROUTER_APP_TITLE",
        },
    },
}

# Keep each provider's default model readable from PROVIDERS as before.
for _provider, _model_short in DEFAULT_MODELS.items():
    PROVIDERS[_provider].update(model_short=_model_short, model_version=MODELS[_model_short]["model_version"])


def _resolve_header_env_vars(header_env_vars: dict[str, str]) -> dict[str, str]:
    return {
        header: value
        for header, env_var in header_env_vars.items()
        if (value := os.getenv(env_var))
    }


def resolve_openrouter_model(model: str = "", model_short: str = "") -> tuple[str, str]:
    if not model:
        preset = OPENROUTER_MODEL_PRESETS["openai"]
        return preset["model_version"], model_short or preset["model_short"]
    if model in OPENROUTER_MODEL_PRESETS:
        preset = OPENROUTER_MODEL_PRESETS[model]
        return preset["model_version"], model_short or preset["model_short"]
    normalized_model_short = (
        model
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    return model, model_short or f"openrouter_{normalized_model_short}"

def resolve_registered_model(provider: str, model: str = "", model_short: str = "") -> tuple[str, str]:
    """Return (model_version, model_short) for a provider, filling whichever is missing from MODELS.

    Passing only model_short selects a registered model, passing only model looks its model_short up,
    and passing neither gives the provider's default model.
    """
    if not model and not model_short:
        model_short = DEFAULT_MODELS[provider]

    if model_short in MODELS:
        entry = MODELS[model_short]
        if entry["provider"] != provider or (model and model != entry["model_version"]):
            raise ValueError(f"model_short {model_short!r} is {entry['provider']}/{entry['model_version']}, not {provider}/{model or '<default>'}.")
        return entry["model_version"], model_short

    if model and not model_short:
        matches = [short for short, entry in MODELS.items() if entry == {"provider": provider, "model_version": model}]
        if not matches:
            raise ValueError(f"Model {provider}/{model} is not in MODELS; add it with its model_short.")
        return model, matches[0]

    if not model:
        raise ValueError(f"Unknown model_short {model_short!r}; add it to MODELS.")
    return model, model_short


def resolve_model_config(provider: str, model: str = "", model_short: str = "") -> dict:
    provider_key = "google" if provider == "gemini" else provider
    if provider_key not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    info = PROVIDERS[provider_key]
    if provider_key == "openrouter":
        resolved_model, resolved_model_short = resolve_openrouter_model(model, model_short)
    else:
        resolved_model, resolved_model_short = resolve_registered_model(provider_key, model, model_short)

    return {
        **info,
        "provider": provider_key,
        "requested_provider": provider,
        "model_version": resolved_model,
        "model_short": resolved_model_short,
    }

# --- GEMINI ---
def setup_gemini(*, model="", model_short="", system_message:str, temperature=1.0, max_tokens=4096, provider="google"):
    model_cfg = resolve_model_config(provider, model=model, model_short=model_short)
    info = PROVIDERS[model_cfg["provider"]]
    env_key = info["env"]

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
        "provider": model_cfg["provider"],
        "model_version": model_cfg["model_version"],
        "model_short": model_cfg["model_short"],
        "rpm": info["rpm"],
        "config": config,
    }
    return client, cfg

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
def setup_anthropic(*, model="", model_short="", system_message:str, temperature=1.0, max_tokens=5000, provider="anthropic"):
    model_cfg = resolve_model_config(provider, model=model, model_short=model_short)
    info = PROVIDERS[model_cfg["provider"]]
    env_key = info["env"]

    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"Missing environment variable: {env_key}")

    client = Anthropic(api_key=api_key)
    cfg = {
        "provider": model_cfg["provider"],
        "model_version": model_cfg["model_version"],
        "model_short": model_cfg["model_short"],
        "rpm": info["rpm"],
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
                    'media_type': f"image/{query_args['image_format']}",
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
                "image_url": {"url": f"data:image/{query_args['image_format']};base64,{row.base64_screen}"}
            })

    # Add target screen image
    if base64_target:
        content_array.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{query_args['image_format']};base64,{base64_target}"}
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

def setup_client(
        provider,
        *,
        model="",
        model_short="",
        system_message="",
        temperature=1.0,
        max_tokens=4000,
    ):
    
    cfg = resolve_model_config(provider, model=model, model_short=model_short)
    info = PROVIDERS[cfg["provider"]]

    api_key = os.getenv(info["env"])
    if not api_key:
        raise ValueError(f"Missing env: {info['env']}")

    # ------------------------- OpenAI-compatible -------------------------
    if info["client_type"] == "openai":
        default_headers = _resolve_header_env_vars(info.get("default_headers", {}))
        client = OpenAI(api_key=api_key, base_url=info["base_url"], default_headers=default_headers)
        return client, cfg

    # ------------------------- Anthropic -------------------------
    if info["client_type"] == "anthropic":
        client = Anthropic(api_key=api_key)
        return client, cfg

    # ------------------------- Google Gemini -------------------------
    if info["client_type"] == "google":
        client = genai.Client(api_key=api_key)
        cfg = {
            **cfg,
            "config": types.GenerateContentConfig(
                system_instruction=system_message,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        }
        return client, cfg

    raise ValueError("Unknown provider type")
