"""
LLM Client — interface for talking to the xAI (Grok) model.

Key concepts:
    - System prompt: Sets the model's role and behavior (constant per task).
    - User prompt: The actual question or data (changes per call).
    - Temperature: 0 = deterministic, 1 = creative. Low for analysis.
    - Structured output: JSON format for machine-readable responses.
"""

import json
from openai import OpenAI
from app.config import XAI_API_KEY, XAI_BASE_URL


def get_client() -> OpenAI:
    """Create an OpenAI client pointed at xAI's server."""
    return OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    model: str = "grok-3-mini-fast",
    expect_json: bool = True,
) -> dict | str:
    """
    Send a prompt to the LLM and get a response.

    Returns:
        Parsed JSON dict if expect_json=True, raw string otherwise.
    """
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"} if expect_json else None,
    )

    raw_text = response.choices[0].message.content

    if expect_json:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"Warning: LLM did not return valid JSON. Raw response:\n{raw_text[:500]}")
            return {"error": "Invalid JSON response", "raw": raw_text}

    return raw_text
