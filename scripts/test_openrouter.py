
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from app.config import get_settings


def main():
    settings = get_settings()

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env")

    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    response = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[
            {
                "role": "user",
                "content": "Say hello in one short sentence.",
            }
        ],
        temperature=0,
        max_tokens=100,
    )

    print("Model:", response.model)
    print("Finish reason:", response.choices[0].finish_reason)
    print("Message:", repr(response.choices[0].message))
    print("Usage:", response.usage)


if __name__ == "__main__":
    main()
