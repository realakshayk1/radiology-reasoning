import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-5.4-mini-2026-03-17"

try:
    print(f"Testing model: {model}")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello, respond with a JSON object: {\"status\": \"ok\"}"}],
        response_format={"type": "json_object"}
    )
    print("Success!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
