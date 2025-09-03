import os
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def call_llm(prompt: str) -> str:
    """Call OpenAI ChatCompletion (placeholder)."""
    if not OPENAI_API_KEY:
        return f"[DEMO MODE] You said: '{prompt}'. This would normally be processed by an AI assistant, but no OpenAI API key is configured. Please set OPENAI_API_KEY environment variable to enable AI responses."
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {
        "model": "gpt-4o-mini",  # Fixed model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return f"[ERROR] Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable."
        else:
            return f"[ERROR] OpenAI API error: {e.response.status_code}"
    except Exception as e:
        return f"[ERROR] Failed to call OpenAI API: {str(e)}"
