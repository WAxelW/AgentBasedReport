from openai import OpenAI

# --------------------------- CONFIG ---------------------------------
OPENAI_API_KEY = "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"

# Models and deterministic settings
MODEL = "gpt-5-mini"
MAX_TOKENS = 800

client = OpenAI(api_key=OPENAI_API_KEY)

messages = [{"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"What is the capital of France?"}]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_completion_tokens=MAX_TOKENS,
)

print(response.choices[0].message.content)