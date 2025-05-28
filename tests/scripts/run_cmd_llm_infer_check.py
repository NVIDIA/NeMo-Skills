from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url=f"http://0.0.0.0:5000/v1", timeout=None)
api_model = client.models.list().data[0].id

response = client.chat.completions.create(
    model=api_model,
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.0,
    max_tokens=4,
    top_p=1.0,
    n=1,
    stream=False,
)
print(response.choices[0].message.content)
