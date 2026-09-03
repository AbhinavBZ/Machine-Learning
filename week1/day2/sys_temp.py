import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"
role="user"
prompt="Suggest a name for my cloth company."

#System

message_system={
    "role":"system",
    "content":"You are a brand manager who suggests name for my cloth company. Name should be in one word. Suggest one name only."
}
message={
    "role":role,
    "content":prompt
}
messages=[message_system, message]

#Temperature_set

response=client.chat.completions.create(model=model,messages=messages, temperature=1)
print(response.choices[0].message.content)
