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

#structure it
from pydantic import BaseModel
class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema=Ticket.model_json_schema()
response_format={
    "type":"json_object"
}
system_prompt=f"""
Extract the personal information from the ticket based on this schema. Give me a json output.
{schema}
"""

message_system={
    "role":"system",
    "content":system_prompt
}

text="Hello my name is Abhinav Bhardwaj. I have recently purchased an iphone from your store and it stopped working. I live near Shiv Shakti madir , Belwatika. Your can contact me thhrough abhinav.bhardwaj1289@gmail.com or 9608933867."
prompt=f"""
This is a customer ticket. Please extract the personal information from this.
{text}
"""
message={
    "role":role,
    "content":prompt
}
messages=[message_system,message]

response=client.chat.completions.create(model=model,messages=messages,response_format=response_format)
print(response.choices[0].message.content)
