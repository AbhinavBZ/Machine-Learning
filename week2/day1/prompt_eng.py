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

def llm_ans(prompt):
    message={
        "role":"user",
        "content":prompt
    }
    messages=[message]

    response=client.chat.completions.create(model=model,messages=messages)
    return response.choices[0].message.content

prompt="""
#ROLE:
You are a support assistent at a mobile/laptop repairing company.
#TASK:
You have to classify the issue in a category
#CONSTRAINT:
You have to classify the issue in three categories namely Biling, Technical, Return.
#OUTPUT FORMAT:
Your answer should be in one word. One word should be one of the categories given in constraint.
#EXAMPLE:
for instance if a user complain says he wants a refund the the category is Return.
#FALLBACK:
If the issue is unrelated to any of the categories mentioned in constraint, then the answer should be Other.
This is a user Complaint:
My laptop is not working.
"""
print(llm_ans(prompt))
