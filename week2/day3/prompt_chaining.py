import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"


JD="""
We are hiring a Backend Python Developer.

Requirements-
- Strong Pyhton
- FastAPI or Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- 2+ years of experience
"""
RESUME="""
Name: Rahul Sharma

Experience:
3 years as a Software Developer.

Skills:
Python, FastAPI, MySQL, Docker, REST APIs , Git

Projects:
Built a food delievery backend using fastAPI and MySQL

Deployed applications using Docker.
"""
def ask_llm(system_prompt, user_prompt):
    sys_message={
        "role":"system",
        "content":system_prompt
    }
    user_message={
        "role":"user",
        "content":user_prompt
    }
    messages=[sys_message,user_message]
    response=client.chat.completions.create(model=model,messages=messages)
    answer=response.choices[0].message.content
    return answer

def step1_res_extract():
    #extract skills from resume
    system_prompt=f"""
    You are a professional HR assistant. Extract the skills from the candidates resume provided.
    Only return the skills, no other information. Do not invent any skills by yourself.
    """

    user_prompt=f"""
    Extract the skills from this resume
    {RESUME} 
    """
    return ask_llm(system_prompt,user_prompt)

def step2_jd_extract():
    #extract skills from resume
    system_prompt=f"""
    You are a professional HR assistant. Extract the skills from the candidates JD provided.
    Only return the skills, no other information. Do not invent any skills by yourself.
    """

    user_prompt=f"""
    Extract the skills from this JD
    {JD} 
    """
    return ask_llm(system_prompt,user_prompt)
def step3_match(candidate,jd):
    system_prompt=f"""
        You are a professional HR assistant. Compare the skills of the candidate and the skills required in the JD and produce a final score between 0 and 100. also produce a short verdict whether the candiadte is good fit or not for the role.
        """
    user_prompt=f"""
        Compare and match the skills
        JD:
        {jd}
        Candidate:
        {candidate}
        """
    return ask_llm(system_prompt, user_prompt)
candidate=step1_res_extract()
sleep(2)
jd=step2_jd_extract()
sleep(2)
score=step3_match(candidate, jd)
print(score)
