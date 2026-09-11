import requests
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool


@tool('get_weather', return_direct=False, description='Get the current weather for a given city.')
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


llm = ChatOllama(
    model="qwen3:0.6b",
    temperature=0,
    reasoning=False
)
llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke(
    "What is the weather like in New York City?"
)

print(response)
print("Content:", response.content)
print("----------------------------------")
print("Tool calls:", response.tool_calls)
