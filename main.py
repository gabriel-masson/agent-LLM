import requests

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool


@tool(
    "get_weather",
    return_direct=False,
    description="Get the current weather for a city."
)
def get_weather(city: str):
    response = requests.get(
        f"https://wttr.in/{city}?format=j1"
    )

    return response.json()


@tool(
    "soma",
    return_direct=False,
    description="Sum two numbers."
)
def soma(a: int, b: int):
    return a + b


llm = ChatOllama(
    model="qwen3:0.6b",
    temperature=0,
    reasoning=False
)


agent = create_agent(
    model=llm,
    tools=[get_weather, soma],
)


response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "2 + 2 equal?"
        }
    ]
})


print(response)
