from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from decouple import config
import os
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


os.environ["OPENAI_API_KEY"] = "API KEY HERE"

OPENAI_API_KEY= os.getenv("API KEY HERE")
llm = ChatOpenAI(model_name='gpt-3.5-turbo')

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


db = SQLDatabase.from_uri("sqlite:///chinook.db")


db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

tools = [
    Tool(
        name="MathTool",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
    Tool(
        name="Product_Database",
        func=db_chain.run,
        description="useful for when you need to answer questions about products."
    )
]


agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)


def main(message):
    return agent.run(message)