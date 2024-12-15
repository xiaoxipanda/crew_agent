# Import necessary modules from LangChain
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from dotenv import load_dotenv

# os.environ["TAVILY_API_KEY"]
# os.environ["OPENAI_API_KEY"]
# os.environ["LANGCHAIN_API_KEY"]
# os.environ['LANGCHAIN_PROJECT']
# os.environ['OPENAI_API_BASE']
load_dotenv(dotenv_path="../../../.env")

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

# Define the tools the agent will use
tools = [TavilySearchResults(max_results=1)]

# Choose the LLM (Large Language Model) to use
llm = OpenAI()

# Construct the ReAct agent using the LLM, tools, and prompt
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Invoke the agent executor with a specific input
response = agent_executor.invoke({"input": "What is TikTok？"})

# Print the response
print(response)
