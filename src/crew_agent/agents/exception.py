import warnings
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
import os

from dotenv import load_dotenv

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

# os.environ["OPENAI_API_KEY"]
# os.environ['OPENAI_API_BASE']
# os.environ['SERPER_API_KEY']
load_dotenv(dotenv_path="../../../.env")

# Create a search tool
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Data Researcher Agent using Gemini and SerperSearch
article_researcher = Agent(
    role="Senior Researcher",
    goal='Unccover details regarding the {topic}. If no topic is given, then wait for it to be provided by the user or ask the other agent. Do not choose a topic on your own!',
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "research, eager to explore and share knowledge about the topic. However you are unable to think a topic on your own and need specific topics before you can research"
    ),
    tools=[search_tool, ]

)

# Article Writer Agent using GPT
article_writer = Agent(
    role='Senior Writer',
    goal='Narrate compelling tech stories about {topic}. If no topic is given, then wait for it to be provided by the user or ask the other agent. Do not choose a topic on your own!',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner. However you are unable to think a topic on your own and need specific topics before you can write."
    ),
    allow_delegation=True
)

# Research Task
research_task = Task(
    description=(
        "Conduct a thorough analysis on the given {topic}."
        "Utilize the search tool for any necessary online research."
        "Summarize key findings in a detailed report."
    ),
    expected_output='A detailed report on the data analysis with key insights.',
    tools=[search_tool, scrape_tool],
    agent=article_researcher,
)

# Writing Task
writing_task = Task(
    description=(
        "Write an insightful article based on the data analysis report. "
        "The article should be clear, engaging, and easy to understand."
    ),
    expected_output='A 6-paragraph article summarizing the data insights.',
    agent=article_writer,
)

from crewai import Crew, Process

# Form the crew and define the process
crew = Crew(
    agents=[article_researcher, article_writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

research_inputs = {
    'topic': ''
}

# Kick off the crew
result = crew.kickoff(inputs=research_inputs)
