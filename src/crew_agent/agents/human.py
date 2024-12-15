import warnings
import os

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool, YoutubeVideoSearchTool

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

# os.environ["OPENAI_API_KEY"]
# os.environ['OPENAI_API_BASE']
load_dotenv(dotenv_path="../../../.env")

# Create a search tool
search_tool = YoutubeVideoSearchTool(youtube_video_url='https://www.youtube.com/watch?v=wjZofJX0v4M')

# Define the research agent
researcher = Agent(
    role='Video Content Researcher',
    goal='Extract key insights from YouTube videos on AI advancements',
    backstory=(
        "You are a skilled researcher who excels at extracting valuable insights from video content. "
        "You focus on gathering accurate and relevant information from YouTube to support your team."
    ),
    verbose=True,
    tools=[search_tool],
    memory=True
)

# Define the writing agent
writer = Agent(
    role='Tech Article Writer',
    goal='Craft an article based on the research insights',
    backstory=(
        "You are an experienced writer known for turning complex information into engaging and accessible articles. "
        "Your work helps make advanced technology topics understandable to a broad audience."
    ),
    verbose=True,
    tools=[search_tool],  # The writer may also use the YouTube tool for additional context
    memory=True
)

# Create the research task
research_task = Task(
    description=(
        "Research and extract key insights from YouTube regarding Educative. "
        "Compile your findings in a detailed summary."
    ),
    expected_output='A summary of the key insights from YouTube',
    agent=researcher
)

# Create the writing task
writing_task = Task(
    description=(
        "Using the summary provided by the researcher, write a compelling article on what is Educative. "
        "Ensure the article is well-structured and engaging for a tech-savvy audience. must use chinese."
    ),
    expected_output='A well-written article on Educative based on the YouTube video research.',
    agent=writer,
    human_input=True  # Allow for human feedback after the draft
)

from crewai import Crew

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True,
    memory=True
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)

from IPython.display import Markdown

# Convert the CrewOutput object to a Markdown string
result_markdown = result.raw

# Display the result as Markdown
Markdown(result_markdown)
