import warnings
import os

from IPython.core.display_functions import display
from dotenv import load_dotenv
from IPython.display import Markdown

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

from crewai import Agent, Task, Crew, Process

# os.environ['OPENAI_API_BASE']
# os.environ["OPENAI_API_KEY"]
load_dotenv(dotenv_path="../../../.env")

# Create an agent with code execution enabled
analysis_agent = Agent(
    role="mathematician",
    goal="Analyze data and provide insights.",
    backstory="You are an experienced mathematician with experience in statistics.",
    verbose=True
)

# Create a task that requires code execution
data_analysis_task = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent,
    expected_output="Provide the dataset first and thent the the average age of the participants."
)

# Create a crew and add the task
analysis_crew = Crew(
    agents=[analysis_agent],
    tasks=[data_analysis_task]
)

# List of datasets to analyze
datasets = [
    {"ages": [25, 30, 35, 40, 45]},
    {"ages": [20, 25, 30, 35, 40]},
    {"ages": [30, 35, 40, 45, 50]}
]

result = analysis_crew.kickoff_for_each(inputs=datasets)

for crew_output in result:
    result_markdown = crew_output.raw
    display(Markdown(result_markdown))
