import warnings
import os

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

# os.environ["OPENAI_API_KEY"]
# os.environ['OPENAI_API_BASE']
load_dotenv(dotenv_path="../../../.env")

# Create agents
analysis_agent1 = Agent(
    role="Mathematician",
    goal="Analyze data and provide insights.",
    backstory="You are an experienced mathematician with experience in statistics.",
    verbose=True
)

analysis_agent2 = Agent(
    role="Mathematician",
    goal="Analyze data and provide insights.",
    backstory="You are an experienced mathematician with experience in statistics.",
    verbose=True
)

# Create tasks
data_analysis_task1 = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent1,
    expected_output="Provide the dataset first and thent the the average age of the participants.",
    async_execution=True
)

# Create a task that requires code execution
data_analysis_task2 = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent2,
    expected_output="Provide the dataset first and thent the the average age of the participants."
)

# Create a crew and add the task
analysis_crew1 = Crew(
    agents=[analysis_agent1],
    tasks=[data_analysis_task1]
)

analysis_crew2 = Crew(
    agents=[analysis_agent2],
    tasks=[data_analysis_task2]
)

# result_1 = await analysis_crew1.kickoff_async(inputs={"ages": [25, 30, 35, 40, 45]})
result_1 = analysis_crew1.kickoff_async(inputs={"ages": [25, 30, 35, 40, 45]})
result_2 = analysis_crew2.kickoff(inputs={"ages": [20, 25, 30, 35, 40]})

print("Async Crew Thread Output:", result_1)
print("Main Thread Output", result_2)
