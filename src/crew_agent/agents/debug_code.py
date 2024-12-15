import warnings
import os

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

# os.environ["OPENAI_API_KEY"]
# os.environ['OPENAI_API_BASE']
load_dotenv(dotenv_path="../../../.env")

# Create a debugging agent with code execution enabled
debugging_agent = Agent(
    role="Python Debugger",
    goal="Identify and fix issues in existing Python code",
    backstory="You are an experienced Python developer with a knack for finding and fixing bugs.",
    allow_code_execution=True,
    verbose=True
)

# Define a task that involves debugging the provided code
debug_task = Task(
    description=(
        "The following Python code is supposed to return the square of each number in the list, "
        "but it contains a bug. Please identify and fix the bug:\n"
        "```\n"
        "numbers = [2, 4, 6, 8]\n"
        "squared_numbers = [n*m for n in numbers]\n"
        "print(squared_numbers)\n"
        "```"
    ),
    agent=debugging_agent,
    expected_output="The corrected code should output the squares of the numbers in the list. Provide the updated code and tell what was the bug and how you fixed it."
)

# Form a crew and assign the debugging task
debug_crew = Crew(
    agents=[debugging_agent],
    tasks=[debug_task]
)

# Execute the crew and retrieve the result
result = debug_crew.kickoff()

from IPython.display import Markdown

# Convert the CrewOutput object to a Markdown string
result_markdown = result.raw

# Display the result as Markdown
Markdown(result_markdown)