import warnings
import os

from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

# os.environ["OPENAI_API_KEY"]
# os.environ["SERPER_API_KEY"]
# os.environ['OPENAI_API_BASE']
load_dotenv(dotenv_path="../../../.env")

# Create a search tool
search_tool = SerperDevTool()

venue_finder = Agent(
    role="Conference Venue Finder",
    goal="Find the best venue for the upcoming conference",
    backstory=(
        "You are an experienced event planner with a knack for finding the perfect venues. "
        "Your expertise ensures that all conference requirements are met efficiently. "
        "Your goal is to provide the client with the best possible venue options."
    ),
    tools=[search_tool],
    verbose=True
)

venue_quality_assurance_agent = Agent(
    role="Venue Quality Assurance Specialist",
    goal="Ensure the selected venues meet all quality standards and client requirements",
    backstory=(
        "You are meticulous and detail-oriented, ensuring that the venue options provided "
        "are not only suitable but also exceed the client's expectations. "
        "Your job is to review the venue options and provide detailed feedback."
    ),
    tools=[search_tool],
    verbose=True
)

find_venue_task = Task(
    description=(
        "Conduct a thorough search to find the best venue for the upcoming conference in Shenzhen, China. "
        "Consider factors such as capacity, location, amenities, and pricing. "
        "Use online resources and databases to gather comprehensive information."
    ),
    expected_output=(
        "A list of 5 potential venues with detailed information on capacity, location, amenities, pricing, and availability."
    ),
    tools=[search_tool],
    agent=venue_finder,
)

quality_assurance_review_task = Task(
    description=(
        "Review the venue options provided by the Conference Venue Finder. "
        "Ensure that each venue meets all the specified requirements and standards. "
        "Provide a detailed report on the suitability of each venue."
    ),
    expected_output=(
        "A detailed review of the 5 potential venues, highlighting any issues, strengths, and overall suitability."
    ),
    tools=[search_tool],
    agent=venue_quality_assurance_agent,
)

event_planning_crew = Crew(
  agents=[venue_finder, venue_quality_assurance_agent],
  tasks=[find_venue_task, quality_assurance_review_task],
  verbose=True
)

inputs = {
    "conference_name": "AI Innovations Summit",
    "requirements": "Capacity for 5000, central location, modern amenities, budget up to $50,000"
}

result = event_planning_crew.kickoff(inputs=inputs)