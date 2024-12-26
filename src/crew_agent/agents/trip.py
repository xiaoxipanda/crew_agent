from crewai import Agent, Task
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from dotenv import load_dotenv

from crew_agent.agents.model.itinerary import Itinerary

load_dotenv(dotenv_path="../../../.env")

planner = Agent(
    role="Activity Planner",
    goal='Research and find cool things to do at the destination, including activities and events that match the '
         'traveler\'s interests and age group',
    backstory=(
        "You are skilled at creating personalized itineraries that cater to the specific preferences and demographics "
        "of travelers."
    ),
    tools=[SerperDevTool(), ScrapeWebsiteTool()],  # Example of custom tool, loaded at the beginning of file
    verbose=True,
    allow_delegation=False,
)

restaurant_scout = Agent(
    role="Restaurant Scout",
    goal='Find highly-rated restaurants and dining experiences at the destination, and recommend scenic locations and fun activities',
    backstory=(
        "As a food lover, you know the best spots in town for a delightful culinary experience. You also have a knack for finding picturesque and entertaining locations."
    ),
    tools=[SerperDevTool(), ScrapeWebsiteTool()],  # Example of custom tool, loaded at the beginning of file
    verbose=True,
    allow_delegation=False,
)

itinerary_compiler = Agent(
    role="Itinerary Compiler",
    goal='Compile all researched information into a comprehensive day-by-day itinerary, ensuring the integration of flights and hotel information',
    backstory=(
        "With an eye for detail, you organize all the information into a coherent and enjoyable travel plan."
    ),
    tools=[SerperDevTool()],
    verbose=True,
    allow_delegation=False,
)

restaurant_scenic_location_scout_task = Task(
    description=(
        '''
        Find highly-rated restaurants and dining experiences at {destination}.
        Recommend scenic locations and fun activities that align with the traveler's preferences.
        Use internet search tools, restaurant review sites, and travel guides.
        Make sure to find a variety of options to suit different tastes and budgets, and ratings for them.
    
        Traveler's information:
        
        - origin: {origin}
    
        - destination: {destination}
    
        - age of the traveler: {age}
    
        - hotel localtion: {hotel_location}
    
        - flight infromation: {flight_information}
    
        - how long is the trip: {trip_duration}
          '''
    ),
    agent=restaurant_scout,
    expected_output='''
    A list of recommended restaurants, scenic locations, and fun activities for each day of the trip.
    Each entry should include the name, location (address), type of cuisine or activity, and a brief description and ratings.
    ''',
)

personalized_activity_planning_task = Task(
    description=(
        '''
        Research and find cool things to do at {destination}.
        Focus on activities and events that match the traveler's interests and age group.
        Utilize internet search tools and recommendation engines to gather the information.
    
    
        Traveler's information:

        - origin: {origin}
    
        - destination: {destination}
    
        - age of the traveler: {age}
    
        - hotel localtion: {hotel_location}
    
        - flight infromation: {flight_information}
    
        - how long is the trip: {trip_duration}
          '''
    ),
    agent=planner,
    expected_output='''
    A list of recommended activities and events for each day of the trip.
    Each entry should include the activity name, location, a brief description, and why it's suitable for the traveler.
    And potential reviews and ratings of the activities.
    ''',
)

itinerary_compilation_task = Task(
    description=(
        '''
        Compile all researched information into a comprehensive day-by-day itinerary for the trip to {destination}.
        Ensure the itinerary integrates flights, hotel information, and all planned activities and dining experiences.
        Use text formatting and document creation tools to organize the information.
        '''
    ),
    agent=planner,
    expected_output='''
    A detailed itinerary document, the itinerary should include a day-by-day 
    plan with flights, hotel details, activities, restaurants, and scenic locations.
    ''',
    output_json=Itinerary
)

from crewai import Crew, Process

# Form the crew and define the process
crew = Crew(
    agents=[planner, restaurant_scout, itinerary_compiler],
    tasks=[personalized_activity_planning_task, restaurant_scenic_location_scout_task, itinerary_compilation_task],
    process=Process.sequential,
    verbose=True
)

inputs = {
    'origin': 'China, Shenzhen',
    'destination': 'China, Shanghai',
    'age': 29,
    'hotel_location': 'Shanghai',
    'flight_information': 'leaving at December 28th, 2024, 7:00',
    'trip_duration': '7 days'
}

# Kick off the crew
result = crew.kickoff(inputs=inputs)
print(result)
