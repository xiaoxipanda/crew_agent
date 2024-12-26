from pydantic import BaseModel, Field
from typing import List, Optional

class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    location: str = Field(..., description="Location of the activity")
    description: str = Field(..., description="Description of the activity")
    date: str = Field(..., description="Date of the activity")
    cousine: str = Field(..., description="Cousine of the restaurant")
    why_its_suitable: str = Field(..., description="Why it's suitable for the traveler")
    reviews: Optional[List[str]] = Field(..., description="List of reviews")
    rating: Optional[float] = Field(..., description="Rating of the activity")

class DayPlan(BaseModel):
    date: str = Field(..., description="Date of the day")
    activities: List[Activity] = Field(..., description="List of activities")
    restaurants: List[str] = Field(..., description="List of restaurants")
    flight: Optional[str] = Field(None, description="Flight information")
class Itinerary(BaseModel):
    name: str = Field(..., description="Name of the itinerary, something funny")
    day_plans: List[DayPlan] = Field(..., description="List of day plans")
    hotel: str = Field(..., description="Hotel information")