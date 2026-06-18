from pydantic import BaseModel
from typing import Literal


class DietRecommendationRequest(BaseModel):
    age: int
    gender: Literal["Male", "Female"]
    height: float
    weight: float
    goal: Literal["Muscle Gain", "Weight Loss"]
    meal_type: Literal["breakfast", "lunch", "dinner"]
    diet_type: str