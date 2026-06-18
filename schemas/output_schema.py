from pydantic import BaseModel

class FoodRecommendation(BaseModel):
    food_name: str
    protein: float
    calories: float
    carbs: float
    fat: float


class DietRecommendationResponse(BaseModel):
    recommendations: list[FoodRecommendation]