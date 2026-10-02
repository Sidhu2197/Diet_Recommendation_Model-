import sys
import os

# Add root dir to path so schemas can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

app = FastAPI(
    title="Diet Recommendation API",
    description="AI-powered diet recommendation using KNN",
    version="1.0.0"
)

# ─── Schemas (inlined to avoid import path issues on Vercel) ───────────────────

class DietRecommendationRequest(BaseModel):
    age: int
    gender: Literal["Male", "Female"]
    height: float
    weight: float
    goal: Literal["Muscle Gain", "Weight Loss"]
    meal_type: Literal["breakfast", "lunch", "dinner"]
    diet_type: str

class FoodRecommendation(BaseModel):
    food_name: str
    protein: float
    calories: float
    carbs: float
    fat: float

class DietRecommendationResponse(BaseModel):
    recommendations: list[FoodRecommendation]

# ─── Data Loading ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(os.path.join(BASE_DIR, "food_600.xlsx"))
df["meal_type"] = df["meal_type"].str.lower().str.strip()
df["diet_type"] = df["diet_type"].str.lower().str.strip()

# ─── Core Calculations ─────────────────────────────────────────────────────────

def calculate_bmr(u):
    if u["gender"] == "Male":
        return (10 * u["weight"]) + (6.25 * u["height"]) - (5 * u["age"]) + 5
    else:
        return (10 * u["weight"]) + (6.25 * u["height"]) - (5 * u["age"]) - 161

def calculate_targets(u):
    bmr = calculate_bmr(u)
    total_cal = bmr + 400 if u["goal"] == "Muscle Gain" else bmr - 400
    protein = u["weight"] * (1.8 if u["goal"] == "Muscle Gain" else 1.5)
    carbs = (total_cal * 0.45) / 4
    fat   = (total_cal * 0.25) / 9
    split = {"breakfast": 0.3, "lunch": 0.4, "dinner": 0.3}
    r = split[u["meal_type"]]
    return [total_cal * r, protein * r, carbs * r, fat * r]

def recommend(df, user, k=8):
    target = calculate_targets(user)
    filtered = df[
        (df["meal_type"] == user["meal_type"]) &
        (df["diet_type"] == user["diet_type"])
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    features = ["calories", "protein", "carbs", "fat"]
    X = filtered[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    user_scaled = scaler.transform([target])
    knn = NearestNeighbors(n_neighbors=min(k, len(filtered)))
    knn.fit(X_scaled)
    _, idx = knn.kneighbors(user_scaled)
    result = filtered.iloc[idx[0]]
    return result[["food_name", "protein", "calories", "carbs", "fat"]]

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Diet Recommendation API is running 🥗", "docs": "/docs"}

@app.post("/recommend", response_model=DietRecommendationResponse)
def get_recommendations(user: DietRecommendationRequest):
    user_dict = user.model_dump()
    result = recommend(df, user_dict)
    if result.empty:
        return {"recommendations": []}
    return {"recommendations": result.to_dict(orient="records")}
