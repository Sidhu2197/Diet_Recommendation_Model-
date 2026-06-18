import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from pydantic import BaseModel
from schemas import input_schema, output_schema
from schemas.input_schema import DietRecommendationRequest
from schemas.output_schema import DietRecommendationResponse, FoodRecommendation
app = FastAPI()

# Load data
df = pd.read_excel("food_600.xlsx")
df["meal_type"] = df["meal_type"].str.lower().str.strip()
df["diet_type"] = df["diet_type"].str.lower().str.strip()


# Core Calculations
def calculate_bmr(u):
    if u["gender"] == "Male":
        return (10*u["weight"]) + (6.25*u["height"]) - (5*u["age"]) + 5
    else:
        return (10*u["weight"]) + (6.25*u["height"]) - (5*u["age"]) - 161

def calculate_targets(u):
    bmr = calculate_bmr(u)
    total_cal = bmr + 400 if u["goal"] == "Muscle Gain" else bmr - 400
    protein = u["weight"] * (1.8 if u["goal"] == "Muscle Gain" else 1.5)

    carbs = (total_cal * 0.45) / 4
    fat   = (total_cal * 0.25) / 9

    split = {"breakfast": 0.3, "lunch": 0.4, "dinner": 0.3}
    r = split[u["meal_type"]]

    return [
        total_cal * r,
        protein * r,
        carbs * r,
        fat * r
    ]

# function for recommendation 
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


# API route
@app.post("/recommend", response_model=DietRecommendationResponse)
def get_recommendations(user: DietRecommendationRequest):
    user_dict = user.model_dump()

    result = recommend(df, user_dict)

    if result.empty:
        return {"recommendations": []}

    return {
        "recommendations": result.to_dict(orient="records")
    }