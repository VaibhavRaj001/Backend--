from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from typing import List

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_methods=["*"],
    allow_headers=["*"],
)


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)


star_to_imdb = {
    "1 star": (0, 2, "Very Negative"),
    "2 stars": (3, 4, "Negative"),
    "3 stars": (5, 6, "Neutral"),
    "4 stars": (7, 8, "Positive"),
    "5 stars": (9, 10, "Very Positive"),
}

class ReviewRequest(BaseModel):
    texts: List[str]

def convert_imdb_rating(star_label, confidence):
    """Convert star rating to a weighted IMDb rating based on confidence score."""
    imdb_min, imdb_max, sentiment_category = star_to_imdb[star_label]
    imdb_rating = round(imdb_min + (imdb_max - imdb_min) * confidence)
    return imdb_rating, sentiment_category

def analyze_reviews(texts):
    """Batch analyze multiple reviews."""
    results = sentiment_pipeline(texts)
    analyzed_data = []

    for text, result in zip(texts, results):
        star_label = result["label"]
        confidence = result["score"]
        imdb_rating, sentiment_category = convert_imdb_rating(star_label, confidence)

        analyzed_data.append({
            "Review": text,
            "Sentiment": sentiment_category,
            "IMDb Rating": imdb_rating,
            "Stars": star_label,
            "Confidence": round(confidence, 2)
        })
    return analyzed_data

@app.post("/analyze/")
async def analyze_sentiment(request: ReviewRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No reviews provided.")
    
    results = analyze_reviews(request.texts)
    return {"results": results}

@app.post("/visualize/")
async def visualize_sentiment(request: ReviewRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No reviews provided.")
    
    results = analyze_reviews(request.texts)
    df = pd.DataFrame(results)

    plt.figure(figsize=(15, 10))
    heatmap = df.pivot_table(index=df.index + 1, columns='IMDb Rating', values='Confidence')
    sns.heatmap(heatmap, annot=True, linewidth=0.5, cmap="crest", vmin=0, vmax=10)
    plt.title("Sentiment Heatmap")

    heatmap_path = "heatmap.png"
    plt.savefig(heatmap_path, format="png")
    plt.close()


    sentiment_counts = df["Sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140,
        colors=['#4CAF50', '#FFC107', '#F44336', '#2196F3', '#FF5722'],
        wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 12}, shadow=True
    )
    plt.title("Sentiment Distribution of Reviews", fontsize=16, fontweight="bold")

    piechart_path = "piechart.png"
    plt.savefig(piechart_path, format="png")
    plt.close()

    print("Sentiment Analysis Results:", results)  

    return {
        "message": "Visualizations generated",
        "heatmap_url": "http://localhost:8000/heatmap",
        "piechart_url": "http://localhost:8000/piechart",
        "results": results,
    }

@app.get("/heatmap")
async def get_heatmap():
    file_path = "heatmap.png"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"error": "Heatmap not found"}

@app.get("/piechart")
async def get_piechart():
    file_path = "piechart.png"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"error": "Pie Chart not found"}
