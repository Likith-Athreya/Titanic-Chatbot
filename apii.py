import pandas as pd
import seaborn as sns
from fastapi import FastAPI
import uvicorn

def load_data():
    """Load Titanic dataset from seaborn."""
    return sns.load_dataset("titanic")

df = load_data()
app = FastAPI()

def analyze_query(query, df):
    """Analyze user query and return insights."""
    query = query.lower()
    if "survival rate" in query:
        return survival_rate(df)
    elif "age distribution" in query:
        return age_distribution(df)
    elif "class distribution" in query:
        return class_distribution(df)
    elif "gender distribution" in query:
        return gender_distribution(df)
    elif "average ticket fare" in query:
        return average_fare(df)
    elif "embarkation distribution" in query:
        return embarkation_distribution(df)
    else:
        return "I'm sorry, I didn't understand the question."

def survival_rate(df):
    rates = df['survived'].value_counts(normalize=True) * 100
    return rates.to_dict()

def age_distribution(df):
    return f"Median age: {df['age'].median()}"

def class_distribution(df):
    return df['class'].value_counts().to_dict()

def gender_distribution(df):
    return df['sex'].value_counts(normalize=True).to_dict()

def average_fare(df):
    return f"The average ticket fare was ${df['fare'].mean():.2f}."

def embarkation_distribution(df):
    return df['embark_town'].value_counts().to_dict()

@app.get("/query")
def query_titanic(question: str):
    response = analyze_query(question, df)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
