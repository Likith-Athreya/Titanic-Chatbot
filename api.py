from fastapi import FastAPI
import seaborn as sns
import pandas as pd

app = FastAPI()
df = sns.load_dataset("titanic")

def analyze_query(query, df):
    """Analyze user query and return insights."""
    query = query.lower()
    if "survival rate" in query:
        return f"Survival rate: {df['survived'].mean() * 100:.2f}%"
    elif "age distribution" in query or "histogram of passenger ages" in query:
        return "Age distribution histogram will be shown in Streamlit."
    elif "class distribution" in query:
        return df['class'].value_counts().to_dict()
    elif "percentage of passengers were male" in query or "gender distribution" in query:
        return df['sex'].value_counts(normalize=True).mul(100).to_dict()
    elif "average ticket fare" in query:
        return f"The average ticket fare was ${df['fare'].mean():.2f}."
    elif "how many passengers embarked from each port" in query:
        return df['embark_town'].value_counts().to_dict()
    else:
        return "Sorry, I didn't understand the question."

@app.get("/query")
def query_titanic(question: str):
    response = analyze_query(question, df)
    return {"response": response}
 
