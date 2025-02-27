import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI

def load_data():
    """Load Titanic dataset from seaborn."""
    return sns.load_dataset("titanic")

def analyze_query(query, df):
    """Analyze user query and return insights."""
    query = query.lower()
    if "survival rate" in query:
        return survival_rate(df)
    elif "age distribution" in query or "histogram of passenger ages" in query:
        return age_distribution(df)
    elif "class distribution" in query:
        return class_distribution(df)
    elif "percentage of passengers were male" in query or "gender distribution" in query:
        return gender_distribution(df)
    elif "average ticket fare" in query:
        return average_fare(df)
    elif "how many passengers embarked from each port" in query:
        return embarkation_distribution(df)
    else:
        return "I'm sorry, I didn't understand the question. Try asking about survival rate, age distribution, class distribution, gender distribution, average fare, or embarkation stats."

def survival_rate(df):
    """Calculate and plot survival rate."""
    rates = df['survived'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    sns.barplot(x=rates.index, y=rates.values, ax=ax)
    ax.set_xticklabels(['Did Not Survive', 'Survived'])
    ax.set_ylabel('Percentage')
    ax.set_title('Survival Rate')
    st.pyplot(fig)
    return rates.to_dict()

def age_distribution(df):
    """Plot age distribution."""
    fig, ax = plt.subplots()
    sns.histplot(df['age'].dropna(), bins=20, kde=True, ax=ax)
    ax.set_title('Age Distribution')
    st.pyplot(fig)
    return "Displayed age distribution graph."

def class_distribution(df):
    """Plot class distribution."""
    fig, ax = plt.subplots()
    sns.countplot(x=df['class'], order=['First', 'Second', 'Third'], ax=ax)
    ax.set_title('Passenger Class Distribution')
    st.pyplot(fig)
    return df['class'].value_counts().to_dict()

def gender_distribution(df):
    """Calculate and plot gender distribution."""
    gender_counts = df['sex'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax)
    ax.set_ylabel('Percentage')
    ax.set_title('Gender Distribution')
    st.pyplot(fig)
    return gender_counts.to_dict()

def average_fare(df):
    """Calculate average ticket fare."""
    avg_fare = df['fare'].mean()
    return f"The average ticket fare was ${avg_fare:.2f}."

def embarkation_distribution(df):
    """Plot number of passengers per embarkation port."""
    fig, ax = plt.subplots()
    sns.countplot(x=df['embark_town'].dropna(), ax=ax)
    ax.set_title('Embarkation Distribution')
    st.pyplot(fig)
    return df['embark_town'].value_counts().to_dict()

# FastAPI setup
app = FastAPI()

df = load_data()

@app.get("/query")
def query_titanic(question: str):
    response = analyze_query(question, df)
    return {"response": response}

# Streamlit UI
def main():
    st.title("Titanic Chatbot")
    user_query = st.text_input("Ask a question about Titanic passengers:")
    if user_query:
        response = analyze_query(user_query, df)
        st.write(response)

if __name__ == "__main__":
    main()
