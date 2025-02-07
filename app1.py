import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd

# Load the Excel sheet
excel_path = "fomc_final_sheets.xlsx"
excel_data = pd.read_excel(excel_path)

def load_model_and_tokenizer():
    # Load the model configuration
    model_config = 'distilbert-base-uncased'
    model = DistilBertForSequenceClassification.from_pretrained(model_config, num_labels=3)

    # Load the tokenizer separately
    tokenizer = DistilBertTokenizer.from_pretrained(model_config)

    return model, tokenizer

def determine_overall_sentiment(sentence_sentiments):
    # Calculate the percentage of each sentiment
    total_sentences = len(sentence_sentiments)
    sentiment_counts = {
        "Negative": sentence_sentiments.count("Negative") / total_sentences * 100,
        "Neutral": sentence_sentiments.count("Neutral") / total_sentences * 100,
        "Positive": sentence_sentiments.count("Positive") / total_sentences * 100
    }

    # Determine the overall sentiment based on the highest percentage
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return sentiment_counts["Negative"], sentiment_counts["Neutral"], sentiment_counts["Positive"], overall_sentiment

def analyze_sentiment(statement, model, tokenizer):
    # Split the statement into sentences using '.' as the delimiter
    sentences = [s.strip() for s in statement.split('.') if s.strip()]  # Avoid empty sentences

    # Analyze the sentiment of each sentence
    sentence_sentiments = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentence_sentiments.append(sentiment_mapping[predictions.item()])

    # Determine the overall sentiment and percentages
    negative_percentage, neutral_percentage, positive_percentage, overall_sentiment = determine_overall_sentiment(sentence_sentiments)

    return negative_percentage, neutral_percentage, positive_percentage, overall_sentiment

def main():
    st.sidebar.write("Select a page")
    page = st.sidebar.radio("Pages:", ["Home", "FOMC statement Tone estimator", "FOMC statement sentiments (Years 2006 to 2023)"])
    st.sidebar.write("Dr. Narendra Regmi")
    st.sidebar.write("Assistant Professor")
    st.sidebar.write("Macroeconomics, International trade, Economic growth")
    st.sidebar.write("Wisconsin University")

    if page == "Home":
        st.title("FOMC Statement Tone Estimator")
        st.write("Select a page from the sidebar.")

    elif page == "FOMC statement Tone estimator":
        st.title("FOMC statement tone estimator")
        st.write("Enter a statement to analyze its sentiment.")

        statement = st.text_area("Enter your statement:")
        submit_button = st.button("Submit")

        # Load the model and tokenizer
        model, tokenizer = load_model_and_tokenizer()

        if submit_button:
            if statement:
                negative_percentage, neutral_percentage, positive_percentage, overall_sentiment = analyze_sentiment(statement, model, tokenizer)
                st.write(f"Negative Percentage: {negative_percentage:.2f}%")
                st.write(f"Neutral Percentage: {neutral_percentage:.2f}%")
                st.write(f"Positive Percentage: {positive_percentage:.2f}%")
                st.write(f"Overall Sentiment: {overall_sentiment}")
            else:
                st.warning("Please enter a statement.")

    elif page == "FOMC statement sentiments (Years 2006 to 2023)":
        st.title("FOMC statement sentiments (Years 2006 to 2023)")
        st.write("Select a year and date to display sentiment analysis")

        # Dropdown for selecting the year
        selected_year = st.selectbox("Select Year", excel_data["Year"].unique())

        # Dropdown for selecting the date based on the selected year
        selected_dates = excel_data[excel_data["Year"] == selected_year]["Date"].unique()
        selected_date = st.selectbox("Select Date", selected_dates)

        submit_excel_button = st.button("Submit")

        if submit_excel_button:
            selected_row = excel_data[(excel_data["Year"] == selected_year) & (excel_data["Date"] == selected_date)].iloc[0]
            statement = selected_row["Statement"]
            negative_percentage = selected_row["Mean Negative"]
            positive_percentage = selected_row["Mean Positive"]
            neutral_percentage = selected_row["Mean Neutral"]
            overall_sentiment = selected_row["Tone"]

            st.write(f"Statement: {statement}")
            st.write(f"Negative Percentage: {negative_percentage:.2f}%")
            st.write(f"Positive Percentage: {positive_percentage:.2f}%")
            st.write(f"Neutral Percentage: {neutral_percentage:.2f}%")
            st.write(f"Overall Sentiment: {overall_sentiment}")

if __name__ == "__main__":
    main()
