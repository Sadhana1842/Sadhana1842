import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained model
model = BertForSequenceClassification.from_pretrained('ahmedrachid/FinancialBERT-Sentiment-Analysis', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('ahmedrachid/FinancialBERT-Sentiment-Analysis')

def analyze_sentiment(statement):
    inputs = tokenizer(statement, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment_result = sentiment_mapping[predictions.item()]
    return sentiment_result

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Home", "Sentiment Analysis"])

    if page == "Home":
        st.title("Welcome to the Sentiment Analysis App")
        st.write("Select a page from the sidebar.")

    elif page == "Sentiment Analysis":
        st.title("Sentiment Analysis")
        st.write("Enter a statement to analyze its sentiment.")

        statement = st.text_area("Enter your statement:")
        submit_button = st.button("Submit")

        if submit_button:
            if statement:
                sentiment_result = analyze_sentiment(statement)
                st.write(f"Sentiment: {sentiment_result}")
            else:
                st.warning("Please enter a statement.")

if __name__ == "__main__":
    main()
