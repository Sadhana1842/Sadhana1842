import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import nltk

# Download NLTK sentence tokenizer data
nltk.download('punkt')

def load_model_and_tokenizer():
    # Load the model configuration
    model_config = 'distilbert-base-uncased'
    model = DistilBertForSequenceClassification.from_pretrained(model_config, num_labels=3)

    # Load the tokenizer separately
    tokenizer = DistilBertTokenizer.from_pretrained(model_config)

    return model, tokenizer

def analyze_sentiment(statement, model, tokenizer):
    # Tokenize the statement into sentences
    sentences = nltk.sent_tokenize(statement)

    # Analyze the sentiment of each sentence
    sentence_sentiments = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentence_sentiments.append(sentiment_mapping[predictions.item()])

    # Determine the overall sentiment of the entire statement
    overall_sentiment = determine_overall_sentiment(sentence_sentiments)

    return overall_sentiment

def determine_overall_sentiment(sentence_sentiments):
    # Logic to determine overall sentiment based on individual sentence sentiments
    # You can customize this logic based on your requirements
    # For example, you might consider the majority sentiment or a weighted average

    # For simplicity, here we consider the majority sentiment
    sentiment_counts = {
        "Negative": sentence_sentiments.count("Negative"),
        "Neutral": sentence_sentiments.count("Neutral"),
        "Positive": sentence_sentiments.count("Positive")
    }

    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return overall_sentiment

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

        # Load the model and tokenizer
        model, tokenizer = load_model_and_tokenizer()

        if submit_button:
            if statement:
                overall_sentiment = analyze_sentiment(statement, model, tokenizer)
                st.write(f"Overall Sentiment: {overall_sentiment}")
            else:
                st.warning("Please enter a statement.")

if __name__ == "__main__":
    main()
