import streamlit as st
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
from tqdm import tqdm
from collections import Counter
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from gensim import corpora
from gensim.models import LdaModel
import os

api_key = os.getenv('API_KEY')
url = os.getenv('API_URL')
if api_key and url:
    st.success("API key and URL loaded successfully.")
else:
    st.error("API key or URL not found. Please check your environment variables.")

# ------------------------------
# Download NLTK Resources
# ------------------------------
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# ------------------------------
# IBM Watson NLU Setup
# ------------------------------
# Replace these with your actual API key and URL

authenticator = IAMAuthenticator(api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',  # Use an appropriate version date
    authenticator=authenticator
)
nlu.set_service_url(url)

def analyze_with_nlu(text):
    """
    Analyze text using IBM Watson NLU, including sentiment, keywords,
    entities, and emotion analysis.
    """
    response = nlu.analyze(
        text=text,
        features={
            'sentiment': {},
            'keywords': {},
            'entities': {},
            'emotion': {}
        }
    ).get_result()
    return response

# ------------------------------
# Processing Functions
# ------------------------------
def clean_text(text):
    """
    Lowercase the text, remove HTML tags, non-alphabetic characters,
    and extra spaces.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load grammar correction model and tokenizer
tokenizer_gc = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
model_gc = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

def correct_grammar(text):
    """
    Correct grammar mistakes using a seq2seq model.
    """
    inputs = tokenizer_gc(text, return_tensors="pt", padding=True)
    outputs = model_gc.generate(**inputs, max_length=256)
    corrected = tokenizer_gc.decode(outputs[0], skip_special_tokens=True)
    return corrected

def get_pos_tags(text):
    """
    Tokenize text and get Part-of-Speech tags.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags

def get_nlu_sentiment_score(analysis_result):
    """
    Extract the sentiment score from the IBM NLU analysis result.
    Returns None if sentiment information is missing.
    """
    try:
        return analysis_result['sentiment']['document']['score']
    except (KeyError, TypeError):
        return None

def categorize_nlu_sentiment(score):
    if score is None:
        return 'Unknown'
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_emotion_scores(analysis_result):
    """
    Extracts emotion scores from the IBM NLU analysis result.
    Returns a dictionary of emotions or an empty dict if not available.
    """
    try:
        return analysis_result['emotion']['document']['emotion']
    except (KeyError, TypeError):
        return {}

# Streamlit Dashboard Layout - IMPROVED
# ------------------------------
st.title("NLP Processing & Visualization Dashboard")
# Sidebar Styling - Added background color
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #e0e0e0; /* Lighter grey - more noticeable than before */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    st.header("Navigation")
    analysis_section = st.radio("Go to",
                                 ['Data Input', 'Visualizations', 'Download Data']) #, 'Processing Options' - if we add options later

# ------------------------------
# Data Input Section
# ------------------------------
if analysis_section == 'Data Input':
    st.header("Data Input")
    st.info(
        "Please upload a CSV file that includes a header with a 'text' column. "
        "Each row should contain a text entry that you wish to process."
    )
    uploaded_file = st.file_uploader("Upload CSV file with a 'text' column", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if 'text' not in data.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                st.success("Data loaded successfully! Processing will start automatically.") # Updated success message
                st.subheader("Original Data Sample")
                st.write(data.head(3))

                # --- Step 1: Text Cleaning --- # Processing starts immediately
                with st.status("Data Cleaning in progress...") as status:
                    data['cleaned_text'] = data['text'].apply(clean_text)
                    status.update(label="Grammar Correction in progress...")
                    data['corrected_text'] = data['cleaned_text'].progress_apply(correct_grammar)
                    status.update(label="IBM Watson NLU Analysis in progress...")
                    data['nlu_analysis'] = data['corrected_text'].progress_apply(analyze_with_nlu)
                    status.update(label="POS Tagging in progress...")
                    data['pos_tags'] = data['corrected_text'].progress_apply(get_pos_tags)
                    status.update(label="Topic Modeling in progress...")
                    data['tokens'] = data['corrected_text'].apply(lambda x: nltk.word_tokenize(x))
                    dictionary = corpora.Dictionary(data['tokens'])
                    corpus = [dictionary.doc2bow(text) for text in data['tokens']]
                    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
                    data['nlu_sentiment'] = data['nlu_analysis'].apply(get_nlu_sentiment_score)
                    data['nlu_sentiment_category'] = data['nlu_sentiment'].apply(categorize_nlu_sentiment)
                    data['emotion_scores'] = data['nlu_analysis'].apply(get_emotion_scores)
                    for emotion in ['sadness', 'joy', 'fear', 'disgust', 'anger']:
                        data[emotion] = data['emotion_scores'].apply(lambda x: x.get(emotion, None))
                    st.session_state['processed_data'] = data # Store processed data in session state
                    st.session_state['lda_model'] = lda_model # Store LDA model
                    st.success("Processing complete! Go to Visualizations.")


        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Awaiting CSV file upload.")


# ------------------------------
# Visualizations Section
# ------------------------------
elif analysis_section == 'Visualizations':
    st.header("Visualizations")
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        lda_model = st.session_state['lda_model']

        # --- Tabbed Visualizations ---
        visualization_tab = st.radio("Select Visualization Category",
                                    ['Text Overview', 'Sentiment Analysis', 'Emotion Analysis', 'Topic Modeling'])

        if visualization_tab == 'Text Overview':
            st.subheader("Text Overview Visualizations")

            # 1. Text Length Distributions
            st.subheader("Text Length Distributions")
            st.info("Distribution of text lengths before and after cleaning. Helps understand the impact of the cleaning process.")
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            sns.histplot(data['text'].str.len(), kde=True, bins=30, color='skyblue', ax=ax[0])
            ax[0].set_title('Original Text Length Distribution')
            ax[0].set_xlabel('Text Length')
            ax[0].set_ylabel('Frequency')
            sns.histplot(data['cleaned_text'].str.len(), kde=True, bins=30, color='orange', ax=ax[1])
            ax[1].set_title('Cleaned Text Length Distribution')
            ax[1].set_xlabel('Text Length')
            ax[1].set_ylabel('Frequency')
            st.pyplot(fig)

            # 2. Word Clouds
            st.subheader("Word Clouds")
            st.info("Word clouds showing the most frequent words in the original and cleaned text.")
            col1, col2 = st.columns(2) # Display wordclouds side-by-side
            with col1:
                st.subheader("Original Text")
                original_text_combined = ' '.join(data['text'])
                wordcloud_original = WordCloud(width=400, height=200, background_color='white').generate(original_text_combined) # Smaller wordclouds
                fig_wc1, ax_wc1 = plt.subplots(figsize=(5, 3)) # Smaller figures
                ax_wc1.imshow(wordcloud_original, interpolation='bilinear')
                ax_wc1.axis('off')
                st.pyplot(fig_wc1)
            with col2:
                st.subheader("Cleaned Text")
                cleaned_text_combined = ' '.join(data['cleaned_text'])
                wordcloud_cleaned = WordCloud(width=400, height=200, background_color='white').generate(cleaned_text_combined) # Smaller wordclouds
                fig_wc2, ax_wc2 = plt.subplots(figsize=(5, 3)) # Smaller figures
                ax_wc2.imshow(wordcloud_cleaned, interpolation='bilinear')
                ax_wc2.axis('off')
                st.pyplot(fig_wc2)


        elif visualization_tab == 'Sentiment Analysis':
            st.subheader("Sentiment Analysis Visualizations")

            # 4. Sentiment Score Distribution
            st.subheader("IBM NLU Sentiment Score Distribution")
            st.info("Distribution of sentiment scores from IBM NLU. Scores near 0 are neutral, positive scores indicate positive sentiment, and negative scores indicate negative sentiment.")
            fig_sent, ax_sent = plt.subplots(figsize=(8, 6))
            sns.histplot(data['nlu_sentiment'].dropna(), kde=True, bins=30, color='green', ax=ax_sent)
            ax_sent.axvline(x=0, color='black', linestyle='--', label='Neutral Threshold')
            ax_sent.set_title('IBM NLU Sentiment Score Distribution')
            ax_sent.set_xlabel('Sentiment Score')
            ax_sent.set_ylabel('Frequency')
            ax_sent.legend()
            st.pyplot(fig_sent)

            # 5. Sentiment Category Bar Chart
            st.subheader("Sentiment Category Counts")
            st.info("Counts of text entries categorized as Positive, Negative, Neutral, or Unknown sentiment based on IBM NLU scores.")
            fig_cat, ax_cat = plt.subplots(figsize=(8, 6))
            sns.countplot(x='nlu_sentiment_category', data=data,
                            palette={'Positive':'lightgreen', 'Neutral':'gray', 'Negative':'lightcoral', 'Unknown':'orange'},
                            ax=ax_cat)
            ax_cat.set_title('IBM NLU Sentiment Category Counts')
            ax_cat.set_xlabel('Sentiment Category')
            ax_cat.set_ylabel('Count')
            st.pyplot(fig_cat)

            # 6. Scatter Plot: Sentiment vs. Cleaned Text Length
            st.subheader("Sentiment vs. Cleaned Text Length")
            st.info("Scatter plot showing the relationship between cleaned text length and sentiment score. Color-coded by sentiment category.")
            data['cleaned_text_length'] = data['cleaned_text'].str.len()
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='cleaned_text_length', y='nlu_sentiment', data=data,
                            hue='nlu_sentiment_category',
                            palette={'Positive':'lightgreen', 'Neutral':'gray', 'Negative':'lightcoral', 'Unknown':'orange'},
                            ax=ax_scatter)
            ax_scatter.set_title('Sentiment Score vs. Cleaned Text Length')
            ax_scatter.set_xlabel('Cleaned Text Length')
            ax_scatter.set_ylabel('IBM NLU Sentiment Score')
            ax_scatter.legend(title='Sentiment Category')
            st.pyplot(fig_scatter)


        elif visualization_tab == 'Emotion Analysis':
            st.subheader("Emotion Analysis Visualizations")

            # 7. Average Emotion Scores
            st.subheader("Average Emotion Scores (IBM NLU)")
            st.info("Bar chart showing the average emotion scores (sadness, joy, fear, disgust, anger) derived from IBM NLU.")
            emotion_means = {
                'sadness': data['sadness'].mean(),
                'joy': data['joy'].mean(),
                'fear': data['fear'].mean(),
                'disgust': data['disgust'].mean(),
                'anger': data['anger'].mean()
            }
            fig_emotion, ax_emotion = plt.subplots(figsize=(8, 6))
            sns.barplot(x=list(emotion_means.keys()), y=list(emotion_means.values()), palette='viridis', ax=ax_emotion)
            ax_emotion.set_title('Average Emotion Scores (IBM NLU)')
            ax_emotion.set_xlabel('Emotion')
            ax_emotion.set_ylabel('Average Score')
            ax_emotion.set_ylim(0, 1)
            st.pyplot(fig_emotion)


        elif visualization_tab == 'Topic Modeling':
            st.subheader("Topic Modeling Visualizations")

            # 8. Word Cloud for Topic 0 (LDA) - Example, could be made selectable
            st.subheader("Word Cloud for LDA Topic 0") # Let's start with just topic 0 for now
            st.info("Word cloud for Topic 0 from LDA topic modeling, showing the most relevant words for this topic.")
            topic_id = 0  # Display topic 0 initially
            topic_words = dict(lda_model.show_topic(topic_id, topn=50))
            wordcloud_topic = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
            fig_topic, ax_topic = plt.subplots(figsize=(10, 5))
            ax_topic.imshow(wordcloud_topic, interpolation='bilinear')
            ax_topic.set_title(f'Word Cloud for LDA Topic {topic_id}')
            ax_topic.axis('off')
            st.pyplot(fig_topic)


    else:
        st.info("Please upload and process data in the 'Data Input' section first to see visualizations.")


# ------------------------------
# Download Data Section
# ------------------------------
elif analysis_section == 'Download Data':
    st.header("Download Processed Data")
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        csv_bytes = csv_data.encode('utf-8')

        st.download_button(
            label="Download Processed CSV",
            data=csv_bytes,
            file_name="enhanced_processed_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Process data in the 'Data Input' section first to enable download.")

# ------------------------------
# Footer - optional
# ------------------------------
st.markdown("---")
st.markdown("Dashboard created with Streamlit.")
