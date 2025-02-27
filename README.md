# IMPROVING TEXT ANALYTICS DATA QUALITY WITH ADVANCED NLP

## Overview

This Streamlit application provides a user-friendly dashboard designed to enhance the quality of text analytics data through advanced Natural Language Processing (NLP) techniques. It empowers users to upload CSV files containing text data, apply a suite of sophisticated NLP processes, and visually explore the enriched data.  The dashboard is built using Streamlit for an interactive interface, pandas for efficient data manipulation, NLTK and transformers for core text processing, IBM Watson Natural Language Understanding for deep analytical insights, and matplotlib/seaborn for compelling visualizations.

## Features

- **Data Input:**
    - Accepts CSV file uploads, requiring a 'text' column to specify the text data for analysis.
    - Initiates automatic data loading and processing immediately after successful file upload.
    - Displays a sample of the original data for quick validation and preview.

- **Advanced Text Processing for Data Quality Improvement:**
    - **Text Cleaning:**  Standardizes text by lowercasing, removing HTML artifacts, eliminating non-alphabetic characters, and normalizing whitespace, thus improving data consistency.
    - **Grammar Correction:** Employs the `prithivida/grammar_error_correcter_v1` transformer model to rectify grammatical inaccuracies, enhancing text clarity and readability for downstream analytics.
    - **Part-of-Speech (POS) Tagging:**  Utilizes NLTK to assign POS tags to words, enabling nuanced linguistic analysis and feature extraction for improved text understanding.
    - **Topic Modeling (LDA):**  Applies Latent Dirichlet Allocation to discover latent topics within the text corpus, structuring unstructured text and revealing thematic patterns.

- **Deep Textual Insights with IBM Watson NLU:**
    - **Sentiment Analysis:**  Leverages IBM Watson NLU to perform in-depth sentiment analysis, providing a granular sentiment score and categorical classification (Positive, Negative, Neutral, Unknown) for each text entry.
    - **Emotion Analysis:**  Detects a spectrum of emotions—sadness, joy, fear, disgust, and anger—within the text using IBM Watson NLU, quantifying emotional content for richer insights.
    - **Keyword and Entity Extraction:**  *(Functionality inherent in IBM Watson NLU analysis, although current visualizations focus on sentiment, emotion, and topics. Expandable for future feature enhancements.)*

- **Interactive Data Visualizations for Quality Assessment:**
    - **Text Overview Visualizations:**
        - **Text Length Distribution:** Histograms visualize the distribution of text lengths pre- and post-cleaning, demonstrating the impact of cleaning processes on data structure.
        - **Word Clouds:**  Visually represent the most frequent terms in both original and cleaned text, highlighting key vocabulary and the effect of text preprocessing.
    - **Sentiment Analysis Visualizations:**
        - **Sentiment Score Distribution:**  Histograms display the distribution of IBM NLU sentiment scores, allowing assessment of overall sentiment polarity and intensity within the dataset.
        - **Sentiment Category Counts:** Bar charts present counts of text entries per sentiment category, providing a clear overview of sentiment distribution.
        - **Sentiment vs. Text Length:** Scatter plots explore correlations between cleaned text length and sentiment scores, colored by sentiment category, to identify potential relationships.
    - **Emotion Analysis Visualizations:**
        - **Average Emotion Scores:** Bar charts showcase average emotion scores across emotions detected by IBM NLU, revealing dominant emotional tones in the text data.
    - **Topic Modeling Visualizations:**
        - **Topic Word Cloud:**  Word clouds for each LDA-derived topic (currently demonstrating Topic 0) visualize the most salient terms per topic, aiding in topic interpretation and validation.

- **Processed Data Download:**
    - Enables download of the fully processed and analyzed dataset as a CSV file. The downloaded data includes original text alongside cleaned text, grammar-corrected text, NLU analysis results, sentiment classifications, emotion scores, POS tags, and topic modeling outputs – facilitating further in-depth analysis and external application.

## Setup and Installation

1. **Prerequisites:**
    - Python 3.x installed on your system.
    - pip package installer.

2. **Clone the Repository:**
 
**Install Dependencies:**
It is recommended to create a virtual environment.
```
python -m venv venv
source venv/bin/activate  # On Linux/macOS

venv\Scripts\activate  # On Windows

pip install -r requirements.txt

Then run pip install -r requirements.txt.
```


## Set up IBM Watson NLU API Credentials:

IBM Cloud Account: Sign up for an IBM Cloud account if you don't have one.
Natural Language Understanding Service: Create a Natural Language Understanding service instance in your IBM Cloud account.
API Key and URL: Generate an API key and service URL for your NLU service instance.

**Configure Environment Variables:**

Create a .env file in the root directory of your project.
Add your IBM Watson NLU API key and URL to the .env file:
```
API_KEY="YOUR_IBM_NLU_API_KEY"
API_URL="YOUR_IBM_NLU_URL"
```


**Run the Streamlit Application:**
Navigate to your project directory in the terminal and run:
```
streamlit run app.py
```
**Usage**
Data Input:  
Open the application in your browser.  
Navigate to the "Data Input" section using the sidebar.  
Click on "Browse files" to upload a CSV file from your local machine.  
Ensure your CSV file has a column named 'text' containing the text data you want to analyze.    
Once uploaded successfully, the application will automatically begin processing your data.  
Status updates will be displayed to track the progress of each processing step.  

Explore Visualizations:  
After processing is complete, navigate to the "Visualizations" section using the sidebar.
Choose a visualization category from the radio buttons: "Text Overview", "Sentiment Analysis", "Emotion Analysis", or "Topic Modeling".
Explore the interactive visualizations generated for the selected category. Information messages are provided above each visualization to explain its purpose and interpretation.
Download Processed Data:

To download the processed and analyzed data, navigate to the "Download Data" section in the sidebar.  
Click the "Download Processed CSV" button.  
A CSV file named enhanced_processed_data.csv will be downloaded to your browser's default download location. This file contains your original data augmented with the results of all NLP processing steps.  

**Example Workflow**
Prepare your CSV data: Ensure your data is in CSV format and includes a column named 'text' with the text entries you want to analyze.

Upload Data: In the "Data Input" section of the dashboard, upload your CSV file.

Wait for Processing: Monitor the status updates in the application as your data is cleaned, analyzed, and processed.

Visualize Results: Once processing is complete, go to the "Visualizations" section and explore the different categories to understand your text data through various charts and word clouds.

Download Enhanced Data: If you need the processed data for further analysis or record-keeping, download it from the "Download Data" section.


**Deployed Project:**  
https://textanalysis-aas4wymhmc2y5iqjyz8yfw.streamlit.app/
