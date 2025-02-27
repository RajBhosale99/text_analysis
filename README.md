## Setup and Installation
**Prerequisites:**

    - Python 3.x installed on your system.
    
    - pip package installer.
    
**Then Clone the Repository**

**Install Dependencies:**
It is recommended to create a virtual environment.
```
python -m venv venv
source venv/bin/activate  # On Linux/macOS

venv\Scripts\activate  # On Windows

pip install -r requirements.txt

Then run pip install -r requirements.txt.
```


#**Set up IBM Watson NLU API Credentials:**

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

streamlit run app.py




**Usage**
Data Input:

Open the application in your browser.
Navigate to the "Data Input" section using the sidebar.
Click on "Browse files" to upload a CSV file from your local machine. Ensure your CSV file has a column named 'text' containing the text data you want to analyze.
Once uploaded successfully, the application will automatically begin processing your data. Status updates will be displayed to track the progress of each processing step.
Explore Visualizations:

After processing is complete, navigate to the "Visualizations" section using the sidebar.
Choose a visualization category from the radio buttons: "Text Overview", "Sentiment Analysis", "Emotion Analysis", or "Topic Modeling".
Explore the interactive visualizations generated for the selected category. Information messages are provided above each visualization to explain its purpose and interpretation.
Download Processed Data:

To download the processed and analyzed data, navigate to the "Download Data" section in the sidebar.
Click the "Download Processed CSV" button. A CSV file named enhanced_processed_data.csv will be downloaded to your browser's default download location. This file contains your original data augmented with the results of all NLP processing steps.




**Example Workflow**
Prepare your CSV data: Ensure your data is in CSV format and includes a column named 'text' with the text entries you want to analyze.
Upload Data: In the "Data Input" section of the dashboard, upload your CSV file.
Wait for Processing: Monitor the status updates in the application as your data is cleaned, analyzed, and processed.
Visualize Results: Once processing is complete, go to the "Visualizations" section and explore the different categories to understand your text data through various charts and word clouds.
Download Enhanced Data: If you need the processed data for further analysis or record-keeping, download it from the "Download Data" section.


**Deployed Project:**
https://textanalysis-aas4wymhmc2y5iqjyz8yfw.streamlit.app/
