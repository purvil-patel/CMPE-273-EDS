**Deloitte Audit Assistance Tool: Tax Question Checker**

**Introduction**

This project was developed as a part of the Deloitte Auditor team's initiative to leverage GPT-based models for assisting auditors in understanding US Tax law. The web interface facilitates the querying of tax-related questions to a GPT model and showcases the responses, helping auditors make informed decisions about tax deductions.

**Architecture**

The application is built using Flask, a micro web framework in Python. The backend utilizes the transformers library to employ a pre-trained BERT model for initial classification of the user's query to determine if it is tax-related. If the question is deemed tax-related, the query is further sent to OpenAI's GPT model for a detailed response.

**Key Components:**

BERT Classifier: Classifies if the question is tax-related.

OpenAI GPT Model: Provides detailed responses to tax-related questions.

Flask Web Interface: Allows users to input their questions and displays responses.

Rate Limiter: Ensures the system isn't overwhelmed with too many requests.

**Demo**

Below is a working demo video showcasing the functionality of the Tax Question Checker:

https://github.com/purvil-patel/CMPE-273-EDS/assets/67355397/7df8cb82-a945-4717-b97f-bd81d93cee8e

