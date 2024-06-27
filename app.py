import os
import fitz  # PyMuPDF
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.generativeai import GenerativeModel
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Configure Google API
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize Gemini Pro model
model = GenerativeModel(model_name='gemini-pro')

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(pdf_file)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    pdf_document.close()
    return text

# Function to extract key components from policy text
def extract_components(text):
    components = {
        "Premium": r"(?:premium|premiums|amount|amt|price)[:\s]*(?:rs|rps|rupees|inr)?[\s]*([0-9,.]+)",
        "Policy Period": r"(?:policy period)[:\s]*([\d/]+)[\s]*to[\s]*([\d/]+)",
        "Coverage": r"(?:coverage)[:\s]*(?:upto|up to|of)?[\s]*([0-9,.]+)",
        "Tenure": r"(?:tenure|duration)[:\s]*([\d.]+)\s*(?:years|months|days)?",
        "Insured Amount": r"(?:insured amount)[:\s]*([0-9,.]+)"
    }
    extracted_data = {}

    
    for component, pattern in components.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(",", "")
            if value.strip() == '.' or value.strip() == '':
                extracted_data[component] = '0'
            else:
                extracted_data[component] = value
        else:
            extracted_data[component] = '0'  # Default value if extraction fails
    print(extracted_data)
    return extracted_data

# Function to classify policy based on components
def classify_policy(components):
    premium = float(components.get("Premium", 0))
    coverage = float(components.get("Coverage", 0))
    if premium < 500 and coverage < 5000:
        return 'Basic'
    elif 5000 < premium < 12000 and coverage < 10000:
        return 'Standard'
    elif premium > 12000 or coverage > 10000:
        return 'Premium'
    else:
        return 'Addons'


# Folder containing PDF policy files
policy_folder = "Fw_ Policy Copies"

# Initialize lists to store data
premiums = []
coverages = []
tenures = []
insured_amounts = []
policies = []
categories = []

# Iterate over PDF files in the folder
for file_name in os.listdir(policy_folder):
    if file_name.lower().endswith(".pdf"):
        file_path = os.path.join(policy_folder, file_name)
        policy_text = extract_text_from_pdf(file_path)
        components = extract_components(policy_text)
        
        # Extracted components
        premiums.append(float(components.get("Premium", 0)))
        coverages.append(float(components.get("Coverage", 0)))
        tenures.append(float(components.get("Tenure", 0)))
        insured_amounts.append(float(components.get("Insured Amount", 0)))
        policies.append(file_name)
        categories.append(classify_policy(components))

# Create a DataFrame
data = pd.DataFrame({
    "Policy": policies,
    "Premium": premiums,
    "Coverage": coverages,
    "Tenure": tenures,
    "Insured Amount": insured_amounts,
    "Category": categories
})

# Display file names with corresponding categories
print(data[['Policy', 'Category']])

# # Plotting the graphs
# sns.histplot(data['Premium'], kde=True)
# plt.title('Distribution of Premiums')
# plt.show()

# sns.histplot(data['Coverage'], kde=True)
# plt.title('Distribution of Coverage')
# plt.show()
