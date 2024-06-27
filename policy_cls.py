import os
import fitz  # PyMuPDF
import pandas as pd
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Safety settings for GenAI model
safety_settings = [
    {"category": category, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} 
    for category in [
        "HARM_CATEGORY_HARASSMENT", 
        "HARM_CATEGORY_HATE_SPEECH", 
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
        "HARM_CATEGORY_DANGEROUS_CONTENT"
    ]
]

# Function to extract text from a single PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(pdf_file)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    pdf_document.close()
    return text

# Function to get text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to extract policy information
def extract_policy_information(chunks, model):
    prompt_template = {
        "Policy Gross Premium": "Extract policy gross premium or premium amount from the uploaded policy document.",
        "Policy Period (Start)": "Extract the policy start date from the uploaded policy document.",
        "Policy Period (End)": "Extract the policy end date from the uploaded policy document.",
        "Sum Insured": "Extract the sum insured from the uploaded policy document.",
        "Key Coverages & Benefits": "Extract key coverages and benefits including the limits from the uploaded policy document."
    }
    extracted_info = {}
    for key, prompt in prompt_template.items():
        try:
            response = model.generate_content(prompt + " " + ' '.join(chunks))
            extracted_info[key] = response.text if response.text else ""
            print(f"Extracted {key}: {extracted_info[key]}")  # Debug statement
        except Exception as e:
            print(f"Error during extraction for {key}: {e}")
            extracted_info[key] = ""
    return extracted_info

# Function to parse custom date formats
def custom_date_parser(date_str):
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        try:
            return datetime.strptime(date_str, "Midnight %d-%b-%Y")
        except ValueError:
            return None


# Function to normalize coverage data
def normalize_coverages(coverage_str):
    coverages = {}
    if not isinstance(coverage_str, str):
        return coverages
    
    # Define patterns for key coverages
    coverage_patterns = {
        "Hospitalization Expenses": r"Hospitalization Expenses:\s*(.*)",
        "Pre-Hospitalization and Post-Hospitalization Expenses": r"Pre-Hospitalization and Post-Hospitalization Expenses:\s*(.*)",
        "Organ Donor Expenses": r"Organ Donor Expenses:\s*(.*)",
        "Preventive Health Check-up": r"Preventive Health Check-up:\s*(.*)",
        "Emergency Air Ambulance": r"Emergency Air Ambulance:\s*(.*)",
        "Daily Cash for Shared Accommodation": r"Daily Cash for Shared Accommodation:\s*(.*)",
        "Protect Benefit": r"Protect Benefit:\s*(.*)",
        "Plus Benefit": r"Plus Benefit:\s*(.*)",
        "Secure Benefit": r"Secure Benefit:\s*(.*)",
        "Automatic Restore Benefit": r"Automatic Restore Benefit:\s*(.*)",
        "Aggregate Deductible": r"Aggregate Deductible:\s*(.*)",
        "E-Opinion for Critical Illness": r"E-Opinion for Critical Illness:\s*(.*)",
        "Home Healthcare": r"Home Healthcare:\s*(.*)",
        "Domiciliary Hospitalization": r"Domiciliary Hospitalization:\s*(.*)",
        "Ayush Treatment": r"Ayush Treatment:\s*(.*)",
        "Road Ambulance": r"Road Ambulance:\s*(.*)",
        "Dental Treatment": r"Dental Treatment:\s*(.*)",
        "Plastic Surgery": r"Plastic Surgery:\s*(.*)",
        "Day Care Treatment": r"Day Care Treatment:\s*(.*)"
    }
    for key, pattern in coverage_patterns.items():
        match = re.search(pattern, coverage_str)
        if match:
            coverages[key] = match.group(1).strip()
        else:
            coverages[key] = "Not Covered"
    return coverages

# Function to classify policy documents based on coverages and other criteria
def classify_documents(policy_data):
    classifications = []
    for policy in policy_data:
        classification = {"Filename": policy["Filename"]}

        # Extract and normalize coverage details
        coverages = normalize_coverages(policy.get("Key Coverages & Benefits", ""))
        
        premium_str = policy.get("Policy Gross Premium", "")
        print(f"Processing premium: {premium_str}")  # Debug statement
        if premium_str:
            # Remove non-numeric characters from premium string
            premium_str = ''.join(c for c in premium_str if c.isdigit() or c == '.')
            premium = float(premium_str) if premium_str.replace('.', '', 1).isdigit() else 0.0
        else:
            premium = 0.0

        sum_insured_str = policy.get("Sum Insured", "")
        print(f"Processing sum insured: {sum_insured_str}")  # Debug statement
        if sum_insured_str :
            # Remove non-numeric characters from sum insured string
            sum_insured_str = ''.join(c for c in sum_insured_str if c.isdigit() or c == '.')
            sum_insured = float(sum_insured_str) if sum_insured_str.replace('.', '', 1).isdigit() else 0.0
        else:
            sum_insured = 0.0

        policy_period_start_str = policy.get("Policy Period (Start)", "")
        policy_period_start = custom_date_parser(policy_period_start_str)

        policy_period_end_str = policy.get("Policy Period (End)", "")
        policy_period_end = custom_date_parser(policy_period_end_str)

        if policy_period_start and policy_period_end:
            tenure = (policy_period_end - policy_period_start).days / 365
        else:
            tenure = 1  # Handle case where policy start or end date is not found

        print(f"Classifying document: {policy['Filename']} with premium: {premium}, sum insured: {sum_insured}, tenure: {tenure}")  # Debug statement

        # Define rating criteria
        coverage_score = sum(1 for coverage in coverages.values() if coverage != "Not Covered")
        print(f"Coverage score: {coverage_score}")  # Debug statement

        # Classification conditions
        if coverage_score >= 15 and premium >= 5000 and sum_insured >= 500000 and tenure >= 1:
            classification["Rating"] = "5 Star"
        elif 10 <= coverage_score < 15 and 1001 > premium <= 2000 and sum_insured >= 400000 and tenure >= 1:
            classification["Rating"] = "4 Star"
        elif 5 <= coverage_score < 10 and premium <= 1000 and sum_insured >= 200000 and tenure >= 1:
            classification["Rating"] = "3 Star"
        else:
            classification["Rating"] = "Other"

        classifications.append(classification)
    return classifications

# Main function to process PDF files in a folder
def process_pdf_folder(folder_path):
    model = genai.GenerativeModel(model_name='gemini-pro', safety_settings=safety_settings)
    policy_data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            chunks = get_text_chunks(text)
            policy_info = extract_policy_information(chunks, model)
            policy_info["Filename"] = filename
            policy_data.append(policy_info)

    # Save the extracted information to a CSV file
    df = pd.DataFrame(policy_data)
    df.to_csv("extracted_policy_information.csv", index=False)
    print("Extracted policy information saved to CSV.")  # Debug statement

    # Analyze and classify the policy documents
    classifications = classify_documents(policy_data)
    classification_df = pd.DataFrame(classifications)
    classification_df.to_csv("classified_policy_information.csv", index=False)
    print("Classified policy information saved to CSV.")  # Debug statement

    return classification_df

# Streamlit app to upload PDF folder and process the files
def main():
    st.title("Policy Document Analysis")

    uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        folder_path = "uploaded_pdfs"
        if os.path.exists(folder_path):
            # Clear the folder before saving new files
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
        else:
            os.makedirs(folder_path, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(folder_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        classification_df = process_pdf_folder(folder_path)
        st.write("Processing complete. Check the extracted_policy_information.csv and classified_policy_information.csv files.")

        # Display the ratings in the UI
        st.write("Policy Ratings:")
        st.dataframe(classification_df[['Filename', 'Rating']])

if __name__ == "__main__":
    main()
