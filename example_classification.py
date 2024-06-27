import os
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from datetime import datetime
import base64
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

# Function to extract text from PDF bytes
def extract_text_from_bytes(pdf_bytes):
    text = ""
    pdf_document = fitz.Document(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    print(text)
    return text

# Function to get text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to extract policy information
def extract_policy_information(chunks, model):
    prompt_template = {
        "Policy Number": "Extract policy number from the uploaded policy document.",
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
            extracted_info[key] = response.text
            print(f"Extracted {key}: {extracted_info[key]}")  # Debug statement
        except Exception as e:
            print(f"Error during extraction for {key}: {e}")
            extracted_info[key] = None
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

# Main function to process PDF files uploaded via Streamlit
def process_uploaded_pdfs(uploaded_files):
    model = genai.GenerativeModel(model_name='gemini-pro', safety_settings=safety_settings)
    policy_data = []
    coverages_data = []
    
    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.read()
        text = extract_text_from_bytes(pdf_bytes)
        chunks = get_text_chunks(text)
        
        # Extract policy information
        policy_info = extract_policy_information(chunks, model)
        policy_info["Filename"] = uploaded_file.name
        policy_data.append(policy_info)
        
        # Extract coverages and benefits
        coverages_data.append({"Filename": uploaded_file.name, "Key Coverages & Benefits": policy_info["Key Coverages & Benefits"]})

    # Save the extracted coverages and benefits to a CSV file
    coverages_df = pd.DataFrame(coverages_data)
    coverages_df.to_csv("coverages.csv", index=False)
    print("Coverages and benefits saved to coverages.csv.")  # Debug statement

    # Analyze and classify the policy documents
    classifications = classify_documents(policy_data)
    classification_df = pd.DataFrame(classifications)
    classification_df.to_csv("classified_policy_information.csv", index=False)
    print("Classified policy information saved to classified_policy_information.csv.")  # Debug statement

    # Create an Excel file with summary information
    excel_df = pd.DataFrame(policy_data)
    excel_df['Premium Amount'] = excel_df['Policy Gross Premium'].apply(extract_numeric)
    excel_df['Sum Insured Amount'] = excel_df['Sum Insured'].apply(extract_numeric)
    excel_df['Tenure (years)'] = excel_df.apply(lambda row: calculate_tenure(row['Policy Period (Start)'], row['Policy Period (End)']), axis=1)
    excel_df = excel_df[['Filename', 'Policy Number', 'Premium Amount', 'Sum Insured Amount', 'Tenure (years)']]
    excel_df.to_excel("policy_document_summary.xlsx", index=False)
    print("Summary information saved to policy_document_summary.xlsx.")  # Debug statement

    return classification_df

# Function to classify policy documents based on criteria
def classify_documents(policy_data):
    classifications = []
    for policy in policy_data:
        classification = {"Filename": policy["Filename"]}
        
        # Extract and clean premium amount
        premium_str = policy["Policy Gross Premium"]
        if premium_str is not None:
            premium_str = ''.join(c for c in premium_str if c.isdigit() or c == '.')
            premium = float(premium_str) if premium_str.replace('.', '', 1).isdigit() else 0.0
        else:
            premium = 0.0
        
        # Extract and clean sum insured amount
        sum_insured_str = policy["Sum Insured"]
        if sum_insured_str is not None:
            sum_insured_str = ''.join(c for c in sum_insured_str if c.isdigit() or c == '.')
            sum_insured = float(sum_insured_str) if sum_insured_str.replace('.', '', 1).isdigit() else 0.0
        else:
            sum_insured = 0.0
        
        # Parse policy period start and end dates
        policy_period_start_str = policy["Policy Period (Start)"]
        policy_period_start = custom_date_parser(policy_period_start_str)
        
        policy_period_end_str = policy.get("Policy Period (End)")
        policy_period_end = custom_date_parser(policy_period_end_str)
        
        # Calculate tenure in years
        if policy_period_start and policy_period_end:
            tenure = (policy_period_end - policy_period_start).days / 365
        else:
            tenure = 0
        
        # Classify based on conditions
        if premium <= 1000 and sum_insured < 200000 and tenure == 1:
            classification["Rating"] = "3 Star"
        elif (premium > 1000 and premium <= 2000) or (sum_insured > 200000 and sum_insured <= 400000) or tenure == 2:
            classification["Rating"] = "4 Star"
        elif premium > 5000 or sum_insured > 500000 or tenure == 2:
            classification["Rating"] = "5 Star"
        else:
            classification["Rating"] = "Other"
        
        classifications.append(classification)
    
    return classifications

# Helper function to extract numeric values
def extract_numeric(text):
    if text is None:
        return 0.0
    cleaned_text = ''.join(c for c in text if c.isdigit() or c == '.' or c == ',')
    cleaned_text = cleaned_text.replace(',', '')
    return float(cleaned_text) if cleaned_text.replace('.', '', 1).isdigit() else 0.0

# Helper function to calculate tenure in years
def calculate_tenure(start_date, end_date):
    start_date = custom_date_parser(start_date)
    end_date = custom_date_parser(end_date)
    if start_date and end_date:
        return (end_date - start_date).days / 365
    return 0

# Streamlit app to upload and process PDF files
def main():
    st.title("Policy Document Analysis")
    
    uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        classification_df = process_uploaded_pdfs(uploaded_files)
        st.write("Processing complete.")
        
        # Display ratings in UI
        st.write("Policy Ratings:")
        st.dataframe(classification_df[['Filename', 'Rating']])
        
        # Provide download links for generated files
        st.markdown(get_download_link("coverages.csv", "Download coverages.csv"), unsafe_allow_html=True)
        st.markdown(get_download_link("classified_policy_information.csv", "Download classified_policy_information.csv"), unsafe_allow_html=True)
        st.markdown(get_download_link("policy_document_summary.xlsx", "Download policy_document_summary.xlsx"), unsafe_allow_html=True)

# Function to generate a download link for files
def get_download_link(file_path, text):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{file_path}">{text}</a>'

if __name__ == "__main__":
    main()
