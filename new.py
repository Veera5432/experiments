import streamlit as st
import fitz  # PyMuPDF
import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

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
        "Policy Gross Premium": "Extract policy gross premium from the uploaded policy document.",
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
        except Exception as e:
            print(f"Error during extraction for {key}: {e}")
            extracted_info[key] = None
    return extracted_info

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

    # Analyze and classify the policy documents
    base_document = policy_data[0]  # Consider the first document as the base for comparison
    classifications = classify_documents(policy_data, base_document)
    classification_df = pd.DataFrame(classifications)
    classification_df.to_csv("classified_policy_information.csv", index=False)

    # Display the ratings in the UI
    st.write("Policy Ratings:")
    st.dataframe(classification_df)


def classify_documents(policy_data, base_document):
    classifications = []
    for policy in policy_data:
        classification = {"Filename": policy["Filename"]}
        # Your custom classification logic
        premium_str = policy["Policy Gross Premium"]
        # Remove non-numeric characters from premium string
        premium_str = ''.join(c for c in premium_str if c.isdigit() or c == '.')
        premium = float(premium_str) if premium_str.replace('.', '', 1).isdigit() else 0.0

        sum_insured_str = policy["Sum Insured"]
        if sum_insured_str is not None:
            # Remove non-numeric characters from sum insured string
            sum_insured_str = ''.join(c for c in sum_insured_str if c.isdigit() or c == '.')
            sum_insured = float(sum_insured_str) if sum_insured_str.replace('.', '', 1).isdigit() else 0.0
        else:
            sum_insured = 0.0

        policy_period_start = pd.to_datetime(policy["Policy Period (Start)"])
        policy_period_end_str = policy.get("Policy Period (End)")
        if policy_period_end_str:
            policy_period_end = pd.to_datetime(policy_period_end_str)
            tenure = (policy_period_end - policy_period_start).days / 365
        else:
            # Handle case where policy end date is not found
            tenure = 0  # Set tenure to 0 or handle differently based on your requirement

        if premium <= 1000 and sum_insured < 200000 and tenure == 1:
            classification["Rating"] = "3 Star"
        elif premium <= 2000 and sum_insured < 500000 and tenure == 2:
            classification["Rating"] = "4 Star"
        else:
            classification["Rating"] = "5 Star"

        classifications.append(classification)
    return classifications




# Streamlit app to upload PDF folder and process the files
def main():
    st.title("Policy Document Analysis")

    uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        folder_path = "uploaded_pdfs"
        os.makedirs(folder_path, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(folder_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        process_pdf_folder(folder_path)
        st.write("Processing complete. Check the extracted_policy_information.csv and classified_policy_information.csv files.")

if __name__ == "__main__":
    main()
