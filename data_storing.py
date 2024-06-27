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
        "Policy Gross Premium": "Extract policy gross premium or premium amount from the uploaded policy document, extract only numeric value.",
        "Policy Period (Start)": "Extract the policy start date from the uploaded policy document.",
        "Policy Period (End)": "Extract the policy end date from the uploaded policy document.",
        "Sum Insured": "Extract the sum insured from the uploaded policy document.",
        "Key Coverages & Benefits": "Extract key coverages and benefits including the limits from the uploaded policy document."
    }
    extracted_info = {}
    for key, prompt in prompt_template.items():
        try:
            response = model.generate_content(prompt + " " + ' '.join(chunks))
            extracted_info[key] = response.text.strip()
            print(f"Extracted {key}: {extracted_info[key]}")  # Debug statement
        except Exception as e:
            print(f"Error during extraction for {key}: {e}")
            extracted_info[key] = None
    return extracted_info

# Function to parse custom date formats
def custom_date_parser(date_str):
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except ValueError:
        try:
            return datetime.strptime(date_str, "Midnight %d-%b-%Y")
        except ValueError:
            return None

# Main function to process PDF files uploaded via Streamlit
def process_uploaded_pdfs(uploaded_files):
    model = genai.GenerativeModel(model_name='gemini-pro', safety_settings=safety_settings)
    policy_data = []
    
    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.read()
        text = extract_text_from_bytes(pdf_bytes)
        chunks = get_text_chunks(text)

        # Extract policy information
        policy_info = extract_policy_information(chunks, model)
        policy_info["Filename"] = uploaded_file.name
        policy_data.append(policy_info)

    # Prepare data for CSV (only filename and key coverages)
    policy_data_csv = [{'Filename': data['Filename'], 'Key Coverages & Benefits': data['Key Coverages & Benefits']} for data in policy_data]
    
    # Save the extracted information to a CSV file
    if os.path.exists("policy_data.csv"):
        existing_policy_df = pd.read_csv("policy_data.csv")
        policy_df_csv = pd.concat([existing_policy_df, pd.DataFrame(policy_data_csv)], ignore_index=True)
    else:
        policy_df_csv = pd.DataFrame(policy_data_csv)
    
    policy_df_csv.to_csv("policy_data.csv", index=False, columns=['Filename', 'Key Coverages & Benefits'], header=True)
    print("Policy information saved to policy_data.csv.")

    # Prepare data for Excel
    excel_data = []
    for data in policy_data:
        row = {
            'Filename': data.get('Filename', ''),
            # 'Poicy Number': data.get("Policy Number", ""),
            'Policy Number': extract_numeric(data.get('Policy Number')),
            'Premium Amount': extract_numeric(data.get('Policy Gross Premium').replace('Rs.', '').replace(',', '')),  # Adjust for specific format
            'Sum Insured Amount': extract_numeric(data.get('Sum Insured').replace(',', '').replace('Rs.', '')),  # Adjust for specific format
            'Tenure (years)': calculate_tenure(data.get('Policy Period (Start)'), data.get('Policy Period (End)'))
        }
        excel_data.append(row)
    
    # Load existing Excel data if file exists
    if os.path.exists("policy_data.xlsx"):
        existing_excel_df = pd.read_excel("policy_data.xlsx")
        excel_df = pd.concat([existing_excel_df, pd.DataFrame(excel_data)], ignore_index=True)
    else:
        excel_df = pd.DataFrame(excel_data)
    
    # Save summary information to Excel
    excel_df.to_excel("policy_data.xlsx", index=False, header=True)
    print("Summary information saved to policy_data.xlsx.")

    return excel_df

# Helper function to extract numeric values
def extract_numeric(text):
    if text is None:
        return 0.0
    if isinstance(text, (int, float)):
        return float(text)
    if isinstance(text, type(None)):
        return None
    # Strip non-numeric characters except '.' and ','
    cleaned_text = ''.join(c for c in text if c.isdigit() or c == '.' or c == ',')
    # Replace ',' with '' to handle thousands separators
    cleaned_text = cleaned_text.replace(',', '')
    try:
        return float(cleaned_text)
    except ValueError:
        return 0.0

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
        summary_df = process_uploaded_pdfs(uploaded_files)
        st.write("Processing complete.")
        
        # Display summary in UI
        st.write("Policy Summary:")
        st.dataframe(summary_df)
        
        # Provide download links for generated files
        st.markdown(get_download_link("policy_data.csv", "Download policy_data.csv"), unsafe_allow_html=True)
        st.markdown(get_download_link("policy_data.xlsx", "Download policy_data.xlsx"), unsafe_allow_html=True)

# Function to generate a download link for files
def get_download_link(file_path, text):
    if not os.path.exists(file_path):
        return f"File '{file_path}' not found."
    
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:file/{file_path.split(".")[-1]};base64,{b64}" download="{file_path}">{text}</a>'

if __name__ == "__main__":
    main()
