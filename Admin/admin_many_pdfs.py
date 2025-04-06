import os
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# AWS S3 Configuration
s3_client = boto3.client("s3")
BUCKET_NAME = "yojitha-chat-with-pdf"

# Ensure AWS Region is Set
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Initialize Bedrock Client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Initialize Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

folder_path = "/tmp/"

# --- Utility Functions ---

def clean_file_name(file_name):
    return "".join(c if c.isalnum() or c in ('.', '_') else "_" for c in file_name)

def split_text(pages, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(pages)

def faiss_exists_in_s3(index_name):
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=f"faiss_files/{index_name}.faiss")
        return True
    except:
        return False

def create_vector_store(file_name, documents):
    local_folder = os.path.join(folder_path, file_name)
    os.makedirs(local_folder, exist_ok=True)

    faiss_index_path = os.path.join(local_folder, "index")
    pkl_path = os.path.join(local_folder, "index.pkl")

    if faiss_exists_in_s3(file_name):
        st.success(f"✅ FAISS index for `{file_name}` already exists in S3. Skipping.")
        return True

    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    vectorstore_faiss.save_local(index_name="index", folder_path=local_folder)

    s3_client.upload_file(faiss_index_path + ".faiss", BUCKET_NAME, f"faiss_files/{file_name}.faiss")
    s3_client.upload_file(pkl_path, BUCKET_NAME, f"faiss_files/{file_name}.pkl")

    st.success(f"✅ FAISS index for `{file_name}` created and uploaded to S3!")

    return True

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return split_text(pages)

# --- Streamlit App ---

def main():
    st.title("Chat with Your PDF")

    # Upload PDFs
    st.subheader("Upload PDFs to Create Searchable Index")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            original_file_name = os.path.splitext(uploaded_file.name)[0]
            clean_name = clean_file_name(original_file_name)

            st.write(f"Processing PDF: {uploaded_file.name}")

            # Save uploaded PDF to /tmp
            saved_file_path = os.path.join("/tmp", f"{clean_name}.pdf")
            with open(saved_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                documents = load_and_split_pdf(saved_file_path)
                create_vector_store(clean_name, documents)
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")

if __name__ == "__main__":
    main()
