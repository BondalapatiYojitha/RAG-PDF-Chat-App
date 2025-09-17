import os
import boto3
import base64
import streamlit as st
# from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# AWS S3 Configuration
s3_client = boto3.client("s3")
BUCKET_NAME = "yojitha-chat-with-pdf"

# Set AWS Region
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Initialize Bedrock Clients
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
# bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

folder_path = "/tmp/"

# ---- Utility Functions ----

def clean_file_name(file_name):
    return "".join(c if c.isalnum() or c in ('.', '_') else "_" for c in file_name)

def split_text(pages, chunk_size=3000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
        st.success(f"✅ FAISS index for `{file_name}` already exists. Skipping.")
        return True

    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    vectorstore_faiss.save_local(index_name="index", folder_path=local_folder)

    s3_client.upload_file(faiss_index_path + ".faiss", BUCKET_NAME, f"faiss_files/{file_name}.faiss")
    s3_client.upload_file(pkl_path, BUCKET_NAME, f"faiss_files/{file_name}.pkl")

    return True

def upload_pdf_to_s3(file_path, file_name):
    try:
        s3_client.upload_file(file_path, BUCKET_NAME, f"faiss_files/{file_name}.pdf")
    except Exception as e:
        st.error(f"Failed to upload PDF to S3: {e}")

def list_indexes():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="faiss_files/")
    if "Contents" in response:
        return sorted(set(obj["Key"].split("/")[-1].split(".")[0] for obj in response["Contents"] if obj["Key"].endswith(".faiss")))
    return []

def load_faiss_index(index_name):
    faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
    pkl_file = os.path.join(folder_path, f"{index_name}.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.faiss", faiss_file)
        try:
            s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.pkl", pkl_file)
        except:
            pass

    return FAISS.load_local(index_name=index_name, folder_path=folder_path, embeddings=bedrock_embeddings, allow_dangerous_deserialization=True)

def load_all_vectorstores():
    indexes = list_indexes()
    all_vectorstores = []
    for idx in indexes:
        vs = load_faiss_index(idx)
        all_vectorstores.append(vs)
    if all_vectorstores:
        merged = all_vectorstores[0]
        for other in all_vectorstores[1:]:
            merged.merge_from(other)
        return merged
    return None

def delete_document(index_name):
    errors = []
    for ext in [".pdf", ".faiss", ".pkl"]:
        try:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=f"faiss_files/{index_name}{ext}")
        except Exception as e:
            errors.append(str(e))

    if errors:
        st.warning(f"⚠️ Some files may not have been deleted: {errors}")
    else:
        st.success(f"🗑️ Deleted all related files for `{index_name}` successfully!")

def load_full_document_text(index_name):
    local_pdf_path = os.path.join(folder_path, f"{index_name}.pdf")
    if not os.path.exists(local_pdf_path):
        try:
            s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.pdf", local_pdf_path)
        except Exception as e:
            st.error(f"❌ PDF file not found: {e}")
            return ""
    loader = PyPDFLoader(local_pdf_path)
    pages = loader.load()
    return "\n".join(page.page_content for page in pages)

def get_llm():
    # return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 800})
    return ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        client=bedrock_client,
        max_tokens=800,
    )

def build_qa_chain(vectorstore):
    prompt_template = """
Human: Please provide a clear, concise answer from the provided context.

<context>
{context}
</context>

Answer:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=get_llm(), retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15}), return_source_documents=False, chain_type="stuff", chain_type_kwargs={"prompt": PROMPT})

def ask_deep_question(full_text, user_question):
    llm = get_llm()
    prompt = f"""
Human: You are given a full document and a user question.

Even if the document covers multiple topics or is broad, please do your best to summarize the main ideas, key points, and important insights clearly and concisely.

<Document>
{full_text}
</Document>

Question: {user_question}

Assistant:
"""
    response = llm.invoke(prompt)
    return response

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="Chat with Your PDF", layout="wide")

    st.markdown("""
        <style>
            .block-container {padding-top: 1rem !important;}
            header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📂 Upload / Manage PDFs", "🔍 Ask Questions"])

    with tab1:
        st.header("Upload PDFs to Create Searchable Index")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                clean_name = clean_file_name(os.path.splitext(uploaded_file.name)[0])
                saved_file_path = os.path.join(folder_path, f"{clean_name}.pdf")
                with open(saved_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                try:
                    loader = PyPDFLoader(saved_file_path)
                    pages = loader.load_and_split()
                    documents = split_text(pages)
                    create_vector_store(clean_name, documents)
                    upload_pdf_to_s3(saved_file_path, clean_name)
                    st.success(f"✅ Uploaded and indexed `{uploaded_file.name}`!")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")

        st.header("Manage Uploaded Documents")
        indexes = list_indexes()
        if indexes:
            search_text = st.text_input("🔍 Search uploaded documents")
            filtered = [i for i in indexes if search_text.lower() in i.lower()]
            selected_deletes = st.multiselect("Select documents to delete", filtered)
            confirm_delete = st.checkbox("⚠️ Confirm delete")
            if st.button("Delete Selected Documents"):
                if confirm_delete and selected_deletes:
                    for doc in selected_deletes:
                        delete_document(doc)
                elif not selected_deletes:
                    st.warning("⚠️ Please select at least one document.")
                else:
                    st.warning("⚠️ Please confirm delete first.")
        else:
            st.info("No documents found yet.")

    with tab2:
        col1, col2 = st.columns([3,7])

        with col1:
            st.header("Ask Questions from Uploaded Documents")
            indexes = list_indexes()
            if not indexes:
                st.warning("No documents found.")
                return

            selected_index = st.selectbox("Select a document", indexes)
            user_query = st.text_input("Ask your question")

            if st.button("Ask Question (Auto Mode)"):
                with st.spinner("Thinking..."):
                    vectorstore = load_faiss_index(selected_index)
                    qa_chain = build_qa_chain(vectorstore)
                    result = qa_chain.invoke({"query": user_query})
                    answer = result["result"]

                    weak_phrases = ["i don't know", "not enough information", "insufficient context", "unable to answer"]
                    if any(w in answer.lower() for w in weak_phrases) or len(answer.strip().split()) < 20:
                        st.warning("⚠️ Retrieved answer incomplete. Trying Deep Mode...")
                        full_text = load_full_document_text(selected_index)
                        if full_text:
                            deep_answer = ask_deep_question(full_text, user_query)
                            st.success("✅ Deep Full Document Answer:")
                            st.write(deep_answer)
                        else:
                            st.error("❌ Could not load full document for deep answering.")
                    else:
                        st.success("✅ Answer (Quick Retrieval):")
                        st.write(answer)

            if st.button("Search Across All Documents"):
                with st.spinner("Searching all documents..."):
                    merged_store = load_all_vectorstores()
                    if merged_store:
                        qa_chain = build_qa_chain(merged_store)
                        result = qa_chain.invoke({"query": user_query})
                        st.success("✅ Answer across all documents:")
                        st.write(result["result"])
                    else:
                        st.error("❌ No documents available to search.")

        with col2:
            local_pdf_path = os.path.join(folder_path, f"{selected_index}.pdf")
            if os.path.exists(local_pdf_path):
                with open(local_pdf_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900px"></iframe>', unsafe_allow_html=True)
            else:
                st.warning("PDF Preview not available.")

if __name__ == "__main__":
    main()



