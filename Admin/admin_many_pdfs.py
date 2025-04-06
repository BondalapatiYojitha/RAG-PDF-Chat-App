import os
import streamlit as st
import boto3
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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

# Load FAISS index
def load_faiss_index(index_name):
    faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
    pkl_file = os.path.join(folder_path, f"{index_name}.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.faiss", faiss_file)
        try:
            s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.pkl", pkl_file)
        except:
            pass

    return FAISS.load_local(
        index_name=index_name,
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

# Initialize LLM
def get_llm():
    return Bedrock(
        model_id="anthropic.claude-v2:1",
        client=bedrock_client,
        model_kwargs={'max_tokens_to_sample': 800}
    )

# Customized QA Chain (Summary Friendly Prompt)
def build_qa_chain(vectorstore):
    prompt_template = """
Human: Please provide a clear, concise **summary** of the given document chunks. 
Focus on the main topics, conclusions, and important points.
If you cannot find enough context, just say "I don't know."

<context>
{context}
</context>

Summary:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15}),
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

# Streamlit App
def main():
    st.title("Ask Questions about Your Uploaded PDFs")

    # List all indexes
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="faiss_files/")
    if "Contents" in response:
        indexes = sorted(set(obj["Key"].split("/")[-1].split(".")[0] for obj in response["Contents"] if obj["Key"].endswith(".faiss")))
    else:
        indexes = []

    if not indexes:
        st.error("No FAISS indexes found in the S3 bucket. Please upload PDFs first.")
        return

    selected_index = st.selectbox("Select a document to query:", indexes)

    user_query = st.text_input("Ask a question (example: What is this document all about?)")

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            vectorstore = load_faiss_index(selected_index)
            qa_chain = build_qa_chain(vectorstore)
            result = qa_chain.invoke({"query": user_query})
            st.success("Answer:")
            st.write(result["result"])

if __name__ == "__main__":
    main()
