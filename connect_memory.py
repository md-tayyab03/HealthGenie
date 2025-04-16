import os
import warnings
import logging
import os
import sys

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional: Suppress Huggingface_hub logs if any
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Redirect stderr (warnings and errors) to null
sys.stderr = open(os.devnull, 'w')


from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS



#Step - 1: Setup LLM(Mistral)

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        task="text-generation",
        temperature = 0.5,
        model_kwargs = {"token":HF_TOKEN,
                        "max_length":512})
    return llm

#Step - 2: Connect LLM with FAISS and create Chain

CUSTOM_PROMPT_TEMPLATE = """
Use the info provided in context to answer user's question, if you don't know answer just say "Don't No", dont try to makeup an answer by own and don't give out of context

Context: {context}
Question: {question}

Start the answer directly. No small talks and give answer in proper alligned manner for example

1. answer the user's question
2. give the reference 


"""

def set_customprompt(custom_prompt_template):
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = ["context","question"])
    return prompt

#Load Database

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {"device": "cpu"})
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


#Create QA Chain

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(HUGGING_FACE_REPO_ID),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {'k':2}),
    return_source_documents = True,
    chain_type_kwargs = {'prompt':set_customprompt(CUSTOM_PROMPT_TEMPLATE)}
)

#Invoke the chain with single Query

user_query = input("How Can I Help!! 🐰 : ")
response = qa_chain.invoke({'query':user_query})

# #Print the response in Terminal

# print("\n🧠 Answer:\n")
# print(response["result"])

# print("\n📚 Source Documents:\n")
# for i, doc in enumerate(response["source_documents"], 1):
#     page = doc.metadata.get('page_label', 'Unknown')
#     source = doc.metadata.get('source', 'Unknown')
#     excerpt = doc.page_content.strip().replace('\n', ' ')[:400]  # Trim and format excerpt

#     print(f"🔹 Source {i}")
#     print(f"   📄 Page: {page}")
#     print(f"   📁 File: {source}")
#     print(f"   📜 Excerpt: {excerpt}...\n")
