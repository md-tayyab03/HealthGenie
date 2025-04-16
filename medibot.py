import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_customprompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512
        }
    )
    return llm

def add_message(role, content, avatar):
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
    st.session_state.message.append({'role': role, 'content': content, 'avatar': avatar})

def main():
    st.title("Hi👋, HealthGenie 💬 ")

    if 'message' not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        with st.chat_message(message['role'], avatar=message.get('avatar', None)):
            st.markdown(message['content'])

    prompt = st.chat_input("Enter your prompt here")

    if prompt:
        add_message('user', prompt, avatar="🧑‍💻")

        CUSTOM_PROMPT_TEMPLATE = """
Use the information provided in the context to answer the user's question.
If you don't know the answer, just say "Don't know." Do not make up an answer or go outside the given context.

Context:
{context}

Question:
{question}

Provide your answer using the following detailed and structured format:

1. **Comprehensive Answer to the Question: **  
   Begin with a clear, thorough, and complete answer. Address all parts of the question and provide depth, avoiding overly brief or vague statements.

2. **Step-by-Step Explanation / Reasoning: **  
   Break down the reasoning, logic, or process used to arrive at the answer. Explain each step clearly, especially if the question involves analysis, interpretation, or problem-solving.

3. **Supporting Evidence or References: **  
   Cite specific parts of the context that support the answer. If applicable, include any in-text evidence, quotes, or references that were crucial to the conclusion.

4. **Additional Insights or Clarifications (if needed): **  
   Add any extra helpful information—such as implications, definitions, alternative interpretations, or important related points that provide more clarity or depth.

Make sure the entire response is:
- Well-organized with bullet points or subheadings if needed
- Written in a professional and informative tone
- Easy to read and understand
"""


        HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        if not HF_TOKEN:
            st.error("Hugging Face token is not set. Please set HF_TOKEN as an environment variable.")
            return

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            llm = load_llm(huggingface_repo_id=HUGGING_FACE_REPO_ID, HF_TOKEN=HF_TOKEN)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_customprompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]

            # Format structured sources
            sources = "\n\n---\n**📚 Source Documents:**\n"
            for i, doc in enumerate(source_documents, 1):
                page = doc.metadata.get('page', 'Unknown')
                excerpt = doc.page_content.strip().replace('\n', ' ')[:400]
                sources += f"\n🔹 **Source {i}**\n"
                sources += f"   📄 Page: {page}\n"
                sources += f"   📜 *Excerpt:* {excerpt}...\n"

            final_result = f"🧠 **Answer:**\n\n{result.strip()}{sources}"
            add_message('assistant', final_result, avatar="🤖")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
