# **HealthGenie — Your AI Medical Assistant**

HealthGenie is an AI-powered chatbot that answers medical-related questions in a conversational way. It uses LangChain for managing dialogue, FAISS for fast vector-based search, and Hugging Face models for generating accurate medical responses. With a Streamlit interface and support for PDF uploads, it helps users access and understand medical information easily.

---

## **Features**

- Context-aware medical Q&A  
- Fast semantic search using FAISS  
- Hugging Face transformer integration via LangChain  
- Optional Streamlit UI for web-based interaction  

---

## **Setting Up with Pipenv**

- Clone the repository & Install Pipenv: `pip install pipenv`
- Install dependencies: 
    - `pipenv install langchain langchain_community langchain_huggingface`
    - `pipenv install faiss-cpu pypdf huggingface_hub streamlit`
- Activate the virtual environment: `pipenv shell`
- Set up API key & create a .env file in the root directory
    - Add the following line to the file: `HF_TOKEN=your_huggingface_token_here`
- Run the project: `streamlit run medibot.py`

---

## **🛠️ Tech Stack**

- Python
- Pipenv                                    
- LangChain                                 
- FAISS                                     
- Hugging Face Transformers + `sentence-transformers` 
- PyPDF
- Streamlit                                 
- Python `dotenv` (`.env`)                  


Feel free to **🙌 Contribute**,
Contact me : **mohammedtayyab242@gmail.com**

