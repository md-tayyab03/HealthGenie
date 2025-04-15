# 🧠 HealthGenie — Your AI Medical Assistant

HealthGenie is an intelligent chatbot that leverages vector search (FAISS), LangChain, and Hugging Face models to answer medical-related queries in a conversational way.

---

## 🚀 Features

- 📚 Context-aware medical Q&A  
- 🔍 Fast semantic search using FAISS  
- 🤖 Hugging Face transformer integration via LangChain  
- 🌐 Optional Streamlit UI for web-based interaction  

---

## 📁 Project Structure

Medibot/ │ ├── vectorstore/db_faiss/ │ ├── index.faiss # Vector index │ └── index.pkl # Metadata associated with vectors │ ├── connect_memory.py # Loads the vector DB and retrieves answers ├── create_memory.py # Builds the FAISS vector index from documents ├── medibot.py # Main chatbot script │ ├── .env # Hugging Face token (not tracked in Git) ├── Pipfile # Pipenv dependency manager file ├── Pipfile.lock # Locked dependency versions └── README.md # Project overview and setup instructions


---

## ⚙️ Setting Up Your Environment with Pipenv

### 🔧 Prerequisite

Install [Pipenv](https://pipenv.pypa.io/en/latest/) if you don't have it already:


pip install pipenv
📦 Install Dependencies
Inside your project folder, run:


pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit
Then activate your virtual environment:


pipenv shell
If you're using a cloned repo and see a Pipfile.lock, simply run:



pipenv install
🔐 Set Up Your API Key
Create a .env file in your root directory:



HF_TOKEN=your_huggingface_token_here


🧠 How to Use
Build the memory index (only once or whenever your data updates):



Launch the chatbot: streamlit run medibot.py


🛠️ Future Ideas

Add support for OpenAI, Cohere, or Anthropic models

Expand Streamlit UI features

Integrate speech-to-text for voice queries

Include live medical databases or knowledge bases

📄 License
MIT License — feel free to use, modify, and share. Just give proper credit when sharing publicly.

🙌 Contributions
Open to pull requests, improvements, or ideas! If it helps people, it's worth contributing.


## 🛠️ Tech Stack

| Layer            | Tools/Libs                               | Purpose                                      |
|------------------|-------------------------------------------|----------------------------------------------|
| **Language**     | Python                                    | Core programming language                    |
| **Env Mgmt**     | Pipenv                                    | Virtual environment & dependency manager     |
| **LLM Framework**| LangChain                                 | LLM orchestration and memory handling        |
| **Vector DB**    | FAISS                                     | Fast similarity search on embedded data      |
| **Embedding**    | Hugging Face Transformers + `sentence-transformers` | Convert text to dense vectors       |
| **PDF Support**  | PyPDF                                     | Extract text from PDFs                       |
| **UI (optional)**| Streamlit                                 | Simple frontend to interact with the bot     |
| **Secrets Mgmt** | Python `dotenv` (`.env`)                  | Keep your API tokens and secrets safe        |


Contact me : mohammedtayyab242@gmail.com