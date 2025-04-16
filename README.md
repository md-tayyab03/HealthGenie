# 🧠 HealthGenie — Your AI Medical Assistant

HealthGenie is an intelligent chatbot that leverages vector search (FAISS), LangChain, and Hugging Face models to answer medical-related queries in a conversational way.

---

## 🚀 Features

- 📚 Context-aware medical Q&A  
- 🔍 Fast semantic search using FAISS  
- 🤖 Hugging Face transformer integration via LangChain  
- 🌐 Optional Streamlit UI for web-based interaction  

---

## ⚙️ Setting Up with Pipenv

Clone the repo and Install dependencies:

``` pip install pipenv ```
``` pipenv install langchain langchain_community langchain_huggingface``` 
``` pipenv install faiss-cpu pypdf huggingface_hub streamlit ```

Activate the virtual environment:``` pipenv shell ```

Set Up API Key Create a .env file with: ``` HF_TOKEN=your_huggingface_token_here ```

Run the project : ``` streamlit run medibot.py ```


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



🙌 Contributions

Feel free to contribute! 
Contact me : mohammedtayyab242@gmail.com

