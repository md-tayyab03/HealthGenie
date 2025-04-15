# 🧠 HealthGenie — Your AI Medical Assistant

HealthGenie is an intelligent chatbot that leverages vector search (FAISS), LangChain, and Hugging Face models to answer medical-related queries in a conversational way.

---

## 🚀 Features

- 📚 Context-aware medical Q&A  
- 🔍 Fast semantic search using FAISS  
- 🤖 Hugging Face transformer integration via LangChain  
- 🌐 Optional Streamlit UI for web-based interaction  

---


## ⚙️ Setting Up Your Environment with Pipenv

### 🔧 Prerequisite

Install [Pipenv](https://pipenv.pypa.io/en/latest/) if you don't have it already:

```bash
pip install pipenv
📦 Install Dependencies
Inside your project folder, run:

bash
Copy
Edit
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit
Then activate your virtual environment:

bash
Copy
Edit
pipenv shell
If you're using a cloned repo and see a Pipfile.lock, simply run:

bash
Copy
Edit
pipenv install
🔐 Set Up Your API Key
Create a .env file in your root directory:

env
Copy
Edit
HF_TOKEN=your_huggingface_token_here
🧠 How to Use
Build the memory index (only once or whenever your data updates):
bash
Copy
Edit
python create_memory.py
Launch the chatbot:
bash
Copy
Edit
streamlit run medibot.py
🛠️ Future Ideas
Add support for OpenAI, Cohere, or Anthropic models

Expand Streamlit UI features

Integrate speech-to-text for voice queries

Include live medical databases or knowledge bases

📄 License
MIT License — feel free to use, modify, and share. Just give proper credit when sharing publicly.

🙌 Contributions
Open to pull requests, improvements, or ideas! If it helps people, it's worth contributing.

🛠️ Tech Stack
Layer	Tools/Libs	Purpose
Language	Python	Core programming language
Env Mgmt	Pipenv	Virtual environment & dependency manager
LLM Framework	LangChain	LLM orchestration and memory handling
Vector DB	FAISS	Fast similarity search on embedded data
Embedding	Hugging Face Transformers + sentence-transformers	Convert text to dense vectors
PDF Support	PyPDF	Extract text from PDFs
UI (optional)	Streamlit	Simple frontend to interact with the bot
Secrets Mgmt	Python dotenv (.env)	Keep your API tokens and secrets safe
Contact me: mohammedtayyab242@gmail.com

pgsql
Copy
Edit

You can copy and paste this directly into your `README.md`. This version eliminates unnecessary details, formats the commands neatly, and presents the information clearly and concisely.