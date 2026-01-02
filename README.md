#  End-to-End Q&A Chatbot

**Streamlit · LangChain · OpenRouter**

An interactive **Q&A chatbot web application** built using **Streamlit** and **LangChain**, powered by large language models via **OpenRouter**. The app allows users to ask natural-language questions and receive intelligent responses with configurable model settings.

[Website_link](https://chatbotopenroute.streamlit.app/)
![website UI](./Screenshot%202026-01-02%20at%208.18.01 AM.png)
---

## 🏗️ Tech Stack

* **Python 3.10+**
* **Streamlit**
* **LangChain**
* **OpenRouter API**
* **dotenv**
* **LangSmith (optional)**

---

## 🔑 API Key Setup (Important)

This project uses **OpenRouter** to access LLMs.

👉 Generate your API key from:
**[https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)**

⚠️ **Never share or commit your API key publicly.**

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/End-to-End-Q-A-chatbot.git
cd End-to-End-Q-A-chatbot
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Environment Variables (Optional)

Create a `.env` file if you want LangSmith tracking:

```env
LANGCHAIN_API_KEY=your_langsmith_key
```

> API keys for OpenRouter are entered **securely via the Streamlit sidebar**, not via `.env`.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```
---

## 🤖 Supported Models (Examples)

* `openai/gpt-4o`
* `openai/gpt-4-turbo`
* `openai/gpt-4.1-mini`
* `openai/gpt-4`

---

