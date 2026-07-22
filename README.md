# Simple Go RAG Chat Application

An easy-to-understand AI chat system built entirely with Go. This project lets you ask questions about your own documents (PDFs, text files, or images) and get accurate answers using AI.

---

## 🧐 What is this project?

When you ask a standard AI a question, it only knows what it was trained on. It doesn't know about *your* private files or local notes. 

This project fixes that. It lets you upload files through a web interface, indexes them, and feeds the relevant parts to an AI so it can answer your questions based **specifically on your data**.

---

## ⚙️ What does it do?

1. **Reads Your Documents:** Processes uploaded text files and PDFs, breaking them down into small readable pieces for search.
2. **Handles & Captions Images:** Allows you to index image descriptions into your knowledge base or generate real-time AI descriptions using Vision models.
3. **Smart Search:** Searches a database (`pgvector`) to find the exact parts of your documents that relate to your question.
4. **Streams Responses:** Generates answers word-by-word in real time via Server-Sent Events (SSE).
5. **Two Ways to Chat:** You can talk to it directly through your **terminal (command line)** or using a **web browser UI**.
6. **Works Offline or Online:** Works with cloud AI like OpenAI or free, local AI running on your laptop using Ollama.

---

## 🛠️ Tools Used

* **Go (Golang):** The programming language used for the backend and routing (Chi router).
* **Postgres + pgvector:** A database used to store and search through document vector embeddings.
* **Docker:** Used to easily run the database on your machine without manual setup.
* **OpenAI or Ollama:** The AI models that embed text, analyze images, and generate answers.