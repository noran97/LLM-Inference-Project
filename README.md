# LLM Inference Project

This repository contains a C++ wrapper class, **LLMInference**, designed to facilitate running inference with a Large Language Model (LLM) using the **llama.cpp** library.  
The code is specifically structured to handle **multi-turn conversational history** and perform **efficient, token-by-token generation**.

---

## ✨ Features

### 🔹 Model Management
- Easily load and manage a **GGUF-formatted** language model.

### 🔹 Conversational History
- Manages the **full chat history** by re-tokenizing the entire conversation on each turn.  
- Ensures the model always has the **complete context**.

### 🔹 Token-by-Token Generation
- Supports **streaming-like output** by generating and returning one token at a time.

### 🔹 Robust llama.cpp Integration
- Correctly handles the `llama_batch` and **token positions**.  
- Prevents common runtime errors associated with **multi-turn conversations**.
