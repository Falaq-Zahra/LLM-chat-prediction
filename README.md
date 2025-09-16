# LLM-chat-prediction
Predictive LLM Model for WhatsApp Conversations 

## Project Overview
This project builds a **Retrieval-Augmented Generation (RAG)** system that predicts replies in WhatsApp conversations. Using a dataset of chat messages, the model retrieves semantically similar past contexts and uses a large language model (LLM) to generate human-like responses in **short, Roman Urdu + English mix** style.

##  Workflow
**Data → Preprocessing → Embeddings → FAISS Index → Retrieval (Top-5) → LLM (RAG) → Reply Generation**

1. **Data Loading & Cleaning**  
   - Load WhatsApp chat dataset (CSV).  
   - Convert timestamps, remove emojis, and normalize text.

2. **Pair Creation**  
   - Construct `(context, reply)` pairs only when sender changes.

3. **Embeddings**  
   - Use pretrained model [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) from SentenceTransformers.  
   - Generate vector embeddings for all contexts.

4. **Indexing with FAISS**  
   - Store embeddings in a FAISS index.  
   - Enable efficient nearest-neighbor search for relevant contexts.

5. **Retrieval-Augmented Generation (RAG)**  
   - For a user message, retrieve **top-5** most similar past contexts.  
   - Construct a prompt with system instructions + retrieved examples.  
   - Use an LLM (via **Groq API**, with OpenAI-compatible model call) to generate a reply.

6. **Evaluation**  
   - Evaluate with BLEU (sacreBLEU) and ROUGE-L metrics.  
   - Example results: BLEU ≈ 0.25, ROUGE-L ≈ 0.32 (on sample run).  
   - Note: These scores are approximate since chat replies are open-ended and multiple correct answers may exist.

7. **Interactive Chat**  
   - A simple function `chat_with_model()` is provided.  
   - It runs an interactive loop where the user types a message and the model generates a reply.  
   - Type `exit` to quit the loop.

##  Libraries Used
- **pandas, numpy** → Data handling and processing.
- **emoji, re** → Text cleaning.
- **sentence-transformers** → Pretrained embedding model.
- **faiss** → Vector similarity search.
- **groq / openai** → LLM inference (Groq API with OpenAI-compatible client).
- **sacrebleu, rouge_score** → Evaluation metrics.
- **tqdm, random** → Utility functions.

##  Installation
```bash
# Clone repo
git clone https://github.com/your-username/llm-chat-prediction.git
cd llm-chat-prediction

# Install dependencies
pip install -r requirements.txt
```

##  Usage
1. Prepare your WhatsApp chat dataset (`Mychat-clean-data.csv`).
2. Run the notebook: `LLM_Task_Chat_Prediction_model.ipynb`.
3. Try interactive chat:
```python
chat_with_model()
```

##  Evaluation Example
| Metric   | Score (sample run) |
|----------|-------------------|
| BLEU     | ~0.25             |
| ROUGE-L  | ~0.32             |

 Note: Automatic metrics underestimate quality because multiple valid replies exist in chat.

##  Limitations & Ethics
- **Privacy:** WhatsApp data is personal → anonymize & use only with consent.
- **Evaluation metrics:** BLEU/ROUGE not ideal for open-ended chat.
- **No fine-tuning:** Only pretrained embeddings + LLM with RAG were used.

##  Future Improvements
-Future Improvements
- collect more data to feed for the model
- Fine-tune embeddings or LLM on domain-specific data.
- Add better evaluation (human judgement, diversity metrics).
- Build a proper chat UI (web/streamlit).

---
 Author: *Falaq Zahra*  
🔗 Based on SentenceTransformers + FAISS + Groq API
