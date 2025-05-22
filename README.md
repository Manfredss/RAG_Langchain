# 🦜🔗 RAG Retrieval with Langchain

## 1. Install dependencies
First install pytorch, then run the following command to install the rest of the dependencies under the same environment:
```bash
pip install -r requirements.txt
```

## 2. How to use

### 1. Start the server ✅
Using Qwen-7B as the model, use up to 80% of the GPU 
memory
```bash
python -m vllm.entrypoints.openai.api_server --model 'Qwen-7B-Chat-Int4' --trust-remote-code -q gptq -dtype float16 --gpu-memory-utilization 0.8
```

### 2. Run indexer.py to get the vector embeddings of the documents 📚
```bash
python indexer.py
```

### 3. Run rag.py to get some taste of the rag-retrieval technique 🤖💬
```bash
python rag.py
```
