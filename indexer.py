from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 解析 pdf，切成 chunk 片段
# 同时允许使用 OCR 提取图片中文字
pdf_loader = PyPDFLoader("./test.pdf", extract_images=True)
# 每 50 个字切一段，每一段与上一段重叠 10 个字，这样能保留一些上下文信息
chunks = pdf_loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10))

# 加载embedding模型
embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')

# 将 chunk 插入到 FAISS 本地向量数据库中
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("LLM.faiss")

print('FAISS saved')