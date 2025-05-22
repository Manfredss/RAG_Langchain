from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate,\
                                    SystemMessagePromptTemplate,\
                                    HumanMessagePromptTemplate,\
                                    AIMessagePromptTemplate,\
                                    MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from operator import itemgetter
import os
# from langchain.memory import ChatMessageHistory
# from langchain_core.runnables import RunnablePassthrough


embeddings = ModelScopeEmbeddings(model_name="iic/nlp_corom_sentence-embedding_chinese-base")

# 加载 FAISS 向量库，用于知识召回
vector_db = FAISS.load_local("LLM.faiss", embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 用 vllm 部署 openai 兼容的服务器端口，使用 ChatOpenAI 客户端
os.environ["VLLM_USE_MODELSCOPE"] = 'True'
chat = ChatOpenAI(model="Qwen-7B-Chat-Int4",
                  openai_api_key='EMPTY',
                  openai_api_base="http://127.0.0.1:8080/v1",
                  stop=['<|im_end|>'])

# 编写 Prompt
system_prompt = SystemMessagePromptTemplate.from_template("""
你是一个基于知识图谱的问答助手，请根据提供的知识图谱回答问题。
请使用中文回答。
""")
user_prompt = HumanMessagePromptTemplate.from_template('''
仅根据下面的上下文回答问题: 
                                                       
{context}
                                                       
Question: {query}
''')

full_chat_prompt = ChatPromptTemplate.from_messages([system_prompt, MessagesPlaceholder(variable_name='chat_history'), user_prompt])

# 聊天 chain
chat_chain = {'context': itemgetter('query') | retriever,
              'query': itemgetter('query'),
              'chat_history': itemgetter('chat_history')} | full_chat_prompt | chat

# 开始对话
chat_history = []
while True:
    query = input('请输入问题：')
    response = chat_chain.invoke({'query': query, 'chat_history': chat_history})
    chat_history.extend(HumanMessage(content=query), response)
    print(response.content)
    chat_history=chat_history[-20:] # 保存最近 10 轮对话
    