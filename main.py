from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time

def get_dirloaders(source_dir: str) -> list[DirectoryLoader]:
    """
    返回所有支持的文档加载器: pdf、docx、md、txt、html
    """
    dirloaders = [
        # 分别加载不同格式 .pdf
        DirectoryLoader(
            path=source_dir,
            glob=['**/*.pdf'], # 指定读取文件的格式
            show_progress=True, # 显示加载进度
            use_multithreading=True, # 显示多线程
            silent_errors=True, # 错误时不抛出异常，直接忽略该文件
            loader_cls=PyPDFLoader, # 指定加载器
        ),
        # 分别加载不同格式 .docx
        DirectoryLoader(
            path=source_dir,
            glob=['**/*.docx'], # 指定读取文件的格式
            show_progress=True, # 显示加载进度
            use_multithreading=True, # 显示多线程
            silent_errors=True, # 错误时不抛出异常，直接忽略该文件
            loader_cls=Docx2txtLoader, # 指定加载器
        ),
        # 分别加载不同格式 .md
        DirectoryLoader(
            path=source_dir, # 指定路径
            glob=['**/*.md'], # 指定读取文件的格式
            show_progress=True, # 显示加载进度
            use_multithreading=True, # 显示多线程
            silent_errors=True, # 错误时不抛出异常，直接忽略该文件
            loader_cls=TextLoader, # 指定加载器
            loader_kwargs={'autodetect_encoding': True}
        ),
        # 分别加载不同格式 .txt
        DirectoryLoader(
            path=source_dir, # 指定路径
            glob=['**/*.txt'], # 指定读取文件的格式
            show_progress=True, # 显示加载进度
            use_multithreading=True, # 显示多线程
            silent_errors=True, # 错误时不抛出异常，直接忽略该文件
            loader_cls=TextLoader, # 指定加载器
            loader_kwargs={'autodetect_encoding': True}
        ),
        # 分割加载不同格式 .html
        DirectoryLoader(
            path=source_dir, # 指定路径
            glob=['**/*.html'], # 指定读取文件的格式
            show_progress=True, # 显示加载进度
            use_multithreading=True, # 显示多线程
            silent_errors=True, #错误时不抛出异常，直接忽略文件
            loader_cls=BSHTMLLoader, # 指定加载器
            loader_kwargs={'bs_kwargs': {'features': 'html.parser'}}
        )
    ]
    return dirloaders

def load_documents(source_dir: str) -> list[Document]:
    """
    加载指定目录下的所有文档、.pdf、.md、.txt、.docx、.html
    """

    print(f"开始加载{source_dir}路径，支持的格式: .pdf、.md、.txt、.docx、.html")
    dirloaders = get_dirloaders(source_dir=source_dir)
    
    # 解析并合并文档列表
    documents: list[Document] = []
    for loader in dirloaders:
        documents.extend(loader.load())
    print(f'成功加载路径下{len(documents)}个文档')
    for doc in documents:
        print(f"文档路径:{doc.metadata['source']}, 内容预览:" + doc.page_content[:20].replace('\n', ' ') + "...")
    return documents

def split_documents(documents: list[Document], chunk_size: int = 100, chunk_overlap: int = 20) -> list[Document]:
    print("开始分割文档，使用RecursiveCharacterTextSplitter分割器")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n','\n','。', '.', '！','!','？','?','，', ',', ';',' '],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # 保留原始文档中的位置信息
    )
    origin_len = len(documents)
    documents = text_splitter.split_documents(documents)
    print(f"原始文档块数:{origin_len}, 分割后文档块数:{len(documents)}")
    for doc in documents:
        print(f"文档路径:{doc.metadata['source']}, 分割内容预览:" + doc.page_content[:20].replace('\n', ' ') + "...")
    return documents

def embedding_documents(documents: list[Document],persist_directory: str,embedding_kwargs: any) -> Chroma:
    print(f"开始向量化数据，存储在{persist_directory}")
    start_time = time.time()
    embedding = ModelScopeEmbeddings(**embedding_kwargs)
    chroma = Chroma.from_documents(collection_name='knowledge',documents=documents, embedding=embedding,persist_directory=persist_directory)
    print(f"完成向量化存储文档数据，执行耗时{time.time()-start_time:.2f}秒")
    return chroma

def get_chroma(persist_directory: str,embedding_kwargs: any) -> Chroma:
    chroma = Chroma(collection_name='knowledge', persist_directory=persist_directory,embedding_function=ModelScopeEmbeddings(**embedding_kwargs))
    return chroma

def create_retrievers(chroma: Chroma) -> VectorStoreRetriever:
    return chroma.as_retriever(
        search_type='mmr', # 平衡相似性与多样性
        search_kwargs={
            "k": 5, # 返回文档数量
            "fetch_k": 10, # 检索通过mmr算法文档数量
            "lambda_mult": 0.5,  # 多样性控制参数（0-1，越大越多样）
            "score_threshold": 0.3,  # 相关性阈值
        }
    )

def create_prompt() -> ChatPromptTemplate:
    system_prompt_templete = """
你是一个智能知识库问答助手，请基于用户问题与以下知识库片段（可能包含多格式内容），用专业且易理解的格式回答：
[向量数据库检索结果]
{context}
------------------
请按以下规则处理：
1. **内容解析**：
   - 若包含技术参数表格，用「对比表格」呈现关键指标
   - 若涉及操作步骤，用「数字编号列表」结构化展示
   - 图文内容需说明示意图核心逻辑
2. **回答要求**：
   - 中文回答优先，专业术语保留英文原文
   - 区分客观事实（来自知识库）与推理建议
   - 技术文档需标注版本/更新时间（若存在）
   - 书籍引用需标注章节/页码位置
3. **多源验证**：
   - 当多个文档存在冲突信息时：
     ① 优先采信PDF/纸质文档内容
     ② 其次参考内部技术文档
     ③ 最后考虑网页版内容
   - 用[⚠️数据冲突]标记并提示复核建议
4. **边界处理**：
   - 知识库未覆盖的问题直接说明能力边界
   - 模糊查询时提供相关主题的知识图谱路径
   - 涉及敏感数据用[权限不足]代替具体内容
5. **打招呼或者询问你的信息**
   - 你只知道自己是超级牛逼的智能知识库问答助手
   - 如果用户只是在打招呼，你要亲切的回复
"""
    human_prompt_template = """
当前问题：「{question}」
"""
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        SystemMessagePromptTemplate.from_template(system_prompt_templete),
        HumanMessagePromptTemplate.from_template(human_prompt_template),
    ])
    return prompt

def create_llm(base_url: str, api_key: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(base_url=base_url,api_key=api_key,model=model,streaming=True,callbacks=[StreamingStdOutCallbackHandler()])


if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    load_dotenv()
    # 百炼模型
    BAILIAN_MODEL = os.getenv('BAILIAN_MODEL')
    # 百炼url
    BAILIAN_BASE_URL = os.getenv('BAILIAN_BASE_URL')
    # 百炼api_ley
    BAILIAN_API_KEY = os.getenv('BAILIAN_API_KEY')
    # 数据源文件夹
    LOAD_PATH=os.getenv('LOAD_PATH')
    # 向量持久化
    PERSIST_PATH=os.getenv('PERSIST_PATH')
    # 嵌入参数（使用ModelScopeEmbeddings，下载modelscope向量模型使用）
    EMBEDDING_KWARGS={
        'model_id': os.getenv('MODELSCOPE_MODEL_ID'),
        'model_revision': os.getenv('MODELSCOPE_MODEL_REVISION'),
    }
    import warnings
    warnings.filterwarnings("ignore", message="The `device` argument is deprecated")
    
    # 1. 加载本地知识库
    chroma = get_chroma(persist_directory=PERSIST_PATH, embedding_kwargs=EMBEDDING_KWARGS)
    collection = chroma.get()
    doc_count = len(collection['ids'])
    if not doc_count:
        # 没有加载过本地知识库，加载一遍
        print("没有加载过知识库，开始加载......")
        # 1. 加载本地知识库文档
        documents = load_documents(LOAD_PATH)
        # 2. 分割文档数据
        documents = split_documents(documents)
        # 3. 数据向量化存储
        chroma = embedding_documents(documents, PERSIST_PATH, EMBEDDING_KWARGS)
    # 4. 创建检索器
    chroma = Chroma(collection_name='knowledge', persist_directory=PERSIST_PATH,embedding_function=ModelScopeEmbeddings(**EMBEDDING_KWARGS))
    retriever = create_retrievers(chroma=chroma)
    # 5. 创建prompt
    prompt = create_prompt()
    prompt
    # 6. 创建llm
    llm = create_llm(BAILIAN_BASE_URL, BAILIAN_API_KEY, BAILIAN_MODEL)
    # 7. 创建memory，存储历史对话
    memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history')
    # 7. 构建检索chain
    def print_info(x):
        print("追踪信息", x)
        return x
    chain = (
        {
            'context': retriever, 
            'question': RunnablePassthrough(), 
            'chat_history': lambda x: memory.load_memory_variables({})['chat_history']
        }
        | prompt
        | llm
    )
    # 8. 循环执行链
    while True:
        question = input('\n问题：').strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        print("回答：", end='',flush=True)
        answer = chain.invoke(question)
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        



