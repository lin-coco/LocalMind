# 本地知识库问答助手

## 项目简介

轻量级AI知识库问答助手 | 基于LangChain的首个RAG项目实践

## 快速开始

```bash
# 克隆项目
git clone https://github.com/lin-coco/LocalMind
# 安装依赖
conda create --name chatgpt python=3.9 -y
conda activate LocalMind
pip install -r requirements.txt
# 配置环境变量
vim .env
# 启动应用
python app.py
```

.env文件配置环境变量

```bash
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BAILIAN_API_KEY=******
BAILIAN_MODEL=qwen-max-2025-01-25
LOAD_PATH=/Users/lincoco/Desktop/2025学习计划/LocalMind/knowledge
PERSIST_PATH=/Users/lincoco/Desktop/2025学习计划/LocalMind/vectorstore
MODELSCOPE_MODEL_ID=iic/nlp_gte_sentence-embedding_chinese-large
MODELSCOPE_MODEL_REVISION=v1.1.0
```

## AI服务

1. [阿里云百炼](https://dashscope.aliyuncs.com)：提供大语言模型调用接口，新出的大语言模型会给免费的token次数
2. [魔塔（ModelScope）](https://www.modelscope.cn)：本地下载开源大语言模型、开源向量模型，有适配langchain调用的接口；开源数据集可以做RAG应用
3. [Hugging Face](https://huggingface.co)：类似魔塔社区，这是海外的

## 开发收获

学会LangChain开发的流程，掌握文档分块→向量化→存储→检索的流程设计

1. 加载文档，每个文件对应类型为`Document`：
   - `langchain_community.document_loaders`库基本上有所有类型文档的加载器
   - `langchain_unstructured`的`UnstructuredLoader`，是一个比较通用的加载器，对于音频类文件需要调用api
2. 分割文档，每个`Document`文件被分割成多个`Document`块，限制每个`Document`的大小，提高之后的检索精确度
   - `RecursiveCharacterTextSplitter`：递归的分割器，每种格式的语言对应的`separators`不一样
     - 英文：`separators=["\n\n", "\n", " ", ""]`对应的就是段落、句子、单词分割
     - 中文+英文：`separators=['\n','\n','。', '.', '！','!','？','?','，', ',', ';',' ']`
   - `HTMLHeaderTextSplitter`：按照HTML头部进行分割，主要是给分割后的内容加上了metadata描述属于哪一个h1等
   - `MarkdownHeaderTextSplitter`：按照Markdown标题进行分割，主要是给分割后的内容加上了metadata描述属于哪一个标题
3. 向量化文档，会用向量模型给每个document向量化成毅哥n维向量
   - 向量模型：从`ModelScope`会在`HuggingFace`下载向量模型
   - 使用`ModelScopeEmbeddings`或者`HuggingFaceEmbeddings`调用向量模型将所有的`documents`向量化
4. 向量数据库，将向量化后的数据直接存入类似`Chroma`这样的向量数据库
5. 检索器，`VectorStoreRetriever`，将用户问题向量化，寻找出向量数据库中多个相似向量，这些就是和用户问题相关的内容
6. 提示词工程，`prompts`将用户问题、相似性内容、系统指令等组成提示词交给LLM模型
   - ChatPromptTemplate.from_messages：拼接多个Prompt
   - HumanMessagePromptTemplate：通常是用户的输入
   - SystemMessagePromptTemplate：通常是系统指令的输入
   - MessagesPlaceholder：存放历史问答记录
7. LLM，大语言模型，通过base_url、api_key、model_name等信息调用模型

## LCEL

**概述**：LangChain的核心是Chain，即对多个组件的组合和一系列调用，LCEL是LangChain的表达式语言，是一种高效简介的调用一系列组件的方式。

**使用方式**：LCEL的使用方式跟Linux中的管道类似，例如：`chain = prompt_tpl | model | output_parser`

**实现原理**：

1. Runable接口定义了`invoke`方法，并重写了`__or__`特殊函数
2. 所有的组件都实现了Runnable接口，例如LLM、prompts、output_parser

所以两个组件进行或运算的时候，就是前一个组件invoke的输出作为后一个组件invoke的输入

**隐式适配**：LangChain会自动将字符串、dict等原始类型包装为`RunnableLambda`，使类似`"Hello" | llm`的写法合法。

1. 字符串 -> RunnableLambda
2. 字典 -> RunnableParallel
3. 列表 -> RunnableSequence（就是一个Chain）
4. 函数 -> RunnableLambda



