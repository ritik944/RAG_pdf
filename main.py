from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm=ChatOpenAI()

system_template = """Use the following pieces of context to answer the users question.
{context}

Begin!
----------------
Question: {question}
Helpful Answer:"""



messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)

loader = PyPDFLoader("data/data_containg_pdf_file/awsgsg_intro.pdf")
data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 100)
# docs = text_splitter.split_documents(data)


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 0)
texts = text_splitter.split_text(data[1].page_content)
docs = [Document(page_content=t) for t in texts[:]]

print(docs,"\n")
embedding=OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=docs,embedding=embedding)

question = "host a website"

qa_chain = RetrievalQA.from_chain_type(
    llm,retriever=vectorstore.as_retriever(),
    chain_type_kwargs = {"prompt": prompt})
result = qa_chain({"query": question})

print(result)