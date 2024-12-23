import dotenv
import streamlit as st
from langchain_community.llms.openai import OpenAIChat
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentType, initialize_agent, load_tools, AgentExecutor, create_openai_tools_agent
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain import hub

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GORQ_API_KEY"] = os.getenv("GORQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

is_save = False
vectordb = ""
retriever = ""
tool = []


llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.6)
print("llm :{}".format(llm))

################################
# WikipediaAPIWrapper          #
################################
wiki_api = WikipediaAPIWrapper(top_k_results=1)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)
tool.append(wiki)

################################
# ArxivAPIWrapper              #
################################
arxi_api = ArxivAPIWrapper(top_k_results=1,ARXIV_MAX_QUERY_LENGTH = 300,load_max_docs=1,doc_content_chars_max=200)
arxi = ArxivQueryRun(api_wrapper=arxi_api)
tool.append(arxi)

################################
#  LOAD DOCUMENTS LOADER       #
################################

loader = TextLoader("./speech.txt")
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10)
docs=text_splitter.split_documents(document)

if is_save:
    try:
        vectordb=FAISS.from_documents(docs,OpenAIEmbeddings())
        vectordb.save_local("C:\\Users\\ajayk\\Desktop\\FAISS_DB_SAVE")
        retriever = vectordb.as_retriever()
        print("vectordb stored into local machine....")
    except:
        print("An exception occurred while saving vector store db")
else:
   print("load vectordb from local machine....")
   try:
        vectordb = FAISS.load_local("C:\\Users\\ajayk\\Desktop\\FAISS_DB_SAVE\\",embeddings=OpenAIEmbeddings(),allow_dangerous_deserialization=True);
        retriever = vectordb.as_retriever()
   except:
       print("An exception occurred while loading vector store db...")

if retriever:
   retriever_tool = create_retriever_tool(retriever=retriever,name="vector_db_search",description="search in local vector store db..")
   tool.append(retriever_tool)

print("tool :{}".format(tool))


################################
#  AGENT and AGENT EXECUTOR    #
################################
prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt)
agent = create_openai_tools_agent(llm=llm,tools=tool,prompt=prompt)
#
agentExecutor = AgentExecutor(agent=agent,tools=tool,verbose=True)

# resp = agentExecutor.invoke({"input":"what is machine learning ?"}) // go to wikipedia
# resp = agentExecutor.invoke({"input":"what is 1706.03762 all about ?"}) // go to arxiv
# resp = agentExecutor.invoke({"input":"The world must be made safe for democracy. Its peace must be planted upon the tested foundations of political liberty."}) // go to vector db
resp = agentExecutor.invoke({"input":"what is 1706.03762 all about ?"})
print("response :{}".format(resp))




