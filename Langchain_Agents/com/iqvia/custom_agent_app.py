from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain import hub
## below 2 imports format_to_openai_tool_messages and OpenAIToolsAgentOutputParser helps to build custom agent
from langchain.agents.output_parsers.openai_tools import  OpenAIToolsAgentOutputParser

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GORQ_API_KEY"] = os.getenv("GORQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

is_save = False
vectordb = ""
retriever = ""
tools = []


llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.6)
print("llm :{}".format(llm))

################################
# WikipediaAPIWrapper          #
################################
wiki_api = WikipediaAPIWrapper(top_k_results=1)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)
tools.append(wiki)

################################
# ArxivAPIWrapper              #
################################
arxi_api = ArxivAPIWrapper(top_k_results=1,ARXIV_MAX_QUERY_LENGTH = 300,load_max_docs=1,doc_content_chars_max=200)
arxi = ArxivQueryRun(api_wrapper=arxi_api)
tools.append(arxi)

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
   tools.append(retriever_tool)

print("tools :{}".format(tools))

llm_with_tools = llm.bind_tools(tools)
################################
#  CUSTOM AGENT                #
################################
prompt_temp = ("As you are a helpful AI assistant, please answer user questions, if you are not able to answer use {tools} to see that you can provide any information"
               "query is : {input}")
prompt = PromptTemplate(
    input_variables=["tools","input"],
    template=prompt_temp
)
print(prompt)
agent = (
    {
      "tools":lambda x: x["tools"],
      "input":lambda x: x["input"]
    }
    |prompt|llm|OpenAIToolsAgentOutputParser()
)
agentExecutor = AgentExecutor(agent=agent,tools=tools,verbose=True)
resp = agentExecutor.invoke({"tools":tools,"input":"what is generative AI ?"})
print(resp["output"])




