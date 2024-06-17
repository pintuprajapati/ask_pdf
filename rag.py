from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
# from sentence_transformers import SentenceTransformer

import os, json, traceback
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

os.getenv('OPENAI_API_KEY')

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


### Load document directory ###
# https://python.langchain.com/docs/modules/data_connection/document_loaders/json
def load_docs(content_directory):
    loader = PyPDFLoader(content_directory)
    # print('➡ loader:', loader)

    #Load the document by calling loader.load()
    pages = loader.load()
    # print('➡ pages:', pages)

    print(len(pages))
    # print("---------", pages[0].page_content[0:500])

    # print("=====", pages[0].metadata)
    # {'source': 'book.pdf', 'page': 0}
    return pages

### split document using the RecursiveCharacterTextSplitter from Langchain. ###
def split_docs(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#   c_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

### Persistence in Chroma DB ###
def store_vector_db(docs, embeddings, persist_directory):
    Chroma()
    vectordb = Chroma.from_documents(
      documents=docs, embedding=embeddings, persist_directory=persist_directory
    )

    print(vectordb._collection.count())
    vectordb.persist()
    print("Vector DB is stored into directory")
    return True

# call this function when you want to create a loca db of vector db
def create_persist_dir(content_directory, embeddings, persist_directory):
  try:
    # load the doc
    documents = load_docs(content_directory)
    
    chunk_size=1000
    chunk_overlap=20

    # split the doc
    docs = split_docs(documents, chunk_size, chunk_overlap)
    
    # # save the  data to the local dir
    store_vector_db(docs, embeddings, persist_directory) 
    
    return True
  except Exception as e:
    print('➡ Error in create persist dir: ', e)
    print("\nError Traceback: ", traceback.print_exc(e))  

# get the answer from already created vectordb (stored locally)
def get_answer_from_vectordb(query, persist_directory, embeddings):
  os.environ["OPENAI_API_KEY"]
  openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo")

  chain = load_qa_chain(llm=openai_llm, chain_type="stuff", verbose=False)


  #NOTE: if you create persist db using some "embeddings" then while fetching the answer from vector db (persist dir), you should use the same "embeddings" otherwise you will get "dimentions error such as: InvalidDimensionException: Dimensionality of (1536) does not match index dimensionality (384)"

  # create an instnace of vector db
  new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

  matching_docs = new_db.similarity_search(query)
  answer = chain.invoke({"input_documents": matching_docs, "question": query})
  print('\ninput text: ', query)
  print('\noutput text: ', answer['output_text'])

############### Query the document based on user query using OpenAI LLM ###############
content_directory = "book.pdf" # your input data (it can be document, txt, pdf, json etc...)

# model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
persist_directory = "docs/chroma/huggingface_embedding" # To store/persist the vectors into local dir
            
# uncomment below two lines if you wanna use OPENAI Embeddings
# embeddings = OpenAIEmbeddings()
# persist_directory = "docs/chroma/openai_embedding" # To store/persist the vectors into local dir

# NOTE: Create persist dir of vectors only when your input data/document is updated with new data/information 
# OR you are creating it for the first time. OR you are changing the embeddings model. Otherwise no need to create it (if created already)
# create_persist_dir(content_directory, embeddings, persist_directory)

# user query (ask any question from your document)
query = "what's the title of the book?"

# This will return the answer of your query
get_answer_from_vectordb(query, persist_directory, embeddings)


