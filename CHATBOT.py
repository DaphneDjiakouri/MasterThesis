# To run it in the terminal: chainlit run CHATBOT.py -w

import os
import openai
import dotenv
#from openai import OpenAI
import sys
sys.path.append('../..')

# import dotenv
dotenv.load_dotenv()
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.getenv("OPENAI_API_KEY") 


#pip install pypdf
#export HNSWLIB_NO_NATIVE = 1

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse




text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
        loader = Loader(file.path)
        pages = loader.load_and_split()
        docs = text_splitter.split_documents(pages)
        return docs
    
def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch
    
@cl.on_chat_start
async def start():

    # Sending an image with the local file path
    await cl.Message(content="You can now chat with your pdfs.").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    vectordb = await cl.make_async(get_docsearch)(file)

    

    # Adding memory
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )

    # create the chain to answer questions 
    from langchain.chains import ConversationalRetrievalChain
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
        chain_type="stuff", 
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    
    cl.user_session.set("qa_chain", qa_chain)

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()


@cl.on_message
async def handle_message_and_save_content(message: cl.Message):
    # Save the message content to a text file
    with open("input_messages.txt", "a", encoding="utf-8") as file:
        file.write(message.content + "\n")
    
    qa_chain = cl.user_session.get("qa_chain")
    llm_response = await qa_chain.acall(message.content, callbacks = [cl.AsyncLangchainCallbackHandler()])
    resp = llm_response['answer']
    resp = resp + '\n\nYou can find information at:'
    
    for source in llm_response["source_documents"]:
        resp = resp + '\n page: ' + str(source.metadata['page'] + 1)
    
    # Send the compiled response back to the user
    await cl.Message(content=resp).send()    
