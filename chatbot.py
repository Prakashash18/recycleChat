import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


API_KEY = st.secrets["openai_secret_key"]

# Define URLs for recycling information
urls = [
    
    'https://www.shell.com.sg/motorists/promotions/e-waste-recycling.html',
    'https://recyclopedia.sg/resources/metalo-non-regulated-e-waste-bins',
    'https://www.towardszerowaste.gov.sg/recycle/what-to-recycle/',
    'https://www.nea.gov.sg/our-services/waste-management/3r-programmes-and-resources/e-waste-management/where-to-recycle-e-waste', #ewaste knowledge
    'https://www.nea.gov.sg/our-services/waste-management/donation-resale-and-repair-channels',
    'https://www.nea.gov.sg/our-services/waste-management/3r-programmes-and-resources/waste-minimisation-and-recycling',
    'https://www.nea.gov.sg/our-services/waste-management/beverage-container-return-scheme',
    'https://www.nea.gov.sg/our-services/waste-management/3r-programmes-and-resources/food-waste-management/food-waste-management-strategies',
    'https://www.nea.gov.sg/our-services/waste-management/reverse-vending-machines',
    'https://www.kgs.com.sg/blog/where-can-i-throw-my-old-computers-and-laptops-in-singapore/',
    # 'https://alba-ewaste.sg/types-e-waste/#regulated-consumer',
    # 'https://alba-ewaste.sg/drop-off-locations/',
]

# Create a WebBaseLoader to fetch data from URLs
loader = WebBaseLoader(urls)
data = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1000)
all_splits = text_splitter.split_documents(data)

# Create embeddings and vectorstore
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=API_KEY))

# Create a ChatOpenAI instance
llm = ChatOpenAI(streaming=True, openai_api_key=API_KEY, callbacks=[StreamingStdOutCallbackHandler()])

# Create a ConversationSummaryMemory
memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)

# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever()

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. If you do not know the answer reply with 'I am sorry'.
Your key role is to assist the user by providing recycling information. You are an expert in identifying the correct recycling bins and locations for various items. You are also given data to refer to via the text embeddings. 
Once you find the correct recycling location in singapore, you can ask the user if he wants to find a list of locations based on area such as north,east, west, central. 

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


# Create a ConversationalRetrievalChain with the language model, retriever, and memory
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, condense_question_prompt=CONDENSE_QUESTION_PROMPT, memory=memory)

st.title("Recycling Information Chat")


# Function to handle user input and generate responses
def chatbot(user_input):

    # Get a response from the chatbot
    if len(st.session_state['history']) > 0:
        response = qa({"question": user_input, "chat_history": st.session_state['history']})
    else:
        response = qa({"question": user_input})
    
    st.session_state['history'].append((user_input, response["answer"]))

    print(response)     
    st.session_state['generated'].append(response['answer'])
    return response['answer']


# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me " + "about recycling" + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Where to recycle .......", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = chatbot(user_input)
        st.session_state['past'].append(user_input)
        #st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
