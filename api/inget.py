import os
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere.embeddings import CohereEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
import jq

# Load environment variables from .env file
# load_dotenv('.env')

# # Access your API key
# GOOGLE_API = os.getenv("GOOGLE_API")
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")

GOOGLE_API = os.environ.get("GOOGLE_API")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")


def load_data():
    """
    Load data from a file using the provided file path and JSON schema.

    Args:
        None

    Returns:
        list: A list of Document objects created from the loaded data.

    Raises:
        FileNotFoundError: If the file specified in `file_path` does not exist.
        Exception: If an unexpected error occurs during the loading process.
    """
    try:
        # Set the file path and JSON schema
        file_path = "data/FAQ.json"
        jq_schema = '.[]'

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Create a JSONLoader instance with the specified file path, JSON schema, and text content flag
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=jq_schema,
            text_content=False
        )

        # Load the data using the JSONLoader instance
        data = loader.load()
        # Create Document objects from the loaded data
        # data = [Document(page_content=doc.page_content) for doc in data]

        return data

    except FileNotFoundError as e:
        # Print an error message if the file does not exist
        print(f"File error: {e}")
    except Exception as e:
        # Print an error message for any unexpected errors
        print(f"An unexpected error occurred: {e}")


@st.cache_resource
def create_embeddings(model_name):
    """
    Creates an instance of `HuggingFaceEmbeddings` with the specified `model_name` and caches it for future use.

    Args:
        model_name (str): The name of the Hugging Face model to use for embedding creation.

    Returns:
        HuggingFaceEmbeddings: The cached instance of `HuggingFaceEmbeddings` with the specified `model_name`.

    """
    # Initialize Hugging Face Embeddings with the specified model name
    # This instance is cached for future use within the Streamlit session
    embeddings = CohereEmbeddings(
        model=model_name, cohere_api_key=COHERE_API_KEY)

    # Return the cached instance of Hugging Face Embeddings
    return embeddings

# def create_embeddings(model_name, hf_api_key, texts):
#     # Initialize Hugging Face Inference API embeddings with LangChain
#     embeddings = HuggingFaceInferenceAPIEmbeddings(
#         api_key=hf_api_key,
#         model_name=f"sentence-transformers/{model_name}"
#     )
#     # Generate embeddings for each text
#     results = [embeddings.embed_query(text) for text in texts]
#     return results


@st.cache_resource
def create_chroma_db(persist_directory, _embeddings):
    """
    Creates a Chroma database using the given persist directory and embedding function.

    If a directory named 'vectorstore' exists in the current working directory, it loads the database from the directory.
    Otherwise, it loads the data from a JSON file, creates the database from the documents, and persists it to the specified directory.

    Args:
        persist_directory (str): The directory where the database will be persisted.
        _embeddings (HuggingFaceEmbeddings): The embedding function to use.

    Returns:
        Chroma: The created Chroma database.
    """

    # Check if the 'vectorstore' directory exists
    if os.path.isdir(os.getcwd() + '/vectorstore'):
        # Load the database from the directory
        db = Chroma(persist_directory=persist_directory,
                    embedding_function=_embeddings)
    else:
        # Load the data from a JSON file
        data = load_data()
        # Create the database from the documents
        db = Chroma.from_documents(
            data, _embeddings, persist_directory=persist_directory)

    return db.as_retriever()


# @st.cache_resource
def create_pinecone_db(data, _embeddings):
    """
    Creates a Pinecone index from the given data using the provided embeddings.

    Args:
        data (List[str]): A list of documents to be indexed.
        _embeddings (PineconeEmbeddings): The embeddings to be used for indexing.

    Returns:
        PineconeIndex: The created Pinecone index.

    Raises:
        PineconeError: If there is an error creating the index.

    Note:
        The dimension of the index is assumed to be 1536, which is the dimension of the embeddings.
        The repository used is DENSE.
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)  # Initialize Pinecone client

    # Create a Pinecone index from documents
    index_name = 'pinecone'  # Name of the index
    index = pc.create_index(
        name=index_name,  # Name of the index
        service_type=ServiceType.SIMILARITY,  # Service type for similarity search
        # Dimension of the index (assuming embeddings is a numpy array)
        dimension=1536,
        repository=Repository.DENSE,  # Repository type
        data=data  # Data to be indexed
    )

    return index


def create_llm(model, temperature, max_tokens, top_p, GOOGLE_API):
    """
    Creates a ChatGoogleGenerativeAI object with the given parameters.

    Args:
        model (str): The name of the model to use for the ChatGoogleGenerativeAI.
        temperature (float): The temperature parameter for the ChatGoogleGenerativeAI.
        max_tokens (int): The maximum number of tokens to generate for the ChatGoogleGenerativeAI.
        top_p (float): The top-p parameter for the ChatGoogleGenerativeAI.
        GOOGLE_API (str): The API key for the ChatGoogleGenerativeAI.

    Returns:
        ChatGoogleGenerativeAI: The created ChatGoogleGenerativeAI object.

    Note:
        The ChatGoogleGenerativeAI is a language model that can generate human-like text based on a prompt.
        The parameters control the style and length of the generated text.
    """
    # Create a ChatGoogleGenerativeAI object with the given parameters
    llm = ChatGoogleGenerativeAI(
        model=model,  # The name of the model to use
        temperature=temperature,  # The temperature parameter for the model
        max_tokens=max_tokens,  # The maximum number of tokens to generate
        top_p=top_p,  # The top-p parameter for the model
        google_api_key=GOOGLE_API  # The API key for the model
    )
    return llm


def create_prompt(system_prompt):
    """
    Creates a chat prompt template using the given system prompt.

    Args:
        system_prompt (str): The system prompt to be used in the chat prompt template.

    Returns:
        ChatPromptTemplate: The created chat prompt template.

    Note:
        The ChatPromptTemplate is a class that represents a chat prompt template.
        It consists of a system prompt and a human prompt.
        The system prompt is used to provide context and instructions to the model.
        The human prompt is used to prompt the model to generate a response.
    """
    # Create a chat prompt template with the given system prompt and human prompt
    # The system prompt is used to provide context and instructions to the model
    # The human prompt is used to prompt the model to generate a response
    prompt = ChatPromptTemplate.from_messages(
        [
            # The system prompt is used to provide context and instructions to the model
            ("system", system_prompt),
            # The human prompt is used to prompt the model to generate a response
            ("human", "{Question}"),
        ]
    )
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, prompt, llm):
    """
    Creates a RAG chain using the specified retriever, prompt, and llm objects.
    Args:
        retriever: The retriever object used in the RAG chain.
        prompt: The prompt object used in the RAG chain.
        llm: The llm object used in the RAG chain.
    Returns:
        The created RAG chain.
    """
    rag_chain = (
        {"context": retriever | format_docs, "Question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def create_correction_prompt_template(query):
    """
    Creates a correction prompt template for a given user query.

    Args:
        query (str): The user query to be corrected.

    Returns:
        PromptTemplate: The created correction prompt template.

    The correction prompt template is a PromptTemplate object that prompts the user to correct the given query as a standalone question for grammar and spelling errors. If there are no errors, the same query is returned.

    Example:
        >>> create_correction_prompt_template("What is the weather today?")
        <PromptTemplate: Please correct the following user query as a standalone question for grammar and spelling if there are any errors. Otherwise, return the same query: {query}>

    """
    template = PromptTemplate(
        input_variables=["query"],
        template="Please correct the following user query as a standalone question for grammar and spelling if there are any errors. Otherwise, return the same query: `{query}`",
    )
    return template.format(query=query)


async def process_query(query):
    """
    Process a user query by invoking a correction prompt template to check for grammar and spelling errors. If there are no errors, the same query is returned. 

    Args:
        query (str): The user query to be processed.

    Returns:
        str: The processed query with any grammar and spelling errors corrected.
    """

    GOOGLE_API = os.getenv("GOOGLE_API")
    model = "gemini-1.5-flash"
    temperature = 0.7
    max_tokens = 100
    top_p = 0.9

    system_prompt = (
        """
        
    **Primary Goal:** Provide concise and accurate answers to frequent customer questions about the ecommerce website, mimicking human-like responses without offering extra information, solely based on the provided context.
    if the user's question can be answered using the provided context, the chatbot should answer the user question. if some details are present in the context then the chatbot should answer only those details.
    Response Prompt Template:

    Provide a brief and direct answer to the user's question, only using the provided context

    **Evaluation Criteria:**

    1. Accuracy: The chatbot's response accurately addresses the user's question, based on the provided context.
    2. Conciseness: The chatbot's response is brief and to the point, without unnecessary additional information.
    3. Human-like tone: The chatbot's response simulates a human customer support agent's tone and language.
    4. Contextual Relevance: The chatbot's response only uses the provided context and does not provide extra information.

    **Additional Response:**

    - If the user's question is irrelevent to customers questions for ecommerce website, the chatbot should respond with: "I'm not familiar with that question."
    - If the user's question does not match the provided context, the chatbot should respond with:
    "I'm not familiar with that question."
    \n\n
        {context}
        """
    )

    retriever = create_chroma_db(
        "/vectorstore", create_embeddings("embed-english-light-v3.0"))
    llm = create_llm(model, temperature, max_tokens, top_p, GOOGLE_API)
    prompt = create_prompt(system_prompt)
    rag_chain = create_rag_chain(retriever, prompt, llm)
    correction_prompt_template = create_correction_prompt_template(query)

    standalone_question = llm.invoke(correction_prompt_template).content
    result = rag_chain.invoke(standalone_question)

    return result
