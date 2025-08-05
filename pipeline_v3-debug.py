import os
import logging
import dotenv
import traceback  # <-- Add this to capture full stack trace
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM  # Updated import

dotenv.load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

logging.info('=== Starting the data pipeline ===')

try:
    # 1. Load PDF files
    logging.info('Getting list of PDF files from "files" folder...')
    files_path = 'files'
    files = [os.path.join(files_path, file) for file in os.listdir(files_path) if file.endswith('.pdf')]
    logging.debug(f'Found {len(files)} PDF files: {files}')

    if not files:
        logging.error("No PDF files found in the 'files' directory.")
        raise ValueError("No PDF files found in the 'files' directory.")

    # 2. Split PDFs into chunks
    logging.info('Instantiating the token text splitter...')
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)

    logging.info('Splitting PDFs into chunks...')
    documents = []

    for file in files:
        try:
            logging.info(f"Loading and splitting PDF: {file}")
            pdf_loader = PyPDFLoader(file_path=file, extract_images=False)
            files_documents = pdf_loader.load_and_split(text_splitter=splitter)
            documents.extend(files_documents)
            logging.debug(f"Split {file} into {len(files_documents)} chunks")
        except Exception as e:
            logging.error(f"Error splitting PDF {file}: {e}\n{traceback.format_exc()}")  # Full traceback

    logging.debug(f'Total number of documents after splitting: {len(documents)}')

    if len(documents) == 0:
        logging.error("No documents created after PDF splitting.")
        raise ValueError("No documents created after PDF splitting.")

    # 3. Instantiate LLM
    logging.info('Loading Ollama LLM...')
    try:
        model_name = os.getenv('OLLAMA_MODEL', 'gemma:2b')
        llm = OllamaLLM(model=model_name, temperature=0.0)
        logging.info(f'Ollama model "{model_name}" loaded successfully.')
    except Exception as e:
        logging.error(f"Error loading Ollama model: {e}\n{traceback.format_exc()}")
        raise

    # 4. Setup prompts and parser
    logging.info('Setting up prompt templates and output parser...')
    from langchain_experimental.graph_transformers.llm import (
        SystemMessage, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate, JsonOutputParser
    )
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_core.pydantic_v1 import BaseModel, Field

    system_prompt = """
    You are an expert data scientist building a knowledge graph from technical news articles and industry reports.
    Your task is to extract key entities (companies, technologies, events, risks, outcomes) and their relationships.
    Output a JSON list where each object contains: head, head_type, relation, tail, tail_type.
    No hallucinations. Only verifiable relationships.
    """

    system_message = SystemMessage(content=system_prompt)

    class IndustryRelation(BaseModel):
        head: str = Field(description="Entity (Company, Technology, Event, Risk, Solution)")
        head_type: str = Field(description="Type of the head entity")
        relation: str = Field(description="Relationship between head and tail")
        tail: str = Field(description="Related entity")
        tail_type: str = Field(description="Type of the tail entity")

    parser = JsonOutputParser(pydantic_object=IndustryRelation)

    examples = [
        # (your examples unchanged)
    ]

    human_prompt = PromptTemplate(
        template="""
Examples:
{examples}

For the following article text, extract entities and relations in the specified JSON format.
{format_instructions}

Text: {input}
""",
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message_prompt])

    llm_transformer = LLMGraphTransformer(llm=llm, prompt=chat_prompt)

    # 5. Process documents safely
    graph_documents = []
    logging.info('Converting documents to graph documents...')

    for idx, doc in enumerate(tqdm(documents, desc="Converting documents")):
        try:
            logging.debug(f"Processing chunk {idx}/{len(documents)}: {doc.page_content[:100]}...")
            gd = llm_transformer.convert_to_graph_documents([doc])  # <-- pass Document, not string
            if gd:
                graph_documents.extend(gd)
                logging.info(f"Converted {len(gd)} graph documents from chunk")
            else:
                logging.warning(f"No graph documents created from chunk {idx}")
        except Exception as e:
            logging.error(f"Failed to process document chunk {idx}: {e}", exc_info=True)


    logging.debug(f"Total graph documents created: {len(graph_documents)}")
    if not graph_documents:
        logging.warning("No graph documents created. Skipping insertion to Neo4j.")

    # 6. Write into Neo4j
    logging.info('Connecting to Neo4j database...')
    from langchain_community.graphs import Neo4jGraph

    try:
        graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        logging.info("Successfully connected to Neo4j.")
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}\n{traceback.format_exc()}")
        raise

    if graph_documents:
        logging.info(f"Inserting {len(graph_documents)} graph documents into Neo4j...")
        try:
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            logging.info("Successfully inserted graph documents into Neo4j.")
        except Exception as e:
            logging.error(f"Failed to insert graph documents into Neo4j: {e}\n{traceback.format_exc()}")
            raise
    else:
        logging.warning("Skipped Neo4j insertion due to no graph documents.")

    logging.info('=== Data pipeline completed successfully! ===')

except Exception as e:
    logging.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
    raise
