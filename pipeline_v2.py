import os
import logging
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

logging.info('Starting the data pipeline...')

try:
    # 1. Load PDF files
    logging.info('Getting list of PDF files from "files" folder')
    files_path = 'files'
    files = [os.path.join(files_path, file) for file in os.listdir(files_path) if file.endswith('.pdf')]
    logging.info(f'List of PDF files: {files}')

    if not files:
        raise ValueError("No PDF files found in the 'files' directory.")

    # 2. Split PDFs into chunks
    logging.info('Instantiating the token text splitter')
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)

    logging.info('Splitting PDFs into chunks')
    from langchain_community.document_loaders import PyPDFLoader
    documents = []

    for file in files:
        pdf_loader = PyPDFLoader(file_path=file, extract_images=False)
        files_documents = pdf_loader.load_and_split(text_splitter=splitter)
        documents.extend(files_documents)
        logging.info(f'Split {file} into {len(files_documents)} chunks')

    # 3. Instantiate LLM
    logging.info('Loading Ollama LLM')
    from langchain_community.llms import Ollama
    llm = Ollama(model=os.getenv('OLLAMA_MODEL', 'gemma:2b'), temperature=0.0)

    # 4. Setup prompts and parser
    from langchain_experimental.graph_transformers.llm import SystemMessage, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate, JsonOutputParser
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
        {
            "text": "Structural Integrity partnered with PRCI to improve hard spot detection in pipelines.",
            "head": "Structural Integrity",
            "head_type": "Company",
            "relation": "partnered_with",
            "tail": "PRCI",
            "tail_type": "Organization"
        },
        {
            "text": "Ultrasonic pipe cleaning was evaluated for reducing maintenance costs in nuclear plants.",
            "head": "Ultrasonic Pipe Cleaning",
            "head_type": "Technology",
            "relation": "reduces",
            "tail": "Maintenance Costs",
            "tail_type": "Risk"
        },
        {
            "text": "SC-SASSI software enhanced seismic analysis for nuclear facilities.",
            "head": "SC-SASSI",
            "head_type": "Technology",
            "relation": "enhances",
            "tail": "Seismic Analysis",
            "tail_type": "Capability"
        }
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
    logging.info('Converting documents to graph documents')

    for doc in tqdm(documents, desc="Converting documents"):
        try:
            gd = llm_transformer.convert_to_graph_documents([doc])
            if gd:
                graph_documents.extend(gd)
        except Exception as e:
            logging.error(f"Failed to process document chunk: {e}")

    # Save checkpoint
    logging.info(f"Saving checkpoint: {len(graph_documents)} graph documents generated")
    with open("graph_documents_checkpoint.txt", "w") as f:
        f.write(str(len(graph_documents)))

    if not graph_documents:
        raise ValueError("No graph documents created. Aborting pipeline.")

    # 6. Write into Neo4j
    logging.info('Connecting to Neo4j and persisting graph documents')
    from langchain_community.graphs import Neo4jGraph

    graph = Neo4jGraph(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD')
    )

    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    logging.info('Data pipeline completed successfully!')

except Exception as e:
    logging.error(f"Pipeline failed: {e}")
