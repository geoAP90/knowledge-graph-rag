import os
import logging
import dotenv
from tqdm import tqdm
from neo4j import GraphDatabase
import json

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
    You are an expert data scientist building a knowledge graph from technical articles, reports, and related documents.
    Your task is to extract key entities (authors, article names, topics, subjects, keywords, and their relationships).
    Articles can have multiple authors, discuss multiple topics, and topics can be correlated across multiple articles. 
    Extract relationships as follows:
    - Author -> wrote -> Article
    - Article -> discusses -> Topic
    - Article -> belongs_to -> Subject
    - Article -> mentions -> Keyword
    - Topic -> relates_to -> Topic
    - Author -> cites -> Author
    - Article -> related_to -> Article (for correlated articles)
    Ensure that relationships between entities can be complex, such as multiple authors for an article or multiple topics discussed by an article.
    Output a JSON list where each object contains: head, head_type, relation, tail, tail_type.
    Only verifiable relationships, no hallucinations.
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
            "text": "Mark W. Marano authored the CEO message in the 2024 edition of News & Views.",
            "head": "Mark W. Marano",
            "head_type": "Author",
            "relation": "wrote",
            "tail": "CEO message in the 2024 edition of News & Views",
            "tail_type": "Article"
        },
        {
            "text": "The article 'Ultrasonic Pipe Cleaning in Nuclear Power Plants' discusses the feasibility and benefits of using ultrasonic vibration waveforms.",
            "head": "Ultrasonic Pipe Cleaning in Nuclear Power Plants",
            "head_type": "Article",
            "relation": "discusses",
            "tail": "feasibility and benefits of using ultrasonic vibration waveforms",
            "tail_type": "Topic"
        },
        {
            "text": "'Combustion Turbine Compressor Hygiene' is categorized under the broader subject of Power Plant Asset Management.",
            "head": "Combustion Turbine Compressor Hygiene",
            "head_type": "Article",
            "relation": "belongs_to",
            "tail": "Power Plant Asset Management",
            "tail_type": "Subject"
        },
        {
            "text": "'Advanced NDE for Hydroelectric Penstock Inspection' mentions the keywords 'non-destructive evaluation' and 'hydroelectric penstocks'.",
            "head": "Advanced NDE for Hydroelectric Penstock Inspection",
            "head_type": "Article",
            "relation": "mentions",
            "tail": "non-destructive evaluation, hydroelectric penstocks",
            "tail_type": "Keyword"
        },
        {
            "text": "'Seismic Analysis of Critical Facilities' and 'Monticello Nuclear Generating Plant - BioShield Evaluation' share common topics like structural integrity and seismic analysis.",
            "head": "Seismic Analysis of Critical Facilities",
            "head_type": "Article",
            "relation": "related_to",
            "tail": "Monticello Nuclear Generating Plant - BioShield Evaluation",
            "tail_type": "Article"
        },
        {
            "text": "Mark W. Marano cites the work of Julio Garcia on seismic analysis in his CEO message.",
            "head": "Mark W. Marano",
            "head_type": "Author",
            "relation": "cites",
            "tail": "Julio Garcia",
            "tail_type": "Author"
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

    # Inspect the structure of one GraphDocument
    if graph_documents:
        print("DIR:", dir(graph_documents[1]))
        print("VARS:", vars(graph_documents[1]))


    logging.info(f"Saving checkpoint: {len(graph_documents)} graph documents generated")
    with open("graph_documents_checkpoint.txt", "w") as f:
        f.write(str(len(graph_documents)))

    if not graph_documents:
        raise ValueError("No graph documents created. Aborting pipeline.")

    # 6. Push to Neo4j using parsed text
    logging.info('Connecting to Neo4j and writing graph relationships')

    uri = os.getenv("NEO4J_URI")
    auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    driver = GraphDatabase.driver(uri, auth=auth)

    def create_kg_record(tx, head, head_type, relation, tail, tail_type):
        tx.run("""
        MERGE (h:Entity {name: $head, type: $head_type})
        MERGE (t:Entity {name: $tail, type: $tail_type})
        MERGE (h)-[r:RELATION {type: $relation}]->(t)
        """, head=head, head_type=head_type, relation=relation, tail=tail, tail_type=tail_type)

    with driver.session() as session:
        for gd in graph_documents:
            try:
                raw = getattr(gd, "text", None) or getattr(gd, "page_content", None)
                if not raw:
                    raise ValueError("GraphDocument missing text content")

                rels = json.loads(raw)
                for rel in rels:
                    session.write_transaction(
                        create_kg_record,
                        rel["head"], rel["head_type"], rel["relation"], rel["tail"], rel["tail_type"]
                    )
            except Exception as e:
                logging.warning(f"Could not process graph doc: {e}")

    logging.info('Data pipeline completed successfully!')

except Exception as e:
    logging.error(f"Pipeline failed: {e}")
