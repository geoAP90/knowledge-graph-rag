import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.info('🚀 Starting up the Knowledge Graph RAG...')

# ✅ Use the community version of Neo4jGraph and Ollama
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain.chains import GraphCypherQAChain

# Instantiate Neo4j connector
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
logging.info(f'🔗 Connecting to Neo4J at: {neo4j_uri}')
graph = Neo4jGraph()

# Instantiate Ollama LLM
logging.info('🧠 Initializing Ollama LLM (llama3)...')
llm = Ollama(model='qwen2.5-coder:14b', temperature=0.0)

# Create the GraphCypherQAChain
logging.info('🔧 Setting up GraphCypherQAChain...')
chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True,
    allow_dangerous_requests=True  # Required for unrestricted Cypher queries
)

logging.info('✅ Knowledge Graph RAG is ready!')
logging.info('=' * 60)

def main():
    logging.info('💬 Type "exit" to quit the program.')
    while True:
        question = input('\nAsk me a question: ')
        if question.lower().strip() == 'exit':
            logging.info('👋 Exiting...')
            break
        try:
            result = chain.invoke({"query": question})
            print(result.get('result', '🤔 No answer returned.'))
        except Exception as e:
            logging.error(f"⚠️ Error during query: {e}")
            print("❌ An error occurred. Check the logs.")

if __name__ == '__main__':
    main()
