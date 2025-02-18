# Run Ollama locally before running this script and pull the llama3 model

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from yaspin import yaspin
spinner = yaspin(text="Thinking...", color="yellow")


embeddings = OllamaEmbeddings(
    model="llama3",
)

collection_name = "my_docs"
connection = "postgresql+psycopg://sujeet@localhost:5432/sujeet"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]

vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

llm = OllamaLLM(model="llama3")  # You can replace "llama3" with any other model

prompt = ChatPromptTemplate.from_messages([
(
"system", 
 """You are an AI assistant with access to a knowledge base. Your task is to answer user queries based on the retrieved documents.

**Instructions:**
- Use only the retrieved information to answer the query.
- If the answer is not explicitly found in the documents, try to infer a relevant response.
- If no relevant information exists, say "I couldn't find relevant information in the documents."

Context:{context}"""
),
("human", "{input}"),
])

# Create RetrievalQA chain with custom prompt
qa_chain = question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# Query and retrieve relevant documents
my_query = "What class community center offers?"


# Now, pass 'input_data' to 'invoke' method
spinner.start()

response = chain.invoke({"input": my_query})

spinner.stop()

print(response['answer'])  # Output the final response
