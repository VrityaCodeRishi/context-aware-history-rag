"""
Weekend Project: Conversational RAG with chat history.

The chatbot should:
1. Answer from your knowledge base
2. Remember conversation context (follow-up questions work)
3. Reformulate follow-up questions using chat history
4. Show retrieved sources
"""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os


embeddings = OpenAIEmbeddings()


if not os.path.exists("./zynthora_db"):
    print("Loading fictional documents into vector store...")
    loader = DirectoryLoader("./fictional_docs", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./zynthora_db"
    )
    print(f"Loaded {len(splits)} document chunks into vector store.\n")
else:
    print("Loading existing vector store...\n")
    vectorstore = Chroma(persist_directory="./zynthora_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-5.2", temperature=0)


contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and latest user question, "
     "reformulate the question to be standalone (understandable without history). "
     "Do NOT answer it, just reformulate if needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the context below. If you can't answer, say so.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


chat_history = []
print("RAG Chatbot ready. Type 'quit' to exit.\n")


while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    result = rag_chain.invoke({"input": question, "chat_history": chat_history})

    print(f"Bot: {result['answer']}\n")
    

    if result.get('context'):
        print("Sources used:")
        for i, doc in enumerate(result['context'][:3], 1):  # Show top 3 sources
            source = doc.metadata.get('source', 'Unknown')
            source_file = source.split('/')[-1] if source else 'Unknown'
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  {i}. {source_file}: {preview}...")
        print()

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["answer"]))


    if len(chat_history) > 20:
        chat_history = chat_history[-10:]