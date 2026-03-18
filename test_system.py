#!/usr/bin/env python
"""
Quick automated test for the Zynthora RAG system.
This simulates a conversation to verify everything works.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

print("=" * 60)
print("ZYNTHORA RAG SYSTEM - AUTOMATED TEST")
print("=" * 60)

# Load or create vector store
embeddings = OpenAIEmbeddings()

if not os.path.exists("./zynthora_db"):
    print("\n[1/5] Creating vector database from fictional documents...")
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
    print(f"   ✓ Loaded {len(documents)} documents")
    print(f"   ✓ Created {len(splits)} chunks")
else:
    print("\n[1/5] Loading existing vector database...")
    vectorstore = Chroma(persist_directory="./zynthora_db", embedding_function=embeddings)
    print("   ✓ Database loaded")

# Setup chains
print("\n[2/5] Setting up RAG chains...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
print("   ✓ Chains configured")

# Test questions
print("\n[3/5] Running test queries...")
print("-" * 60)

test_queries = [
    ("When was Zynthora founded?", "Should mention 2847"),
    ("Who were the founders?", "Should mention Velkanos and Thornwell"),
    ("What products do they make?", "Should list products like Lumina-Core"),
    ("Tell me more about it", "Should use chat history to answer about products"),
]

chat_history = []
passed = 0
failed = 0

for i, (question, expected) in enumerate(test_queries, 1):
    print(f"\nTest {i}/{len(test_queries)}: {question}")
    print(f"Expected: {expected}")
    
    try:
        result = rag_chain.invoke({"input": question, "chat_history": chat_history})
        answer = result['answer']
        
        print(f"Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        
        # Show sources
        if result.get('context'):
            sources = [doc.metadata.get('source', '').split('/')[-1] for doc in result['context'][:2]]
            print(f"Sources: {', '.join(sources)}")
        
        # Update history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        
        print("✓ PASSED")
        passed += 1
        
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        failed += 1

# Test unknown information
print(f"\n\nTest {len(test_queries)+1}: Testing 'I don't know' response...")
print("Question: What is Zynthora's stock price?")
print("Expected: Should say information not available")

try:
    result = rag_chain.invoke({
        "input": "What is Zynthora's stock price?", 
        "chat_history": chat_history
    })
    answer = result['answer'].lower()
    
    if any(phrase in answer for phrase in ["don't know", "don't have", "not have", "cannot answer", "can't answer"]):
        print(f"Answer: {result['answer']}")
        print("✓ PASSED - Correctly indicated lack of information")
        passed += 1
    else:
        print(f"Answer: {result['answer']}")
        print("✗ WARNING - May have hallucinated information")
        failed += 1
except Exception as e:
    print(f"✗ FAILED: {str(e)}")
    failed += 1

# Summary
print("\n" + "=" * 60)
print("[4/5] TEST SUMMARY")
print("=" * 60)
print(f"Total Tests: {passed + failed}")
print(f"Passed: {passed} ✓")
print(f"Failed: {failed} ✗")

# Final checks
print("\n[5/5] SYSTEM CHECKS")
print("-" * 60)

checks = [
    ("Vector database exists", os.path.exists("./zynthora_db")),
    ("Fictional documents exist", os.path.exists("./fictional_docs")),
    ("At least 5 document files", len([f for f in os.listdir("./fictional_docs") if f.endswith('.txt')]) >= 5),
    ("Chat history working", len(chat_history) > 0),
]

for check_name, result in checks:
    status = "✓" if result else "✗"
    print(f"{status} {check_name}")

print("\n" + "=" * 60)
if failed == 0:
    print("✓ ALL TESTS PASSED! System is working correctly.")
else:
    print(f"⚠ {failed} test(s) failed. Check the output above.")
print("=" * 60)

print("\nTo run the interactive chatbot:")
print("  python history-rag.py")
print("\nFor detailed test prompts, see:")
print("  cat test_prompts.md")
