# Conversational RAG with Chat History

A retrieval-augmented generation (RAG) chatbot that maintains conversation context and can answer follow-up questions intelligently. Built with LangChain and powered by OpenAI's GPT models.

## Features

- **Context-Aware Conversations**: Remembers chat history and understands follow-up questions
- **Smart Question Reformulation**: Automatically reformulates vague questions using conversation context
- **Document-Based Answers**: Answers strictly from your knowledge base
- **Source Attribution**: Shows which documents were used for each answer
- **History-Aware Retrieval**: Uses LangChain's history-aware retriever to handle pronouns and references

## Architecture

The system uses a two-stage RAG pipeline:

1. **History-Aware Retriever**: Reformulates user questions based on chat history before searching
2. **Question-Answer Chain**: Generates answers using retrieved documents and conversation context

```
User Question + Chat History
          ↓
  [Contextualize Question]
          ↓
  [Search Vector Store]
          ↓
   [Retrieved Documents]
          ↓
  [Generate Answer with Context]
          ↓
      Final Answer
```

## Prerequisites

- Python 3.9+
- OpenAI API key

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Run the Chatbot

```bash
python history-rag.py
```

## Knowledge Base

The system uses fictional documents about **Zynthora Technologies**, a made-up futuristic tech company. The knowledge base includes:

- `company_history.txt` - Company founding and milestones
- `products.txt` - Product line descriptions and specifications
- `leadership.txt` - Executive team biographies
- `locations.txt` - Office and facility information
- `events.txt` - Major company events and achievements

These documents are completely fictional and designed to test the RAG system without relying on the LLM's pre-trained knowledge.

## Example Usage

### Basic Question
```
You: When was Zynthora founded?
Bot: Zynthora Technologies was founded in 2847 by Dr. Mara Velkanos and 
     Engineer Jax Thornwell in the floating city of Nebulon-7.

Sources used:
  1. company_history.txt: Zynthora Technologies was founded in 2847...
```

### Follow-up Question (Context-Aware)
```
You: Who founded Zynthora?
Bot: Zynthora was founded by Dr. Mara Velkanos and Jax Thornwell.

You: What did they create?
Bot: They created quantum-crystalline processors and the Lumina-Core X9, 
     a bio-synthetic computing chip...
```

Notice how the bot understands "they" refers to the founders!

## Testing

The project includes comprehensive testing tools to verify the RAG system works correctly.

### Automated Testing with `test_system.py`

The automated test suite runs multiple tests to verify all components:

```bash
python test_system.py
```

**What it tests:**

1. **Vector Database Creation**: Verifies documents are loaded and chunked correctly
2. **Basic Retrieval**: Tests simple factual questions
3. **Follow-up Questions**: Validates chat history integration
4. **Context Understanding**: Checks if pronouns and references are resolved
5. **Unknown Information Handling**: Ensures the bot says "I don't know" appropriately
6. **System Components**: Validates all files and dependencies

**Sample Output:**

```
==============================================================
ZYNTHORA RAG SYSTEM - AUTOMATED TEST
==============================================================

[1/5] Loading existing vector database...
   ✓ Database loaded

[2/5] Setting up RAG chains...
   ✓ Chains configured

[3/5] Running test queries...
------------------------------------------------------------

Test 1/4: When was Zynthora founded?
Expected: Should mention 2847
Answer: Zynthora Technologies was founded in 2847...
Sources: company_history.txt, leadership.txt
✓ PASSED

Test 2/4: Who were the founders?
Expected: Should mention Velkanos and Thornwell
Answer: Dr. Mara Velkanos and Jax Thornwell...
✓ PASSED

...

==============================================================
[4/5] TEST SUMMARY
==============================================================
Total Tests: 5
Passed: 5 ✓
Failed: 0 ✗

[5/5] SYSTEM CHECKS
------------------------------------------------------------
✓ Vector database exists
✓ Fictional documents exist
✓ At least 5 document files
✓ Chat history working

==============================================================
✓ ALL TESTS PASSED! System is working correctly.
==============================================================
```

### Manual Testing with `test_prompts.md`

The `test_prompts.md` file contains 50+ detailed test cases organized by category:

**Test Categories:**

1. **Basic Retrieval** - Simple factual questions
   - Company founding, products, locations
   - Expected: Accurate answers from documents

2. **Follow-up Questions** - Chat history integration
   - Tests pronouns ("they", "it", "he", "she")
   - Expected: Context-aware responses

3. **Complex Queries** - Cross-document retrieval
   - Questions requiring information from multiple files
   - Expected: Synthesized answers from multiple sources

4. **Edge Cases** - Unknown information handling
   - Questions about data not in documents
   - Expected: "I don't know" responses

5. **Vague References** - Contextual understanding
   - "Tell me more about it", "What about them?"
   - Expected: Understands context from previous questions

**Example Test Case from `test_prompts.md`:**

```markdown
## Test 5: Historical Events

**Conversation Flow:**
You: What happened in 2855?
Bot: [Should mention The Great Merge - acquisition of Helix Dynamics]

You: How much did that cost?
Bot: [Should answer 47 billion galactic credits]

You: Who joined the company because of it?
Bot: [Should mention Sylas Moonridge]
```

### Quick Test Sequence

For a quick validation, run this sequence in the interactive chatbot:

```bash
python history-rag.py
```

Then paste these questions:

```
When was Zynthora founded?
Who were the founders?
What did they create?
Tell me about the Lumina-Core
How much does it cost?
Who is Sylas Moonridge?
What did he invent?
What happened in 2855?
Where is the headquarters?
quit
```

### What to Look For

When testing, verify:

- **Accuracy**: Answers come from fictional documents only
- **Source Attribution**: "Sources used" section appears after each answer
- **Context Awareness**: Follow-up questions understand previous context
- **Appropriate "I don't know"**: Bot admits when information isn't available
- **No Hallucination**: Doesn't make up information
- **Chunk Quality**: Retrieved text chunks are relevant and complete

## How It Works

### 1. Document Loading and Chunking

Documents are loaded from the `fictional_docs/` directory and split into chunks:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Large enough for complete context
    chunk_overlap=150,     # Ensures continuity
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

### 2. Vector Store Creation

Documents are embedded and stored in ChromaDB:

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./zynthora_db"
)
```

### 3. History-Aware Retrieval

The system reformulates questions using chat history before retrieval:

```python
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and latest user question, "
     "reformulate the question to be standalone..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)
```

### 4. Answer Generation

Retrieved documents are used to generate contextual answers:

```python
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the context below...{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

## Project Structure

```
capstone-projects/
├── history-rag.py              # Main chatbot script
├── fictional_docs/             # Knowledge base documents
│   ├── company_history.txt
│   ├── products.txt
│   ├── leadership.txt
│   ├── locations.txt
│   └── events.txt
├── test_system.py              # Automated test suite
├── test_prompts.md             # Comprehensive test cases
├── zynthora_db/                # Vector database (auto-generated)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Key Concepts Demonstrated

### 1. History-Aware Retrieval
Converts vague follow-up questions into standalone queries:
- **User**: "What did they invent?"
- **Reformulated**: "What did Dr. Mara Velkanos and Jax Thornwell invent?"

### 2. Chat History Management
Maintains conversation context while keeping memory manageable:
```python
# Keeps last 10 messages when history exceeds 20
if len(chat_history) > 20:
    chat_history = chat_history[-10:]
```

### 3. Source Attribution
Shows which documents were used for each answer to ensure transparency.

## Technical Details

### LangChain Components Used

- **`create_history_aware_retriever`**: Reformulates questions using chat history
- **`create_stuff_documents_chain`**: Combines retrieved documents with prompts
- **`create_retrieval_chain`**: Complete RAG pipeline with retrieval + generation

### Embeddings & Models

- **Embeddings**: OpenAI `text-embedding-ada-002`
- **LLM**: OpenAI `gpt-4o-mini` (temperature=0 for consistency)
- **Vector Store**: ChromaDB with persistent storage

### Chunk Size Optimization

Chunk size of 1000 characters was chosen to:
- Keep complete paragraphs together
- Maintain semantic coherence
- Include sufficient context for accurate retrieval

## Customization

### Use Your Own Documents

Replace the files in `fictional_docs/` with your own `.txt` files, then delete the vector database:

```bash
rm -rf zynthora_db
python history-rag.py
```

### Adjust Retrieval Parameters

Modify the number of chunks retrieved:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Change k value
```

### Change Chunk Size

Adjust for your document structure:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Smaller chunks
    chunk_overlap=50
)
```

## Performance Considerations

- **First Run**: Creates vector database (takes ~30 seconds)
- **Subsequent Runs**: Loads existing database (instant)
- **Per Query**: ~2-4 seconds (includes LLM calls)

## Troubleshooting

### ModuleNotFoundError
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

### OpenAI API Error
```bash
# Set your API key
export OPENAI_API_KEY="sk-..."
```

### Generic Answers
If the bot gives generic answers instead of using your documents:
```bash
# Regenerate the vector database
rm -rf zynthora_db
python history-rag.py
```


# Author

