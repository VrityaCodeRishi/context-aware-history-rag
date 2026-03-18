# Test Prompts for Zynthora RAG System

## Quick Setup Check
Run these first to verify the system loads:
```bash
cd rag-fundamentals/capstone-projects
source ../../venv/bin/activate
export OPENAI_API_KEY="your-key-here"
python history-rag.py
```

---

## Test 1: Basic Retrieval (No History Needed)
Test that the system can retrieve simple facts.

**Prompt 1:** `When was Zynthora founded?`
**Expected:** Should mention 2847, Dr. Mara Velkanos, and Jax Thornwell

**Prompt 2:** `What products does Zynthora make?`
**Expected:** Should list products like Lumina-Core, Resonance Drives, Phase-Shift modules

**Prompt 3:** `Where is Zynthora headquarters?`
**Expected:** Should mention Arcturus Prime, Crystalline Valley

---

## Test 2: Follow-up Questions (Chat History)
This tests the history-aware retriever!

**Conversation Flow:**
```
You: Who is Dr. Mara Velkanos?
Bot: [Should give her bio - CEO, born on Kepler-442b, etc.]

You: Where was she born?
Bot: [Should answer "Kepler-442b" using context from previous answer]

You: What about her education?
Bot: [Should mention University of Titan, doctorate in Quantum Biophysics]
```

---

## Test 3: Product Details with Follow-ups

**Conversation Flow:**
```
You: Tell me about the Lumina-Core
Bot: [Should describe bio-synthetic processor, 800 zettaflops, photosynthetic enzymes]

You: How much does it cost?
Bot: [Should mention 12,000 to 45,000 galactic credits]

You: What about the newer version?
Bot: [Should mention X12 model with 1.4 yottaflops]
```

---

## Test 4: Complex Query Across Multiple Documents

**Prompt:** `Who invented the Phase-Shift Algorithm and where do they work?`
**Expected:** Should connect Sylas Moonridge (from leadership.txt) with the Phase-Shift invention (from products.txt and events.txt)

---

## Test 5: Historical Events

**Conversation Flow:**
```
You: What happened in 2855?
Bot: [Should mention The Great Merge - acquisition of Helix Dynamics]

You: How much did that cost?
Bot: [Should answer 47 billion galactic credits]

You: Who joined the company because of it?
Bot: [Should mention Sylas Moonridge]
```

---

## Test 6: Location-specific Questions

**Conversation Flow:**
```
You: Tell me about the Titan facility
Bot: [Should describe Titan Industrial Zone - hydrocarbon lakes, manufacturing]

You: How many people work there?
Bot: [Should mention 125,000 workers]

You: What do they produce?
Bot: [Should mention 80% of Lumina-Core processors and Resonance Drives]
```

---

## Test 7: Unknown Information (Should say "I don't know")

**Prompt:** `What is Zynthora's stock price?`
**Expected:** Should say it doesn't have that information (not in documents)

**Prompt:** `Who is the CFO of Zynthora?`
**Expected:** Should say information not available (CFO not in leadership list)

---

## Test 8: Vague Pronouns (Tests Contextualization)

**Conversation Flow:**
```
You: Who founded Zynthora?
Bot: [Dr. Mara Velkanos and Jax Thornwell]

You: What did they invent?
Bot: [Should understand "they" refers to the founders, mention quantum-crystalline processors, bio-synthetic computing]

You: Where did they start it?
Bot: [Should mention Nebulon-7, the floating city]
```

---

## Test 9: Multiple People/Products

**Prompt:** `Compare Sylas Moonridge and Jax Thornwell`
**Expected:** Should pull information about both from leadership.txt

**Prompt:** `What's the difference between Lumina-Core and Helix Quantum Arrays?`
**Expected:** Should differentiate the two product lines

---

## Test 10: Source Attribution Check

Look for the "📚 Sources used:" section after each answer to verify:
- Correct source files are being retrieved
- Multiple relevant documents are being used when needed
- Preview text makes sense for the query

---

## Expected Behavior Checklist

✅ System loads fictional documents from `fictional_docs/` folder
✅ Creates `zynthora_db` vector database (only on first run)
✅ Answers questions based ONLY on the fictional documents
✅ Shows sources used for each answer
✅ Remembers conversation context (follow-up questions work)
✅ Reformulates vague questions using chat history
✅ Says "I don't know" when information isn't in documents
✅ Chat history is managed (keeps last 10 messages when > 20)

---

## Troubleshooting

**If you get "No module named..." errors:**
- Make sure venv is activated: `source ../../venv/bin/activate`
- Check you're in the right directory: `rag-fundamentals/capstone-projects`

**If you get OpenAI API errors:**
- Set your API key: `export OPENAI_API_KEY="sk-..."`
- Or create a `.env` file with `OPENAI_API_KEY=sk-...`

**If answers seem generic (not from your docs):**
- Check that `zynthora_db` directory was created
- Try deleting `zynthora_db` and running again to regenerate

**To test with verbose output:**
Add this after creating the chains (around line 65):
```python
import langchain
langchain.verbose = True
```

---

## Quick Copy-Paste Test Sequence

```
When was Zynthora founded?
Who were the founders?
What did they create?
Tell me about the Lumina-Core
How much does it cost?
Who is Sylas Moonridge?
What did he invent?
What happened in 2855?
How much did that cost?
Where is the headquarters?
How many people work there?
quit
```
