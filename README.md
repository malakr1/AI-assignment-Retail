# Retail Analytics Copilot

A hybrid AI agent that combines retrieval-augmented generation (RAG) and SQL query generation to answer retail analytics questions. Built with LangGraph, DSPy, and local LLM support via Ollama.

## Important Note: Date Issue

**Known Issue**: The original assignment instructions reference dates from 1997 (e.g., "Summer Beverages 1997", "Winter Classics 1997"). However, the Northwind SQLite database provided via the download link:

```
https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
```

contains order data from **2012-2023**, not 1996-1998. 

**Solution**: This implementation uses **2016** as the reference year for all campaign dates:
- Summer Beverages 2016: June 2016
- Winter Classics 2016: December 2016

The constraint extraction logic in `graph_hybrid.py` has been updated to automatically map these campaign references to the correct 2016 dates.

## Features

- **Intelligent Routing**: Automatically routes questions to RAG, SQL, or hybrid processing based on the query type
- **Document Retrieval**: TF-IDF-based semantic search over markdown policy documents
- **Natural Language to SQL**: Generates and executes SQL queries against retail databases
- **Self-Healing**: Automatic SQL repair on query failures with iterative refinement
- **Template Optimization**: Uses pre-built SQL templates for common query patterns
- **Constraint Extraction**: Extracts date ranges, KPIs, categories, and formulas from context
- **Multi-Format Output**: Supports int, float, dict, and list output formats
- **Citation Tracking**: Provides source citations for answers from both documents and database tables

## Architecture

The system uses a LangGraph-based workflow with the following nodes:

1. **Router**: Classifies questions as RAG, SQL, or hybrid
2. **Retriever**: Searches document chunks using TF-IDF similarity
3. **Planner**: Extracts constraints (dates, KPIs, categories) from retrieved docs
4. **NL-to-SQL**: Generates SQL queries using templates or LLM
5. **Executor**: Runs SQL queries against the database
6. **Validator**: Checks query results and output format
7. **Repair**: Fixes failed SQL queries (up to 2 attempts)
8. **Synthesizer**: Combines RAG and SQL results into final answer

## Installation

```bash
# Install dependencies
pip install dspy-ai langgraph scikit-learn numpy click rich

# Install and start Ollama
# Visit https://ollama.ai for installation instructions

# Pull the default model
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

## Usage

### Basic Usage

```python
from agent.graph_hybrid import RetailAnalyticsCopilot

# Initialize the agent
agent = RetailAnalyticsCopilot(model_name="phi3.5:3.8b-mini-instruct-q4_K_M")

# Run a single question
output, trace = agent.run(
    question="What is the return window for unopened Beverages?",
    format_hint="int",
    question_id="q1"
)

print(f"Answer: {output['final_answer']}")
print(f"Confidence: {output['confidence']}")
print(f"Citations: {output['citations']}")
```

### Batch Processing

```bash
# Process a batch of questions from JSONL file
python run_agent_hybrid.py \
    --batch questions.jsonl \
    --out results.jsonl \
    --model phi3.5:3.8b-mini-instruct-q4_K_M \
    --verbose
```

Input format (JSONL):
```json
{"id": "q1", "question": "What are the top 3 products by revenue?", "format_hint": "list[{product:str, revenue:float}]"}
{"id": "q2", "question": "What was the AOV for Summer Beverages 2016?", "format_hint": "float"}
```

Output format (JSONL):
```json
{"id": "q1", "final_answer": [...], "sql": "SELECT...", "confidence": 0.85, "explanation": "...", "citations": [...]}
```

## Project Structure

```
my_project/ 
├─ agent/ 
│  ├─ graph_hybrid.py           
│  ├─ dspy_signatures.py        
(Router/NL→SQL/Synth) 
│  ├─ rag/retrieval.py          
search) 
│  └─ tools/sqlite_tool.py      
├─ data/ 
│  └─ northwind.sqlite          
├─ docs/ 
│  ├─ marketing_calendar.md 
│  ├─ kpi_definitions.md 
│  ├─ catalog.md 
│  └─ product_policy.md 
# your LangGraph (≥6 nodes + repair loop) 
# DSPy Signatures/Modules 
# TF-IDF or simple retriever (chunking + 
# DB access + schema introspection 
# downloaded DB 
├─ sample_questions_hybrid_eval.jsonl 
├─ run_agent_hybrid.py          
# main entrypoint (CLI contract below) 
└─ requirements.txt 

```

## Configuration
```python
# Local Ollama model
from agent.dspy_signatures import configure_dspy_local
configure_dspy_local("llama3:8b")
```

### Document Path

```python
from pathlib import Path
from agent.rag.retrieval import get_retriever

retriever = get_retriever(docs_path=Path("custom/docs/path"))
```

## Supported Query Types

- **Policy Questions**: Return policies, product guidelines
- **Aggregate Queries**: Revenue totals, quantities, counts
- **Time-Series Analysis**: Date-range filtered metrics
- **KPI Calculations**: AOV, gross margin, conversion rates
- **Top-N Rankings**: Best products, customers, categories

## Requirements

- Python 3.8+
- Ollama (for local LLM inference)
- SQLite database with retail schema
- Markdown documents for RAG

## License

MIT
