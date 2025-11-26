import dspy
from typing import Optional

class RouterSignature(dspy.Signature):
    question: str = dspy.InputField(desc="User's retail analytics question")
    format_hint: str = dspy.InputField(desc="Expected output format (int, float, dict, list)")
    route: str = dspy.OutputField(desc="Route: 'rag', 'sql', or 'hybrid'")
    reasoning: str = dspy.OutputField(desc="Reason for routing decision")

class NLtoSQLSignature(dspy.Signature):
    question: str = dspy.InputField(desc="User question needing SQL")
    db_schema: str = dspy.InputField(desc="Database schema")
    constraints: str = dspy.InputField(desc="Extracted constraints")
    format_hint: str = dspy.InputField(desc="Expected output format")
    sql_query: str = dspy.OutputField(desc="Generated SQL query")
    explanation: str = dspy.OutputField(desc="Explanation of query logic")

class SQLRepairSignature(dspy.Signature):
    question: str = dspy.InputField(desc="Original question")
    db_schema: str = dspy.InputField(desc="Database schema")
    failed_sql: str = dspy.InputField(desc="SQL query that failed")
    error_message: str = dspy.InputField(desc="Error message")
    fixed_sql: str = dspy.OutputField(desc="Corrected SQL query")
    fix_explanation: str = dspy.OutputField(desc="What was fixed")

class SynthesizerSignature(dspy.Signature):
    question: str = dspy.InputField(desc="Original question")
    format_hint: str = dspy.InputField(desc="Required output format")
    doc_chunks: str = dspy.InputField(desc="Retrieved document chunks")
    sql_result: str = dspy.InputField(desc="SQL results")
    sql_query: str = dspy.InputField(desc="Executed SQL query")
    final_answer: str = dspy.OutputField(desc="Answer in required format")
    explanation: str = dspy.OutputField(desc="Explanation")
    doc_citations: str = dspy.OutputField(desc="Doc chunk IDs used")

class ConstraintExtractorSignature(dspy.Signature):
    question: str = dspy.InputField(desc="User question")
    doc_chunks: str = dspy.InputField(desc="Retrieved document chunks")
    date_range: str = dspy.OutputField(desc="Date range or 'none'")
    kpi_formula: str = dspy.OutputField(desc="KPI formula or 'none'")
    categories: str = dspy.OutputField(desc="Relevant categories or 'none'")
    other_constraints: str = dspy.OutputField(desc="Other constraints")

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouterSignature)
    def forward(self, question: str, format_hint: str) -> dspy.Prediction:
        route = self._rule_based_route(question, format_hint)
        if route:
            return dspy.Prediction(route=route, reasoning=f"Rule-based routing: {route}")
        try:
            result = self.classify(question=question, format_hint=format_hint)
            route = result.route.lower().strip().strip("'\"")
            if route not in ['rag','sql','hybrid']:
                if 'sql' in route or 'database' in route or 'query' in route:
                    route='sql'
                elif 'rag' in route or 'doc' in route or 'policy' in route:
                    route='rag'
                else:
                    route='hybrid'
            return dspy.Prediction(route=route, reasoning=result.reasoning)
        except:
            return dspy.Prediction(route="hybrid", reasoning="Defaulted to hybrid")

    def _rule_based_route(self, question: str, format_hint: str) -> Optional[str]:
        q = question.lower()
        if any(x in q for x in ['policy','return window','return policy','returns']) and not any(x in q for x in ['revenue','total','sum','count']):
            return 'rag'
        if 'all-time' in q or 'all time' in q or ('top' in q and 'product' in q and 'revenue' in q):
            return 'sql'
        if any(x in q for x in ['marketing calendar','kpi','aov','average order value','gross margin','summer beverages','winter classics']):
            return 'hybrid'
        return None

class NLtoSQL(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(NLtoSQLSignature)
    def forward(self, question: str, schema: str, constraints: str, format_hint: str) -> dspy.Prediction:
        result = self.generate(question=question, db_schema=schema, constraints=constraints, format_hint=format_hint)
        sql = result.sql_query.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip()=="```" else lines[1:])
        sql = sql.replace("TOP ","LIMIT_PLACEHOLDER ")
        import re
        match = re.search(r'LIMIT_PLACEHOLDER\s+(\d+)', sql)
        if match:
            limit_num = match.group(1)
            sql = re.sub(r'LIMIT_PLACEHOLDER\s+\d+\s*','',sql)+f" LIMIT {limit_num}"
        sql = sql.strip().rstrip(";")+";"
        return dspy.Prediction(sql_query=sql, explanation=result.explanation)

class SQLRepair(dspy.Module):
    def __init__(self):
        super().__init__()
        self.repair = dspy.ChainOfThought(SQLRepairSignature)
    def forward(self, question: str, schema: str, failed_sql: str, error_message: str) -> dspy.Prediction:
        result = self.repair(question=question, db_schema=schema, failed_sql=failed_sql, error_message=error_message)
        sql = result.fixed_sql.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip()=="```" else lines[1:])
        import re
        match = re.search(r'TOP\s+(\d+)', sql, re.IGNORECASE)
        if match:
            limit_num = match.group(1)
            sql = re.sub(r'TOP\s+\d+\s*','',sql,flags=re.IGNORECASE).rstrip()+" LIMIT "+limit_num
        sql = sql.strip().rstrip(";")+";"
        return dspy.Prediction(fixed_sql=sql, fix_explanation=result.fix_explanation)

class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizerSignature)
    def forward(self, question: str, format_hint: str, doc_chunks: str, sql_result: str, sql_query: str) -> dspy.Prediction:
        result = self.synthesize(question=question, format_hint=format_hint, doc_chunks=doc_chunks, sql_result=sql_result, sql_query=sql_query)
        return dspy.Prediction(final_answer=result.final_answer, explanation=result.explanation, doc_citations=result.doc_citations)

class ConstraintExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ConstraintExtractorSignature)
    def forward(self, question: str, doc_chunks: str) -> dspy.Prediction:
        result = self.extract(question=question, doc_chunks=doc_chunks)
        return dspy.Prediction(date_range=result.date_range, kpi_formula=result.kpi_formula, categories=result.categories, other_constraints=result.other_constraints)

def configure_dspy_local(model_name: str = "phi3.5:3.8b-mini-instruct-q4_K_M"):
    lm = dspy.LM(model=f"ollama_chat/{model_name}", api_base="http://localhost:11434", api_key="", temperature=0.1, max_tokens=1024)
    dspy.configure(lm=lm)
    return lm

if __name__=="__main__":
    print("Testing DSPy modules...")
    try:
        configure_dspy_local()
        router = Router()
        result = router(question="What is the return policy for Beverages?", format_hint="int")
        print(f"\nRouter test:\n  Route: {result.route}\n  Reasoning: {result.reasoning}")
    except Exception as e:
        print(f"Error (is Ollama running?): {e}")
