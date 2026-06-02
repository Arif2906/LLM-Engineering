from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import json
import rag_pipeline as rag
from typing import Optional

load_dotenv()
router_api_key=os.getenv("OpenRouter")
router_client= OpenAI(api_key=router_api_key, base_url='https://openrouter.ai/api/v1')

class TestQuestion(BaseModel):
    question: str= Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")

class RetrievalEval(BaseModel):
    mrr: float  = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: Optional[float] = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")

class AnswerEval(BaseModel):
    accuracy: float = Field(description="How factually correct is the answer compared to the reference answer?")
    completeness: float = Field(description="How complete is the answer in addressing all aspects of the question?")    
    relevance: float = Field(description="How relevant is the answer to the specific question asked?")    
    feedback: str = Field(description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context") 

# load jsonl file for evaluation
def load_tests(filepath)-> list[TestQuestion]:
    tests = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tests.append(json.loads(line))
    return tests

def test_eval(test: TestQuestion):
    rag_answer, retrieved_docs =  rag.answer_question(test.question)
    calculate_mrr(test.keywords, retrieved_docs)
    calculate_keyword_coverage(test.keywords, retrieved_docs)
    keywords_found, keyword_coverage=calculate_keyword_coverage(keyword=test.keywords, retrieved_docs=retrieved_docs)
    return RetrievalEval(
        mrr=calculate_mrr(test.keywords, retrieved_docs),
        keywords_found=keywords_found,
        total_keywords=len(test.keywords),
        keyword_coverage=keyword_coverage
    )

def calculate_mrr(keyword: str, retrieved_docs: list):
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0

def calculate_keyword_coverage(keyword: str, retrieved_docs: list):
    keyword_lower_set = set(keyword.lower().split())
    distinct_keywords = set()
    for doc in retrieved_docs:
        for word in doc.page_content.lower().split():
            distinct_keywords.add(word)

    distinct_keywords = distinct_keywords.intersection(keyword_lower_set)
    
    return distinct_keywords,len(distinct_keywords)/len(keyword)
 
def evaluate_answer(test: TestQuestion):
    
    rag_answer, retrieved_docs =  rag.answer_question(test.question)
    system_prompt="""
    You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.4

    Reference Answer :
    {test.reference_answer}

    Generated Answer :
    {rag_answer}

    Question :
    {test.question}

    Guildelines:
    Please evaluate the generated answer based on the reference answer and the retrieved context.
    Accuracy: How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3.
    completeness: How complete the answer is compared to reference answer? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included.
    relevance: How relevant the answer is to the question asked? 1 (irrelevant) to 5 (ideal - perfectly relevant). An answer that is not relevant to the question asked would score 1.
    feedback: Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context

    """
    result = router_client.chat.completions.create(model='nvidia/nemotron-3-super-120b-a12b:free', messages=
    [{"role":"system", "content": system_prompt.format(test=test, rag_answer=rag_answer)}
    ,{"role": "user", "content" : test.question}], response_format= AnswerEval)

    eval = AnswerEval.model_validate_json(result.choices[0].message.content)
    return eval, retrieved_docs , rag_answer


if __name__ == "__main__":
    tests = load_tests("eavaluation/tests.jsonl")
    print(test_eval(tests[0]))
    #evaluate_answer(tests[0])
