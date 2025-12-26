from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

from vectorstore.hybrid_retriever import create_hybrid_retriever

load_dotenv()

def create_qa_chain(vectorstore, memory, role, all_documents):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=512,
    )

    # ---- ROLE INSTRUCTION ----
    if role == "Admin":
        role_instruction = """
You are an enterprise admin assistant.
Give detailed, precise, and policy-oriented answers.
"""
    else:
        role_instruction = """
You are an enterprise user assistant.
Give short, clear, non-technical answers.
"""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
{role_instruction}

IMPORTANT RULES:
- Answer ONLY from the context.
- DO NOT guess.
- If not found, reply exactly:
"Information not found in the provided documents."

Context:
{{context}}

Question:
{{question}}

Answer:
"""
    )

    hybrid_retriever = create_hybrid_retriever(all_documents, vectorstore)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    return chain
