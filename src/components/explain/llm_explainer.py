import os
import hashlib
from dotenv import load_dotenv
from src.logger import get_logger

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables safely
load_dotenv()

logger = get_logger(__name__)

# Define the system instructions
SYSTEM_PROMPT = """You are explaining a product rating prediction made by a retrieval
system. You will be given the predicted rating and a list of the most similar reviews
used to compute it (with similarity scores and individual ratings).

Rules:
- Only reference information that appears in the provided reviews. Do not invent details.
- Be concise: 2-4 sentences.
- Mention both positive and negative themes if both appear.
- Do not repeat the numeric rating back verbatim more than once."""

# 1. Define the Chat Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Predicted rating: {predicted_rating}/5\n\nRetrieved similar reviews:\n{context}\n\nExplain why this rating was predicted, grounded only in the reviews above.")
])

# 2. Initialize Model and Output Parser
# ChatGroq automatically detects os.environ["GROQ_API_KEY"] loaded by dotenv
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
    output_parser = StrOutputParser()
    
    # 3. Construct the LCEL Chain
    chain = prompt_template | llm | output_parser
except Exception as e:
    logger.error(f"Failed to initialize LangChain Groq pipeline: {e}")
    chain = None

def explain_rating(predicted_rating, top_reviews, similarities):
    # Prepare the context string from the dataframe
    context = "\n".join(
        f"- (similarity={sim:.2f}, rating={row['rating']}/5): {row['review_text'][:300]}"
        for (_, row), sim in zip(top_reviews.iterrows(), similarities)
    )
    
    if chain is None:
        logger.error("LCEL Chain is not initialized. Check your GROQ_API_KEY.")
        return f"Predicted {predicted_rating:.2f}/5 based on a weighted average of {len(top_reviews)} similar reviews."
    
    try:
        logger.info("Using LangChain + Groq (llama-3.1-8b-instant) for explanation...")
        
        # 4. Invoke the chain by passing the template variables
        response = chain.invoke({
            "predicted_rating": f"{predicted_rating:.2f}",
            "context": context
        })
        
        return response
            
    except Exception as e:
        logger.error(f"Groq LLM explanation failed: {e}")
        return f"Predicted {predicted_rating:.2f}/5 based on a weighted average of {len(top_reviews)} similar reviews."

# Caching Mechanism
_cache = {}
def explain_rating_cached(predicted_rating, top_reviews, similarities):
    key = hashlib.md5((str(predicted_rating) + "".join(top_reviews["review_text"])).encode()).hexdigest()
    if key not in _cache:
        _cache[key] = explain_rating(predicted_rating, top_reviews, similarities)
    return _cache[key]  