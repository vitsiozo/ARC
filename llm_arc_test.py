import langchain # Main LangChain import
from langchain_openai import ChatOpenAI # To work with OpenAI
from langchain_anthropic import ChatAnthropic # To work with Anthropic (optional)
from langchain_google_genai import ChatGoogleGenerativeAI # To work with Gemini (optional)
from langchain_core.output_parsers import JsonOutputParser # To help with structured output
from langchain_core.prompts import PromptTemplate # To help create our prompt
from langchain_core.pydantic_v1 import BaseModel, Field # To help with defining what output structure we want

from typing import List, Tuple
import os
import json

# Get api key for chatgpt
OPENAI_API_KEY = os.getenv('CHATGPT_API_KEY')
print (OPENAI_API_KEY)