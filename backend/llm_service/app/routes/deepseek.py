from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

deepseekRouter = APIRouter(prefix="/deepseek", tags=["deepseek"])

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

@deepseekRouter.post("/generate", response_class=JSONResponse)
async def generate_text(
    prompt: str,
    temperature: float = 0.7,
):
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="DEEPSEEK_API_KEY not found in environment variables"
            )
            
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        
        messages = [{"role": "user", "content": prompt}]
        
        # Call DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-reasoner", # or "deepseek-reasoner" for reasoning tasks
            messages=messages,
            temperature=temperature
        )
        
        return JSONResponse(content={
            "text": response.choices[0].message.content,
            "model": "deepseek-chat",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )
