import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from datetime import datetime
from anthropic import Anthropic
from pydantic import BaseModel
from typing import Optional
import PyPDF2
import io

claudeRouter = APIRouter(prefix='/claude', tags=["claude"])

SYSTEM_MESSAGE = "Make sure you are kind, welcoming, accurate and precise in responding towards the prompts of the user"

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

load_dotenv()
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

claude_history = []

async def extract_file_content(file: UploadFile) -> str:
    content = await file.read()
    if file.content_type == "application/pdf":
        # Handle PDF files
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        return text_content
    elif file.content_type.startswith("text/"):
        # Handle text files
        return content.decode('utf-8')
    else:
        # For other file types, return file info
        return f"[File: {file.filename} (type: {file.content_type})]"

@claudeRouter.post("/generate", response_class=JSONResponse)
async def claudeTime(
    prompt: str = Form(...),
    temperature: float = Form(0.7),
    file: UploadFile = File(None)
):
    global claude_history
    try:
        # If file is provided, append its content to the prompt
        if file:
            try:
                file_text = await extract_file_content(file)
                full_prompt = f"{prompt}\n\nHere's the file content:\n\n{file_text}"
            except Exception as file_error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing file: {str(file_error)}"
                )
        else:
            full_prompt = prompt

        claude_history.append({
            "role": "user",
            "content": full_prompt
        })

        # Convert history to messages format
        messages = []
        for msg in claude_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            system=SYSTEM_MESSAGE,
            messages=messages,
            temperature=temperature,
            max_tokens=1024
        )

        ass_msg = response.content[0].text
        claude_history.append({
            "role": "assistant",
            "content": ass_msg
        })

        return JSONResponse(content={
            "text": ass_msg,
            "model": "claude-sonnet-4-20250514",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error details: {str(e)}")  # Add detailed logging
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )