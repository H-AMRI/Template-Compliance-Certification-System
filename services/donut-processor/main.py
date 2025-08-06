import os
import io
import torch
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Donut Processor Service", version="1.0.0")

# Global variables for model and processor
donut_model = None
donut_processor = None


class AnalysisResponse(BaseModel):
    document_type: str
    fields: Dict[str, Any]
    confidence: float
    raw_output: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Load Donut model on startup."""
    global donut_model, donut_processor
    
    print("Loading Donut model...")
    model_name = os.getenv("DONUT_MODEL", "naver-clova-ix/donut-base")
    
    try:
        donut_processor = DonutProcessor.from_pretrained(model_name)
        donut_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            donut_model = donut_model.cuda()
            print("Donut model loaded on GPU")
        else:
            print("Donut model loaded on CPU")
        
        donut_model.eval()
        print(f"Successfully loaded Donut model: {model_name}")
        
    except Exception as e:
        print(f"Failed to load Donut model: {str(e)}")
        raise


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(
    file: UploadFile = File(...),
    task: str = "classification"
):
    """
    Analyze a document image using Donut model.
    
    Args:
        file: Image file to analyze
        task: Task type (classification, docvqa, information_extraction)
    
    Returns:
        Analysis results including document type and extracted fields
    """
    if donut_model is None or donut_processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prepare task-specific prompt
        task_prompts = {
            "classification": "<s_classification>",
            "docvqa": "<s_docvqa><s_question>What are the key fields and values in this document?</s_question><s_answer>",
            "information_extraction": "<s_cord-v2>"
        }
        
        prompt = task_prompts.get(task, task_prompts["classification"])
        
        # Process image
        pixel_values = donut_processor(image, return_tensors="pt").pixel_values
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        # Generate output
        with torch.no_grad():
            # Prepare decoder input
            decoder_input_ids = donut_processor.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
            
            if torch.cuda.is_available():
                decoder_input_ids = decoder_input_ids.cuda()
            
            # Generate
            outputs = donut_model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=1024,
                early_stopping=True,
                pad_token_id=donut_processor.tokenizer.pad_token_id,
                eos_token_id=donut_processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )
        
        # Decode output
        decoded_text = donut_processor.batch_decode(outputs.sequences)[0]
        
        # Parse the output
        parsed_result = parse_donut_output(decoded_text, task)
        
        return AnalysisResponse(
            document_type=parsed_result["document_type"],
            fields=parsed_result["fields"],
            confidence=parsed_result["confidence"],
            raw_output=decoded_text if os.getenv("DEBUG", "false").lower() == "true" else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        await file.close()


def parse_donut_output(decoded_text: str, task: str) -> Dict[str, Any]:
    """
    Parse Donut model output into structured format.
    
    Args:
        decoded_text: Raw decoded text from model
        task: Task type used for generation
    
    Returns:
        Parsed result with document_type, fields, and confidence
    """
    result = {
        "document_type": "unknown",
        "fields": {},
        "confidence": 0.0
    }
    
    try:
        # Remove special tokens
        cleaned_text = decoded_text
        for token in ["<s_", "</s_", "<s>", "</s>"]:
            cleaned_text = cleaned_text.replace(token, "")
        
        if task == "classification":
            # Extract classification result
            if "classification>" in decoded_text:
                parts = decoded_text.split("classification>")
                if len(parts) > 1:
                    doc_type = parts[1].split("<")[0].strip()
                    result["document_type"] = doc_type if doc_type else "unknown"
                    result["confidence"] = 0.85  # Placeholder confidence
        
        elif task == "docvqa":
            # Extract Q&A result
            if "answer>" in decoded_text:
                answer_part = decoded_text.split("answer>")[1].split("</")[0]
                
                # Parse key-value pairs
                lines = answer_part.strip().split("\n")
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        result["fields"][key.strip()] = value.strip()
                
                # Infer document type from fields
                result["document_type"] = infer_document_type(result["fields"])
                result["confidence"] = 0.8
        
        elif task == "information_extraction":
            # Parse CORD-like format
            try:
                # Look for JSON-like structure in output
                import re
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if json_match:
                    fields = json.loads(json_match.group())
                    result["fields"] = flatten_nested_dict(fields)
                    result["document_type"] = infer_document_type(result["fields"])
                    result["confidence"] = 0.75
            except json.JSONDecodeError:
                # Fallback to line parsing
                lines = cleaned_text.strip().split("\n")
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        result["fields"][key.strip()] = value.strip()
                result["document_type"] = infer_document_type(result["fields"])
                result["confidence"] = 0.7
        
        # Set minimum confidence if we extracted any fields
        if result["fields"] and result["confidence"] == 0.0:
            result["confidence"] = 0.6
            
    except Exception as e:
        print(f"Error parsing Donut output: {str(e)}")
        result["confidence"] = 0.5
    
    return result


def infer_document_type(fields: Dict[str, Any]) -> str:
    """
    Infer document type based on extracted fields.
    
    Args:
        fields: Dictionary of extracted fields
    
    Returns:
        Inferred document type
    """
    if not fields:
        return "unknown"
    
    # Convert field keys to lowercase for matching
    field_keys = [k.lower() for k in fields.keys()]
    field_text = " ".join(field_keys + [str(v).lower() for v in fields.values()])
    
    # Document type patterns
    document_patterns = {
        "invoice": ["invoice", "bill", "amount", "total", "payment", "due"],
        "receipt": ["receipt", "paid", "transaction", "purchase", "item"],
        "form": ["form", "application", "name", "date", "signature"],
        "contract": ["contract", "agreement", "party", "terms", "signature"],
        "report": ["report", "summary", "analysis", "findings", "conclusion"],
        "certificate": ["certificate", "certify", "issued", "valid", "authority"],
        "letter": ["dear", "sincerely", "regards", "subject", "to:", "from:"],
        "resume": ["experience", "education", "skills", "objective", "references"],
        "statement": ["statement", "balance", "account", "transaction", "period"]
    }
    
    # Score each document type
    scores = {}
    for doc_type, keywords in document_patterns.items():
        score = sum(1 for keyword in keywords if keyword in field_text)
        if score > 0:
            scores[doc_type] = score
    
    # Return the highest scoring type
    if scores:
        return max(scores, key=scores.get)
    
    return "document"  # Generic fallback


def flatten_nested_dict(nested_dict: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten a nested dictionary.
    
    Args:
        nested_dict: Nested dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for concatenated keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_nested_dict(item, f"{new_key}_{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}_{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if donut_model is not None else "unhealthy",
        "service": "donut-processor",
        "model_loaded": donut_model is not None,
        "gpu_available": torch.cuda.is_available()
    }


@app.get("/model-info")
def model_info():
    """Get information about the loaded model."""
    if donut_model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_name": os.getenv("DONUT_MODEL", "naver-clova-ix/donut-base"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "supported_tasks": ["classification", "docvqa", "information_extraction"],
        "max_length": 1024,
        "model_config": {
            "encoder": donut_model.config.encoder.model_type if hasattr(donut_model.config, 'encoder') else "unknown",
            "decoder": donut_model.config.decoder.model_type if hasattr(donut_model.config, 'decoder') else "unknown"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
