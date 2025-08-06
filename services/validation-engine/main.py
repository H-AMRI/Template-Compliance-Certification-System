import os
import base64
import json
import requests
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from dotenv import load_dotenv
import cv2
import numpy as np
from pathlib import Path
import tempfile

from processor import DocumentComplianceProcessor

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost/template_db"
)
TEMPLATE_SERVICE_URL = os.getenv("TEMPLATE_SERVICE_URL", "http://template-manager:80")
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://pdf-generator:80")

app = FastAPI(title="Validation Engine Service", version="1.0.0")

# Initialize processor at startup
processor = DocumentComplianceProcessor()


class ValidationResponse(BaseModel):
    is_compliant: bool
    corrections: list
    pdf_base64: Optional[str] = None
    confidence_score: float
    details: dict


@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup."""
    print("Loading PaddleOCR model...")
    processor.initialize_ocr()
    
    print("Loading LayoutParser model...")
    processor.initialize_layout_model()
    
    print("Loading Donut model...")
    processor.initialize_donut_model()
    
    print("All models loaded successfully")


@app.post("/validate", response_model=ValidationResponse)
async def validate_document(
    template_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Validate a document against a template.
    
    Args:
        template_id: ID of the template to validate against
        file: Document file to validate
    
    Returns:
        Validation results including compliance status and corrections
    """
    temp_file = None
    try:
        # Fetch template rules from Template Manager service
        template_response = requests.get(f"{TEMPLATE_SERVICE_URL}/templates/{template_id}")
        if template_response.status_code != 200:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template_data = template_response.json()
        template_rules = template_data["rules"]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load image
        image = cv2.imread(temp_file_path)
        if image is None:
            # Try to convert PDF to image if needed
            image = processor.pdf_to_image(temp_file_path)
        
        # Extract document features
        layout = processor.extract_layout(image)
        ocr_results = processor.perform_ocr(image)
        donut_results = processor.donut_analysis(image)
        
        # Compare with template
        comparison_results = processor.compare_with_template(
            template_rules, layout, ocr_results, donut_results, image
        )
        
        # Generate compliance report
        report = processor.generate_report(comparison_results)
        
        # If compliant, generate certified PDF
        pdf_base64 = None
        if report["is_compliant"]:
            try:
                # Call PDF generator service
                pdf_response = requests.post(
                    f"{PDF_SERVICE_URL}/generate",
                    json={
                        "document_base64": base64.b64encode(content).decode(),
                        "template_id": template_id,
                        "validation_results": report
                    }
                )
                
                if pdf_response.status_code == 200:
                    pdf_data = pdf_response.json()
                    pdf_base64 = pdf_data.get("pdf_base64")
            except Exception as e:
                print(f"PDF generation failed: {str(e)}")
        
        return ValidationResponse(
            is_compliant=report["is_compliant"],
            corrections=report["corrections"],
            pdf_base64=pdf_base64,
            confidence_score=report["confidence_score"],
            details=report["details"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if file:
            await file.close()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "validation-engine",
        "models_loaded": {
            "ocr": processor.ocr is not None,
            "layout": processor.layout_model is not None,
            "donut": processor.donut_model is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
