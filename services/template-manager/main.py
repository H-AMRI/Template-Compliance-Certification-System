import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil
from pathlib import Path

from models import Base, Template
from analyzer import TemplateAnalyzer

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost/template_db"
)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="Template Manager Service", version="1.0.0")

analyzer = TemplateAnalyzer()


class TemplateResponse(BaseModel):
    id: str
    name: str
    rules: dict
    created_at: datetime


class TemplateListResponse(BaseModel):
    templates: List[TemplateResponse]


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/templates", response_model=TemplateResponse)
async def upload_template(
    name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a template file, analyze it, and store rules in database."""
    try:
        # Generate unique ID
        template_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{template_id}{file_extension}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze template to extract rules
        try:
            rules = analyzer.analyze(str(file_path))
        except Exception as e:
            # Clean up file if analysis fails
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        # Store in database
        template = Template(
            id=template_id,
            name=name,
            rules=rules,
            file_path=str(file_path)
        )
        
        db.add(template)
        db.commit()
        db.refresh(template)
        
        return TemplateResponse(
            id=template.id,
            name=template.name,
            rules=template.rules,
            created_at=template.created_at
        )
        
    except SQLAlchemyError as e:
        db.rollback()
        # Clean up file if database operation fails
        if 'file_path' in locals():
            Path(file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


@app.get("/templates", response_model=TemplateListResponse)
def list_templates(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all templates with pagination."""
    try:
        templates = db.query(Template).offset(skip).limit(limit).all()
        
        template_list = [
            TemplateResponse(
                id=t.id,
                name=t.name,
                rules=t.rules,
                created_at=t.created_at
            )
            for t in templates
        ]
        
        return TemplateListResponse(templates=template_list)
        
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/templates/{template_id}", response_model=TemplateResponse)
def get_template(template_id: str, db: Session = Depends(get_db)):
    """Retrieve metadata and rules for a specific template."""
    try:
        template = db.query(Template).filter(Template.id == template_id).first()
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return TemplateResponse(
            id=template.id,
            name=template.name,
            rules=template.rules,
            created_at=template.created_at
        )
        
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "template-manager"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
