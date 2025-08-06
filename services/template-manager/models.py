from sqlalchemy import Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Template(Base):
    __tablename__ = "templates"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    rules = Column(JSON, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Template(id={self.id}, name={self.name})>"
