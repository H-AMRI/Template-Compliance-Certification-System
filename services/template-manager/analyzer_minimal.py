import hashlib
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class TemplateAnalyzer:
    """Minimal analyzer for testing without ML dependencies."""
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Mock analyze function for testing."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        # Create mock rules for testing
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        return {
            "visual": {
                "image_hash": file_hash,
                "min_ssim_threshold": 0.85
            },
            "layout": {
                "blocks": [],
                "block_count": 0
            },
            "text": {
                "text_blocks": [],
                "total_text_count": 0,
                "key_fields": []
            },
            "metadata": {
                "file_name": file_path.name,
                "analysis_version": "1.0.0-minimal",
                "analyzed_at": datetime.utcnow().isoformat()
            }
        }
