import cv2
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR
import layoutparser as lp
import torch


class TemplateAnalyzer:
    """Analyzes document templates to extract validation rules."""
    
    def __init__(self):
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
        
        # Initialize LayoutParser model
        config_path = "lp://detectron2/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
        self.layout_model = lp.Detectron2LayoutModel(
            config_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a template file and extract validation rules.
        
        Args:
            file_path: Path to the template file
            
        Returns:
            Dictionary containing validation rules
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        # Load image
        image = cv2.imread(str(file_path))
        if image is None:
            # Try loading as PDF and convert first page
            image = self._pdf_to_image(file_path)
        
        # Extract features
        visual_features = self._extract_visual_features(image)
        layout_features = self._extract_layout_features(image)
        text_features = self._extract_text_features(image)
        
        # Combine into validation rules
        rules = {
            "visual": visual_features,
            "layout": layout_features,
            "text": text_features,
            "metadata": {
                "file_name": file_path.name,
                "image_shape": image.shape[:2],
                "analysis_version": "1.0.0"
            }
        }
        
        return rules
    
    def _pdf_to_image(self, pdf_path: Path) -> np.ndarray:
        """Convert first page of PDF to image."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
            if images:
                return np.array(images[0])
        except ImportError:
            pass
        
        # Fallback: create blank image
        return np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    def _extract_visual_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract visual features for template matching."""
        # Convert to grayscale for SSIM
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image hash for quick comparison
        resized = cv2.resize(gray, (8, 8))
        avg = resized.mean()
        hash_bits = (resized > avg).flatten()
        image_hash = ''.join(['1' if bit else '0' for bit in hash_bits])
        
        # Calculate color histogram
        hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_b = cv2.normalize(hist_b, hist_b).flatten().tolist()
        hist_g = cv2.normalize(hist_g, hist_g).flatten().tolist()
        hist_r = cv2.normalize(hist_r, hist_r).flatten().tolist()
        
        # Extract key points for structural matching
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        keypoint_data = []
        if keypoints:
            for kp in keypoints[:100]:  # Limit to top 100 keypoints
                keypoint_data.append({
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                    "angle": float(kp.angle)
                })
        
        return {
            "image_hash": image_hash,
            "color_histogram": {
                "blue": hist_b,
                "green": hist_g,
                "red": hist_r
            },
            "keypoints": keypoint_data,
            "ssim_reference": True,  # Flag to use SSIM comparison
            "min_ssim_threshold": 0.85
        }
    
    def _extract_layout_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract layout structure using LayoutParser."""
        # Convert to PIL Image for LayoutParser
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect layout
        layout = self.layout_model.detect(pil_image)
        
        # Extract layout blocks
        layout_blocks = []
        for block in layout:
            layout_blocks.append({
                "type": block.type,
                "bbox": {
                    "x1": float(block.block.x_1),
                    "y1": float(block.block.y_1),
                    "x2": float(block.block.x_2),
                    "y2": float(block.block.y_2)
                },
                "confidence": float(block.score) if hasattr(block, 'score') else 1.0
            })
        
        # Calculate layout signature
        layout_signature = self._calculate_layout_signature(layout_blocks, image.shape[:2])
        
        return {
            "blocks": layout_blocks,
            "block_count": len(layout_blocks),
            "layout_signature": layout_signature,
            "expected_regions": self._define_expected_regions(layout_blocks)
        }
    
    def _extract_text_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text content and positions using PaddleOCR."""
        # Run OCR
        result = self.ocr.ocr(image, cls=True)
        
        text_blocks = []
        all_text = []
        
        if result and result[0]:
            for detection in result[0]:
                bbox = detection[0]
                text = detection[1][0]
                confidence = detection[1][1]
                
                # Calculate normalized bounding box
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                text_block = {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(min(x_coords)),
                        "y1": float(min(y_coords)),
                        "x2": float(max(x_coords)),
                        "y2": float(max(y_coords))
                    },
                    "is_required": confidence > 0.9  # High confidence text is likely required
                }
                
                text_blocks.append(text_block)
                all_text.append(text)
        
        # Identify key fields (simplified heuristic)
        key_fields = self._identify_key_fields(text_blocks)
        
        return {
            "text_blocks": text_blocks,
            "total_text_count": len(text_blocks),
            "key_fields": key_fields,
            "text_content_hash": hashlib.md5(' '.join(all_text).encode()).hexdigest()
        }
    
    def _calculate_layout_signature(self, blocks: List[Dict], image_shape: Tuple[int, int]) -> str:
        """Generate a signature representing the layout structure."""
        if not blocks:
            return "empty"
        
        height, width = image_shape
        
        # Create grid representation
        grid_size = 10
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        for block in blocks:
            bbox = block["bbox"]
            # Normalize coordinates
            x1 = int((bbox["x1"] / width) * grid_size)
            y1 = int((bbox["y1"] / height) * grid_size)
            x2 = min(int((bbox["x2"] / width) * grid_size), grid_size - 1)
            y2 = min(int((bbox["y2"] / height) * grid_size), grid_size - 1)
            
            # Mark grid cells
            grid[y1:y2+1, x1:x2+1] = 1
        
        # Convert to signature string
        signature = ''.join(str(cell) for row in grid for cell in row)
        return hashlib.md5(signature.encode()).hexdigest()
    
    def _define_expected_regions(self, blocks: List[Dict]) -> List[Dict]:
        """Define expected regions based on detected layout."""
        regions = []
        
        for block in blocks:
            if block["confidence"] > 0.8:
                regions.append({
                    "type": block["type"],
                    "bbox": block["bbox"],
                    "tolerance": 50,  # Pixel tolerance for matching
                    "required": True
                })
        
        return regions
    
    def _identify_key_fields(self, text_blocks: List[Dict]) -> List[Dict]:
        """Identify potential key fields in the document."""
        key_fields = []
        
        # Common field indicators
        field_indicators = ["name", "date", "number", "id", "code", "amount", "total", "signature"]
        
        for block in text_blocks:
            text_lower = block["text"].lower()
            
            # Check if text contains field indicators
            for indicator in field_indicators:
                if indicator in text_lower:
                    key_fields.append({
                        "field_type": indicator,
                        "text": block["text"],
                        "bbox": block["bbox"],
                        "required": True
                    })
                    break
        
        return key_fields
