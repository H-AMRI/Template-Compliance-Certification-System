import cv2
import numpy as np
import torch
import json
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import imagehash
from paddleocr import PaddleOCR
import layoutparser as lp
from transformers import DonutProcessor as HFDonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path


class DocumentComplianceProcessor:
    """Processes documents for compliance validation against templates."""
    
    def __init__(self):
        self.ocr = None
        self.layout_model = None
        self.donut_model = None
        self.donut_processor = None
    
    def initialize_ocr(self):
        """Initialize PaddleOCR model."""
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=torch.cuda.is_available()
        )
    
    def initialize_layout_model(self):
        """Initialize LayoutParser model."""
        config_path = "lp://detectron2/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
        self.layout_model = lp.Detectron2LayoutModel(
            config_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
    
    def initialize_donut_model(self):
        """Initialize Donut model for document understanding."""
        model_name = "naver-clova-ix/donut-base"
        self.donut_processor = HFDonutProcessor.from_pretrained(model_name)
        self.donut_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.donut_model = self.donut_model.cuda()
        
        self.donut_model.eval()
    
    def pdf_to_image(self, pdf_path: str) -> np.ndarray:
        """Convert first page of PDF to image."""
        try:
            images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
            if images:
                return np.array(images[0])
        except Exception as e:
            print(f"PDF conversion error: {str(e)}")
        
        # Return blank image as fallback
        return np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    def extract_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract layout structure from document image."""
        if self.layout_model is None:
            return {"blocks": [], "error": "Layout model not initialized"}
        
        # Convert to PIL Image
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
                "confidence": float(block.score) if hasattr(block, 'score') else 1.0,
                "area": float((block.block.x_2 - block.block.x_1) * 
                             (block.block.y_2 - block.block.y_1))
            })
        
        # Sort blocks by position (top to bottom, left to right)
        layout_blocks.sort(key=lambda b: (b["bbox"]["y1"], b["bbox"]["x1"]))
        
        return {
            "blocks": layout_blocks,
            "total_blocks": len(layout_blocks),
            "layout_hash": self._calculate_layout_hash(layout_blocks, image.shape[:2])
        }
    
    def perform_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform OCR on document image."""
        if self.ocr is None:
            return {"text_blocks": [], "error": "OCR model not initialized"}
        
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
                    "word_count": len(text.split())
                }
                
                text_blocks.append(text_block)
                all_text.append(text)
        
        return {
            "text_blocks": text_blocks,
            "total_text": " ".join(all_text),
            "text_count": len(text_blocks),
            "avg_confidence": np.mean([b["confidence"] for b in text_blocks]) if text_blocks else 0
        }
    
    def donut_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform document understanding using Donut model."""
        if self.donut_model is None or self.donut_processor is None:
            return {"structured_data": {}, "error": "Donut model not initialized"}
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare input
            pixel_values = self.donut_processor(pil_image, return_tensors="pt").pixel_values
            
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            
            # Generate output
            with torch.no_grad():
                task_prompt = "<s_docvqa><s_question>What are the key fields in this document?</s_question><s_answer>"
                decoder_input_ids = self.donut_processor.tokenizer(
                    task_prompt,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids
                
                if torch.cuda.is_available():
                    decoder_input_ids = decoder_input_ids.cuda()
                
                outputs = self.donut_model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,
                    early_stopping=True,
                    pad_token_id=self.donut_processor.tokenizer.pad_token_id,
                    eos_token_id=self.donut_processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[self.donut_processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True
                )
            
            # Decode output
            decoded = self.donut_processor.batch_decode(outputs.sequences)[0]
            
            # Parse structured data
            structured_data = self._parse_donut_output(decoded)
            
            return {
                "structured_data": structured_data,
                "raw_output": decoded,
                "confidence": 0.85  # Placeholder confidence
            }
            
        except Exception as e:
            return {
                "structured_data": {},
                "error": f"Donut analysis failed: {str(e)}"
            }
    
    def compare_with_template(
        self,
        rules: Dict[str, Any],
        layout: Dict[str, Any],
        ocr_results: Dict[str, Any],
        donut_results: Dict[str, Any],
        image: np.ndarray
    ) -> Dict[str, Any]:
        """Compare document features with template rules."""
        
        comparison = {
            "visual_match": self._compare_visual(rules.get("visual", {}), image),
            "layout_match": self._compare_layout(rules.get("layout", {}), layout),
            "text_match": self._compare_text(rules.get("text", {}), ocr_results),
            "structural_match": self._compare_structure(rules, donut_results),
            "corrections_needed": []
        }
        
        # Identify corrections needed
        if comparison["visual_match"]["score"] < 0.8:
            comparison["corrections_needed"].append({
                "type": "visual",
                "issue": "Document visual appearance doesn't match template",
                "score": comparison["visual_match"]["score"],
                "details": comparison["visual_match"]["details"]
            })
        
        if comparison["layout_match"]["score"] < 0.75:
            comparison["corrections_needed"].append({
                "type": "layout",
                "issue": "Document layout structure doesn't match template",
                "score": comparison["layout_match"]["score"],
                "missing_blocks": comparison["layout_match"].get("missing_blocks", [])
            })
        
        if comparison["text_match"]["score"] < 0.85:
            comparison["corrections_needed"].append({
                "type": "text",
                "issue": "Required text fields missing or incorrect",
                "score": comparison["text_match"]["score"],
                "missing_fields": comparison["text_match"].get("missing_fields", [])
            })
        
        # Calculate overall compliance score
        weights = {"visual": 0.25, "layout": 0.25, "text": 0.35, "structural": 0.15}
        overall_score = (
            comparison["visual_match"]["score"] * weights["visual"] +
            comparison["layout_match"]["score"] * weights["layout"] +
            comparison["text_match"]["score"] * weights["text"] +
            comparison["structural_match"]["score"] * weights["structural"]
        )
        
        comparison["overall_score"] = overall_score
        comparison["is_compliant"] = overall_score >= 0.8 and len(comparison["corrections_needed"]) == 0
        
        return comparison
    
    def generate_report(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance validation report."""
        
        report = {
            "is_compliant": comparison["is_compliant"],
            "confidence_score": comparison["overall_score"],
            "corrections": comparison["corrections_needed"],
            "details": {
                "visual_analysis": {
                    "score": comparison["visual_match"]["score"],
                    "passed": comparison["visual_match"]["score"] >= 0.8,
                    "details": comparison["visual_match"].get("details", {})
                },
                "layout_analysis": {
                    "score": comparison["layout_match"]["score"],
                    "passed": comparison["layout_match"]["score"] >= 0.75,
                    "matched_blocks": comparison["layout_match"].get("matched_blocks", 0),
                    "total_blocks": comparison["layout_match"].get("total_blocks", 0)
                },
                "text_analysis": {
                    "score": comparison["text_match"]["score"],
                    "passed": comparison["text_match"]["score"] >= 0.85,
                    "matched_fields": comparison["text_match"].get("matched_fields", 0),
                    "total_fields": comparison["text_match"].get("total_fields", 0)
                },
                "structural_analysis": {
                    "score": comparison["structural_match"]["score"],
                    "passed": comparison["structural_match"]["score"] >= 0.7
                }
            },
            "recommendations": self._generate_recommendations(comparison)
        }
        
        return report
    
    def _calculate_layout_hash(self, blocks: List[Dict], image_shape: Tuple[int, int]) -> str:
        """Calculate hash representing layout structure."""
        if not blocks:
            return "empty"
        
        height, width = image_shape
        
        # Create simplified grid representation
        grid_size = 8
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        for block in blocks:
            bbox = block["bbox"]
            x1 = int((bbox["x1"] / width) * grid_size)
            y1 = int((bbox["y1"] / height) * grid_size)
            x2 = min(int((bbox["x2"] / width) * grid_size), grid_size - 1)
            y2 = min(int((bbox["y2"] / height) * grid_size), grid_size - 1)
            
            grid[y1:y2+1, x1:x2+1] = 1
        
        # Convert to hash
        signature = ''.join(str(cell) for row in grid for cell in row)
        return hashlib.md5(signature.encode()).hexdigest()
    
    def _parse_donut_output(self, decoded: str) -> Dict[str, Any]:
        """Parse Donut model output into structured data."""
        structured = {}
        
        try:
            # Extract key-value pairs from the decoded text
            # This is a simplified parser - adjust based on actual Donut output format
            if "<s_answer>" in decoded:
                answer_part = decoded.split("<s_answer>")[1].split("</s_answer>")[0]
                
                # Simple key-value extraction
                lines = answer_part.strip().split("\n")
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        structured[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error parsing Donut output: {str(e)}")
        
        return structured
    
    def _compare_visual(self, template_visual: Dict, image: np.ndarray) -> Dict[str, Any]:
        """Compare visual features between document and template."""
        score = 1.0
        details = {}
        
        if not template_visual:
            return {"score": score, "details": details}
        
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compare image hash
        if "image_hash" in template_visual:
            resized = cv2.resize(gray, (8, 8))
            avg = resized.mean()
            hash_bits = (resized > avg).flatten()
            doc_hash = ''.join(['1' if bit else '0' for bit in hash_bits])
            
            # Calculate Hamming distance
            template_hash = template_visual["image_hash"]
            distance = sum(c1 != c2 for c1, c2 in zip(doc_hash, template_hash))
            hash_similarity = 1 - (distance / len(template_hash))
            
            score *= hash_similarity
            details["hash_similarity"] = hash_similarity
        
        # Compare color histograms
        if "color_histogram" in template_visual:
            hist_scores = []
            for channel, idx in [("blue", 0), ("green", 1), ("red", 2)]:
                if channel in template_visual["color_histogram"]:
                    hist = cv2.calcHist([image], [idx], None, [32], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    
                    template_hist = np.array(template_visual["color_histogram"][channel])
                    correlation = cv2.compareHist(hist.astype(np.float32), 
                                                 template_hist.astype(np.float32), 
                                                 cv2.HISTCMP_CORREL)
                    hist_scores.append(correlation)
            
            if hist_scores:
                hist_similarity = np.mean(hist_scores)
                score *= hist_similarity
                details["histogram_similarity"] = hist_similarity
        
        # SSIM comparison would require the original template image
        # Using threshold as proxy
        if template_visual.get("ssim_reference") and template_visual.get("min_ssim_threshold"):
            # Simplified: use threshold as score component
            score *= template_visual["min_ssim_threshold"]
            details["ssim_threshold"] = template_visual["min_ssim_threshold"]
        
        return {"score": float(score), "details": details}
    
    def _compare_layout(self, template_layout: Dict, doc_layout: Dict) -> Dict[str, Any]:
        """Compare layout structures between document and template."""
        if not template_layout or not doc_layout:
            return {"score": 1.0, "matched_blocks": 0, "total_blocks": 0}
        
        template_blocks = template_layout.get("blocks", [])
        doc_blocks = doc_layout.get("blocks", [])
        
        if not template_blocks:
            return {"score": 1.0, "matched_blocks": len(doc_blocks), "total_blocks": len(doc_blocks)}
        
        matched = 0
        missing_blocks = []
        
        for t_block in template_blocks:
            # Find matching block in document
            found = False
            for d_block in doc_blocks:
                if t_block["type"] == d_block["type"]:
                    # Check spatial proximity
                    if self._blocks_overlap(t_block["bbox"], d_block["bbox"], tolerance=50):
                        matched += 1
                        found = True
                        break
            
            if not found:
                missing_blocks.append(t_block["type"])
        
        score = matched / len(template_blocks) if template_blocks else 0
        
        return {
            "score": float(score),
            "matched_blocks": matched,
            "total_blocks": len(template_blocks),
            "missing_blocks": missing_blocks
        }
    
    def _compare_text(self, template_text: Dict, ocr_results: Dict) -> Dict[str, Any]:
        """Compare text content between document and template."""
        if not template_text:
            return {"score": 1.0, "matched_fields": 0, "total_fields": 0}
        
        key_fields = template_text.get("key_fields", [])
        doc_text = ocr_results.get("total_text", "").lower()
        doc_blocks = ocr_results.get("text_blocks", [])
        
        if not key_fields:
            return {"score": 1.0, "matched_fields": 0, "total_fields": 0}
        
        matched = 0
        missing_fields = []
        
        for field in key_fields:
            field_text = field.get("text", "").lower()
            field_type = field.get("field_type", "")
            
            # Check if field text appears in document
            found = False
            if field_text in doc_text:
                found = True
            else:
                # Check for partial matches
                for block in doc_blocks:
                    if field_type in block["text"].lower():
                        found = True
                        break
            
            if found:
                matched += 1
            else:
                missing_fields.append(field_type or field_text)
        
        score = matched / len(key_fields) if key_fields else 0
        
        return {
            "score": float(score),
            "matched_fields": matched,
            "total_fields": len(key_fields),
            "missing_fields": missing_fields
        }
    
    def _compare_structure(self, rules: Dict, donut_results: Dict) -> Dict[str, Any]:
        """Compare document structure using Donut analysis."""
        # Simplified structural comparison
        structured_data = donut_results.get("structured_data", {})
        
        if not structured_data:
            return {"score": 0.7}  # Default score if no structured data
        
        # Basic scoring based on presence of structured data
        score = min(1.0, 0.7 + (len(structured_data) * 0.05))
        
        return {"score": float(score)}
    
    def _blocks_overlap(self, bbox1: Dict, bbox2: Dict, tolerance: int = 50) -> bool:
        """Check if two bounding boxes overlap within tolerance."""
        # Expand bbox2 by tolerance
        expanded = {
            "x1": bbox2["x1"] - tolerance,
            "y1": bbox2["y1"] - tolerance,
            "x2": bbox2["x2"] + tolerance,
            "y2": bbox2["y2"] + tolerance
        }
        
        # Check for overlap
        return not (bbox1["x2"] < expanded["x1"] or 
                   bbox1["x1"] > expanded["x2"] or 
                   bbox1["y2"] < expanded["y1"] or 
                   bbox1["y1"] > expanded["y2"])
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if comparison["visual_match"]["score"] < 0.8:
            recommendations.append(
                "Ensure document is scanned at high quality and matches template formatting"
            )
        
        if comparison["layout_match"]["score"] < 0.75:
            recommendations.append(
                "Verify all required sections are present and properly positioned"
            )
        
        if comparison["text_match"]["score"] < 0.85:
            missing = comparison["text_match"].get("missing_fields", [])
            if missing:
                recommendations.append(
                    f"Add missing required fields: {', '.join(missing[:3])}"
                )
            else:
                recommendations.append(
                    "Ensure all text fields are clearly visible and properly filled"
                )
        
        if comparison["is_compliant"]:
            recommendations.append("Document meets all compliance requirements")
        
        return recommendations
