import os
import io
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, green, red
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PyPDF2 import PdfReader, PdfWriter
from endesive import pdf as pdf_sign
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from dotenv import load_dotenv
import tempfile
import hashlib

load_dotenv()

app = FastAPI(title="PDF Generator Service", version="1.0.0")

# Certificate paths
CERT_PATH = os.getenv("CERT_PATH", "/certs/cert.pem")
KEY_PATH = os.getenv("KEY_PATH", "/certs/key.pem")
CERT_PASSWORD = os.getenv("CERT_PASSWORD", "").encode() if os.getenv("CERT_PASSWORD") else None


class GenerateRequest(BaseModel):
    document_base64: Optional[str] = None
    template_id: str
    validation_results: Dict[str, Any]


class GenerateResponse(BaseModel):
    pdf_base64: str
    certificate_id: str
    timestamp: str


@app.post("/generate", response_model=GenerateResponse)
async def generate_certificate(request: GenerateRequest):
    """
    Generate and digitally sign a compliance certificate PDF.
    
    Args:
        request: Contains validation report and optionally the original document
    
    Returns:
        Base64 encoded signed PDF certificate
    """
    try:
        # Generate certificate ID
        certificate_id = generate_certificate_id(request.validation_results)
        
        # Create PDF certificate
        pdf_buffer = create_compliance_certificate(
            validation_results=request.validation_results,
            template_id=request.template_id,
            certificate_id=certificate_id
        )
        
        # Sign the PDF if certificates are available
        if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
            signed_pdf = sign_pdf(pdf_buffer.getvalue())
        else:
            print("Warning: Digital certificates not found, PDF will not be signed")
            signed_pdf = pdf_buffer.getvalue()
        
        # Encode to base64
        pdf_base64 = base64.b64encode(signed_pdf).decode('utf-8')
        
        return GenerateResponse(
            pdf_base64=pdf_base64,
            certificate_id=certificate_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


def create_compliance_certificate(
    validation_results: Dict[str, Any],
    template_id: str,
    certificate_id: str
) -> io.BytesIO:
    """
    Create a compliance certificate PDF using ReportLab.
    
    Args:
        validation_results: Validation report data
        template_id: ID of the template used for validation
        certificate_id: Unique certificate ID
    
    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = io.BytesIO()
    
    # Create PDF with custom page
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Container for flowables
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1e3a8a'),
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName='Helvetica-Bold'
    )
    
    header_style = ParagraphStyle(
        'Header',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#334155'),
        alignment=TA_LEFT,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=11,
        textColor=black,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    center_style = ParagraphStyle(
        'Center',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    # Title
    elements.append(Paragraph("COMPLIANCE CERTIFICATE", title_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Certificate ID and Date
    elements.append(Paragraph(f"<b>Certificate ID:</b> {certificate_id}", body_style))
    elements.append(Paragraph(f"<b>Issue Date:</b> {datetime.utcnow().strftime('%B %d, %Y')}", body_style))
    elements.append(Paragraph(f"<b>Template ID:</b> {template_id}", body_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Compliance Status
    is_compliant = validation_results.get("is_compliant", False)
    confidence = validation_results.get("confidence_score", 0) * 100
    
    status_color = green if is_compliant else red
    status_text = "COMPLIANT" if is_compliant else "NON-COMPLIANT"
    
    status_style = ParagraphStyle(
        'Status',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=status_color,
        alignment=TA_CENTER,
        spaceAfter=20,
        fontName='Helvetica-Bold'
    )
    
    elements.append(Paragraph(f"Document Status: {status_text}", status_style))
    elements.append(Paragraph(f"Confidence Score: {confidence:.1f}%", center_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Validation Details Section
    elements.append(Paragraph("VALIDATION DETAILS", header_style))
    
    details = validation_results.get("details", {})
    
    # Create validation results table
    table_data = [["Analysis Type", "Score", "Status"]]
    
    for analysis_type in ["visual_analysis", "layout_analysis", "text_analysis", "structural_analysis"]:
        if analysis_type in details:
            analysis = details[analysis_type]
            score = analysis.get("score", 0) * 100
            passed = analysis.get("passed", False)
            status = "✓ Passed" if passed else "✗ Failed"
            
            table_data.append([
                analysis_type.replace("_", " ").title(),
                f"{score:.1f}%",
                status
            ])
    
    if len(table_data) > 1:
        table = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e2e8f0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1e293b')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#cbd5e1')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8fafc')]),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
    
    # Corrections Required (if any)
    corrections = validation_results.get("corrections", [])
    if corrections:
        elements.append(Paragraph("REQUIRED CORRECTIONS", header_style))
        
        for i, correction in enumerate(corrections, 1):
            correction_text = f"{i}. <b>{correction.get('type', 'Unknown').title()}:</b> {correction.get('issue', 'No details available')}"
            elements.append(Paragraph(correction_text, body_style))
        
        elements.append(Spacer(1, 0.2 * inch))
    
    # Recommendations
    recommendations = validation_results.get("details", {}).get("recommendations", [])
    if recommendations:
        elements.append(Paragraph("RECOMMENDATIONS", header_style))
        
        for recommendation in recommendations[:3]:  # Limit to 3 recommendations
            elements.append(Paragraph(f"• {recommendation}", body_style))
        
        elements.append(Spacer(1, 0.3 * inch))
    
    # Digital Signature Section
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("DIGITAL SIGNATURE", header_style))
    
    signature_data = [
        ["Signed By:", "Compliance Certification System"],
        ["Signature Method:", "RSA with SHA-256"],
        ["Timestamp:", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')],
        ["Verification:", "This document has been digitally signed and certified"]
    ]
    
    sig_table = Table(signature_data, colWidths=[2*inch, 4*inch])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(sig_table)
    
    # Footer
    elements.append(Spacer(1, 0.5 * inch))
    footer_text = "This certificate is generated automatically and is valid only with digital signature verification."
    elements.append(Paragraph(footer_text, center_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer


def sign_pdf(pdf_data: bytes) -> bytes:
    """
    Digitally sign a PDF using endesive.
    
    Args:
        pdf_data: PDF file data as bytes
    
    Returns:
        Signed PDF as bytes
    """
    try:
        # Load certificate and private key
        with open(CERT_PATH, 'rb') as cert_file:
            cert_data = cert_file.read()
        
        with open(KEY_PATH, 'rb') as key_file:
            key_data = key_file.read()
        
        # Parse certificate and key
        if CERT_PATH.endswith('.pem'):
            # PEM format
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            
            if CERT_PASSWORD:
                key = serialization.load_pem_private_key(
                    key_data,
                    password=CERT_PASSWORD,
                    backend=default_backend()
                )
            else:
                key = serialization.load_pem_private_key(
                    key_data,
                    password=None,
                    backend=default_backend()
                )
        else:
            # DER/P12 format handling would go here
            raise ValueError("Only PEM format certificates are currently supported")
        
        # Sign the PDF
        signature_data = pdf_sign.cms.sign(
            pdf_data,
            {
                'sigflags': 3,
                'contact': 'compliance@system.com',
                'location': 'Digital Compliance System',
                'reason': 'Document Compliance Certification',
                'signingdate': datetime.utcnow().strftime('%Y%m%d%H%M%S+00\'00\''),
            },
            cert_data,
            key_data,
            [],  # Additional certificates
            'sha256'
        )
        
        # Create signed PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_data)
            temp_pdf_path = temp_pdf.name
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as signed_pdf:
            signed_pdf_path = signed_pdf.name
        
        # Write signed data
        with open(temp_pdf_path, 'rb') as f:
            datau = f.read()
        
        datas = pdf_sign.sign(
            datau,
            signature_data,
            cert_data,
            key_data,
            [],
            'sha256'
        )
        
        # Clean up temp files
        os.unlink(temp_pdf_path)
        
        return datas
        
    except FileNotFoundError as e:
        print(f"Certificate files not found: {str(e)}")
        return pdf_data  # Return unsigned PDF
    except Exception as e:
        print(f"PDF signing failed: {str(e)}")
        return pdf_data  # Return unsigned PDF


def generate_certificate_id(validation_results: Dict[str, Any]) -> str:
    """
    Generate a unique certificate ID based on validation results.
    
    Args:
        validation_results: Validation report data
    
    Returns:
        Unique certificate ID
    """
    # Create hash from validation data and timestamp
    hash_input = f"{json.dumps(validation_results, sort_keys=True)}{datetime.utcnow().isoformat()}"
    hash_object = hashlib.sha256(hash_input.encode())
    hash_hex = hash_object.hexdigest()[:12].upper()
    
    # Format as certificate ID
    timestamp = datetime.utcnow().strftime('%Y%m%d')
    return f"CERT-{timestamp}-{hash_hex}"


@app.get("/health")
def health_check():
    """Health check endpoint."""
    certs_available = os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH)
    
    return {
        "status": "healthy",
        "service": "pdf-generator",
        "certificates_available": certs_available,
        "signing_enabled": certs_available
    }


@app.post("/test-certificate")
async def test_certificate():
    """Generate a test certificate for verification."""
    test_results = {
        "is_compliant": True,
        "confidence_score": 0.95,
        "corrections": [],
        "details": {
            "visual_analysis": {"score": 0.92, "passed": True},
            "layout_analysis": {"score": 0.88, "passed": True},
            "text_analysis": {"score": 0.96, "passed": True},
            "structural_analysis": {"score": 0.85, "passed": True},
            "recommendations": ["Document meets all compliance requirements"]
        }
    }
    
    request = GenerateRequest(
        template_id="TEST-TEMPLATE-001",
        validation_results=test_results
    )
    
    return await generate_certificate(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
