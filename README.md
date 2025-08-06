# Template Compliance Certification System

## Overview

The Template Compliance Certification System is a cloud-native, AI-powered document validation platform that ensures documents comply with predefined templates. Built with a microservices architecture, it leverages state-of-the-art computer vision and NLP models to perform comprehensive document analysis.

### Key Features

- **Template Management**: Upload and manage document templates with automatic rule extraction
- **Multi-Modal Analysis**: Visual, layout, text, and structural validation using AI/ML
- **Compliance Certification**: Generate digitally signed PDF certificates for compliant documents
- **Real-time Validation**: Fast, accurate document validation against templates
- **Scalable Architecture**: Microservices design with GPU acceleration support

### Technology Stack

- **Backend**: Python 3.10, FastAPI
- **Frontend**: React 18, TypeScript, Material-UI
- **ML/AI**: PaddleOCR, LayoutParser, Donut, Detectron2
- **Database**: PostgreSQL 13
- **Cache**: Redis
- **Container**: Docker, Kubernetes (GKE)
- **Proxy**: Nginx

## System Architecture

┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Frontend  │────▶│     Nginx       │────▶│  Microservices   │
│   (React)   │     │  (Reverse Proxy)│     │                  │
└─────────────┘     └─────────────────┘     │ ┌──────────────┐ │
│ │Template Mgr  │ │
│ └──────────────┘ │
│ ┌──────────────┐ │
│ │Validation    │ │
│ │Engine        │ │
│ └──────────────┘ │
│ ┌──────────────┐ │
│ │Donut         │ │
│ │Processor     │ │
│ └──────────────┘ │
│ ┌──────────────┐ │
│ │PDF Generator │ │
│ └──────────────┘ │
└──────────────────┘
│
┌────────▼────────┐
│   PostgreSQL    │
│     Redis       │
└─────────────────┘

## Quick Start (Local Development)

### Prerequisites

- Docker Desktop with Docker Compose
- 8GB+ RAM available for Docker
- (Optional) NVIDIA GPU with CUDA support for accelerated ML inference
- (Optional) SSL certificates for HTTPS

### Running Locally

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd template-compliance-system

2. Set up environment variables

# Create .env file in project root
cat > .env << EOF
POSTGRES_USER=compliance_user
POSTGRES_PASSWORD=compliance_pass
POSTGRES_DB=compliance_db
REDIS_URL=redis://redis:6379
EOF

3. Generate SSL certificates (optional, for HTTPS)

mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/key.pem \
  -out certs/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

4. Build and run all services

docker-compose up --build

5. Access the application

Frontend: http://localhost
API Documentation:

Template Manager: http://localhost:8001/docs
Validation Engine: http://localhost:8002/docs
Donut Processor: http://localhost:8003/docs
PDF Generator: http://localhost:8004/docs

Development Commands
# Build specific service
docker-compose build template-manager

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f validation-engine

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Scale a service
docker-compose up --scale validation-engine=3
