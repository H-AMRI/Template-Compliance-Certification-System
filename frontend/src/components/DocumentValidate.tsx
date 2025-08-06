import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  AlertTitle,
  Snackbar,
  Card,
  CardContent,
  Grid,
  Chip,
  IconButton,
  LinearProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon,
  Delete as DeleteIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Download as DownloadIcon,
  Assessment as ValidateIcon,
  PictureAsPdf as PdfIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import Results from './Results';

interface Template {
  id: string;
  name: string;
  rules: any;
  created_at: string;
}

interface ValidationResponse {
  is_compliant: boolean;
  corrections: Array<{
    type: string;
    issue: string;
    score?: number;
    details?: any;
  }>;
  pdf_base64?: string;
  confidence_score: number;
  details: any;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const DocumentValidate: React.FC = () => {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [validating, setValidating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null);
  const [activeStep, setActiveStep] = useState<number>(0);

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    setLoading(true);
    try {
      const response = await axios.get<{ templates: Template[] }>(`${API_BASE_URL}/templates`);
      setTemplates(response.data.templates || []);
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch templates');
      console.error('Error fetching templates:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleValidate = async () => {
    if (!selectedFile || !selectedTemplateId) {
      setError('Please select both a template and a document to validate');
      return;
    }

    setValidating(true);
    setActiveStep(2);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('template_id', selectedTemplateId);

    try {
      const response = await axios.post<ValidationResponse>(
        `${API_BASE_URL}/validate`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setValidationResult(response.data);
      setError(null);
      setActiveStep(3);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Validation failed');
      console.error('Error validating document:', err);
      setActiveStep(1);
    } finally {
      setValidating(false);
    }
  };

  const downloadPDF = () => {
    if (!validationResult?.pdf_base64) return;

    try {
      // Convert base64 to blob
      const byteCharacters = atob(validationResult.pdf_base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'application/pdf' });

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `compliance-certificate-${new Date().toISOString().split('T')[0]}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download PDF');
      console.error('Error downloading PDF:', err);
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
      if (selectedTemplateId) {
        setActiveStep(1);
      }
    }
  }, [selectedTemplateId]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
  });

  const resetForm = () => {
    setSelectedTemplateId('');
    setSelectedFile(null);
    setValidationResult(null);
    setActiveStep(0);
  };

  const steps = [
    'Select Template',
    'Upload Document',
    'Validate',
    'View Results',
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Document Validation
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Stepper activeStep={activeStep} orientation="vertical">
              <Step>
                <StepLabel>Select Template</StepLabel>
                <StepContent>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="template-select-label">Template</InputLabel>
                    <Select
                      labelId="template-select-label"
                      value={selectedTemplateId}
                      label="Template"
                      onChange={(e) => {
                        setSelectedTemplateId(e.target.value);
                        if (selectedFile) {
                          setActiveStep(1);
                        }
                      }}
                      disabled={loading || validating}
                    >
                      <MenuItem value="">
                        <em>Select a template</em>
                      </MenuItem>
                      {templates.map((template) => (
                        <MenuItem key={template.id} value={template.id}>
                          {template.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  {selectedTemplateId && (
                    <Button
                      variant="contained"
                      onClick={() => setActiveStep(1)}
                      sx={{ mt: 1 }}
                    >
                      Continue
                    </Button>
                  )}
                </StepContent>
              </Step>

              <Step>
                <StepLabel>Upload Document</StepLabel>
                <StepContent>
                  <Box
                    {...getRootProps()}
                    sx={{
                      border: '2px dashed',
                      borderColor: isDragActive ? 'primary.main' : 'grey.300',
                      borderRadius: 2,
                      p: 3,
                      textAlign: 'center',
                      cursor: 'pointer',
                      backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                      transition: 'all 0.3s',
                      mb: 2,
                      '&:hover': {
                        borderColor: 'primary.main',
                        backgroundColor: 'action.hover',
                      },
                    }}
                  >
                    <input {...getInputProps()} />
                    <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                    <Typography variant="body1" gutterBottom>
                      {isDragActive
                        ? 'Drop the document here'
                        : 'Drag & drop a document here, or click to select'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Supported formats: Images (PNG, JPG, etc.) and PDF
                    </Typography>
                  </Box>

                  {selectedFile && (
                    <Card sx={{ mb: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <FileIcon sx={{ mr: 1, color: 'primary.main' }} />
                          <Box sx={{ flexGrow: 1 }}>
                            <Typography variant="body2">{selectedFile.name}</Typography>
                            <Typography variant="caption" color="text.secondary">
                              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                            </Typography>
                          </Box>
                          <IconButton
                            size="small"
                            onClick={() => {
                              setSelectedFile(null);
                              setActiveStep(0);
                            }}
                            disabled={validating}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Box>
                      </CardContent>
                    </Card>
                  )}

                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button onClick={() => setActiveStep(0)}>Back</Button>
                    <Button
                      variant="contained"
                      onClick={() => setActiveStep(2)}
                      disabled={!selectedFile}
                    >
                      Continue
                    </Button>
                  </Box>
                </StepContent>
              </Step>

              <Step>
                <StepLabel>Validate Document</StepLabel>
                <StepContent>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <AlertTitle>Ready to Validate</AlertTitle>
                    Document will be validated against the selected template's rules.
                  </Alert>
                  
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button onClick={() => setActiveStep(1)} disabled={validating}>
                      Back
                    </Button>
                    <Button
                      variant="contained"
                      startIcon={validating ? <CircularProgress size={20} /> : <ValidateIcon />}
                      onClick={handleValidate}
                      disabled={validating}
                    >
                      {validating ? 'Validating...' : 'Start Validation'}
                    </Button>
                  </Box>

                  {validating && (
                    <Box sx={{ mt: 2 }}>
                      <LinearProgress />
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                        Analyzing document compliance...
                      </Typography>
                    </Box>
                  )}
                </StepContent>
              </Step>

              <Step>
                <StepLabel>View Results</StepLabel>
                <StepContent>
                  {validationResult && (
                    <Box>
                      <Results
                        is_compliant={validationResult.is_compliant}
                        corrections={validationResult.corrections}
                        confidence_score={validationResult.confidence_score}
                      />
                      
                      {validationResult.is_compliant && validationResult.pdf_base64 && (
                        <Card sx={{ mt: 2, backgroundColor: 'success.light' }}>
                          <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <PdfIcon sx={{ mr: 2, fontSize: 40 }} />
                                <Box>
                                  <Typography variant="subtitle1">
                                    Compliance Certificate Available
                                  </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    Download your digitally signed certificate
                                  </Typography>
                                </Box>
                              </Box>
                              <Button
                                variant="contained"
                                color="success"
                                startIcon={<DownloadIcon />}
                                onClick={downloadPDF}
                              >
                                Download PDF
                              </Button>
                            </Box>
                          </CardContent>
                        </Card>
                      )}
                      
                      <Box sx={{ mt: 3 }}>
                        <Button
                          variant="outlined"
                          onClick={resetForm}
                        >
                          Validate Another Document
                        </Button>
                      </Box>
                    </Box>
                  )}
                </StepContent>
              </Step>
            </Stepper>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Validation Process
            </Typography>
            <Alert severity="info">
              <AlertTitle>How it works</AlertTitle>
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                <li>Select a template that defines the expected format</li>
                <li>Upload the document you want to validate</li>
                <li>Our AI analyzes visual, layout, and text features</li>
                <li>Receive compliance results and corrections if needed</li>
                <li>Download a signed certificate if compliant</li>
              </ol>
            </Alert>
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Analysis Methods
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                <Chip label="Visual Analysis" size="small" color="primary" />
                <Chip label="Layout Detection" size="small" color="primary" />
                <Chip label="OCR" size="small" color="primary" />
                <Chip label="AI Understanding" size="small" color="primary" />
              </Box>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Error Notification */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DocumentValidate;
