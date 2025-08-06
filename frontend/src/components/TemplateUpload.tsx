import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  CircularProgress,
  Alert,
  AlertTitle,
  Snackbar,
  Card,
  CardContent,
  CardActions,
  Grid,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  LinearProgress,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

interface Template {
  id: string;
  name: string;
  rules: any;
  created_at: string;
}

interface UploadResponse {
  id: string;
  name: string;
  rules: any;
  created_at: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const TemplateUpload: React.FC = () => {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [templateName, setTemplateName] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);
  const [detailsOpen, setDetailsOpen] = useState<boolean>(false);

  // Fetch templates on mount
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

  const handleUpload = async () => {
    if (!selectedFile || !templateName.trim()) {
      setError('Please provide both a template name and file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('name', templateName.trim());

    try {
      const response = await axios.post<UploadResponse>(
        `${API_BASE_URL}/templates`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setSuccess(`Template "${response.data.name}" uploaded successfully!`);
      setTemplates([response.data, ...templates]);
      setTemplateName('');
      setSelectedFile(null);
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload template');
      console.error('Error uploading template:', err);
    } finally {
      setUploading(false);
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
  });

  const handleViewDetails = (template: Template) => {
    setSelectedTemplate(template);
    setDetailsOpen(true);
  };

  const handleCloseDetails = () => {
    setDetailsOpen(false);
    setSelectedTemplate(null);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Template Management
      </Typography>

      {/* Upload Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Upload New Template
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Template Name"
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
              placeholder="Enter template name"
              disabled={uploading}
              sx={{ mb: 2 }}
            />
            
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
                  ? 'Drop the template file here'
                  : 'Drag & drop a template file here, or click to select'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Supported formats: Images (PNG, JPG, etc.) and PDF
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            {selectedFile && (
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Selected File
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <FileIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="body2">{selectedFile.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </Typography>
                    </Box>
                    <IconButton
                      size="small"
                      onClick={() => setSelectedFile(null)}
                      disabled={uploading}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </CardContent>
                <CardActions>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={uploading ? <CircularProgress size={20} /> : <UploadIcon />}
                    onClick={handleUpload}
                    disabled={uploading || !templateName.trim()}
                  >
                    {uploading ? 'Uploading...' : 'Upload Template'}
                  </Button>
                </CardActions>
              </Card>
            )}
            
            {!selectedFile && (
              <Alert severity="info">
                <AlertTitle>Instructions</AlertTitle>
                <ol style={{ margin: 0, paddingLeft: 20 }}>
                  <li>Enter a descriptive name for your template</li>
                  <li>Upload a template file (image or PDF)</li>
                  <li>The system will analyze and extract validation rules</li>
                  <li>Use this template to validate documents</li>
                </ol>
              </Alert>
            )}
          </Grid>
        </Grid>
      </Paper>

      {/* Templates List */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Existing Templates
        </Typography>
        
        {loading && (
          <Box sx={{ width: '100%' }}>
            <LinearProgress />
            <Typography variant="body2" sx={{ mt: 2, textAlign: 'center' }}>
              Loading templates...
            </Typography>
          </Box>
        )}
        
        {!loading && templates.length === 0 && (
          <Alert severity="info">
            No templates uploaded yet. Upload your first template above.
          </Alert>
        )}
        
        {!loading && templates.length > 0 && (
          <List>
            {templates.map((template, index) => (
              <React.Fragment key={template.id}>
                {index > 0 && <Divider />}
                <ListItem
                  secondaryAction={
                    <Box>
                      <IconButton
                        edge="end"
                        aria-label="view"
                        onClick={() => handleViewDetails(template)}
                      >
                        <ViewIcon />
                      </IconButton>
                    </Box>
                  }
                >
                  <ListItemIcon>
                    <FileIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body1">{template.name}</Typography>
                        <Chip
                          label={`ID: ${template.id.slice(0, 8)}...`}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Uploaded: {formatDate(template.created_at)}
                        </Typography>
                        {template.rules && (
                          <Box sx={{ mt: 0.5 }}>
                            <Chip
                              label={`${Object.keys(template.rules).length} rule categories`}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          </Box>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        )}
      </Paper>

      {/* Template Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={handleCloseDetails}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Template Details: {selectedTemplate?.name}
        </DialogTitle>
        <DialogContent>
          {selectedTemplate && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Template ID
              </Typography>
              <Typography variant="body2" sx={{ mb: 2, fontFamily: 'monospace' }}>
                {selectedTemplate.id}
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom>
                Created
              </Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                {formatDate(selectedTemplate.created_at)}
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom>
                Validation Rules
              </Typography>
              <Paper variant="outlined" sx={{ p: 2, backgroundColor: 'grey.50' }}>
                <pre style={{ margin: 0, overflow: 'auto' }}>
                  {JSON.stringify(selectedTemplate.rules, null, 2)}
                </pre>
              </Paper>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDetails}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Notifications */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
      
      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
      >
        <Alert onClose={() => setSuccess(null)} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default TemplateUpload;
