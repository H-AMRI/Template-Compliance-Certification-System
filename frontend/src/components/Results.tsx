import React from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Alert,
  AlertTitle,
  Card,
  CardContent,
  LinearProgress,
  Divider,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  Assessment as AssessmentIcon,
  Build as FixIcon,
  Visibility as VisualIcon,
  GridOn as LayoutIcon,
  TextFields as TextIcon,
  Architecture as StructureIcon,
} from '@mui/icons-material';

interface Correction {
  type: string;
  issue: string;
  score?: number;
  details?: any;
  missing_fields?: string[];
  missing_blocks?: string[];
}

interface ResultsProps {
  is_compliant: boolean;
  corrections: Correction[];
  confidence_score?: number;
}

const Results: React.FC<ResultsProps> = ({ is_compliant, corrections, confidence_score }) => {
  const getIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'visual':
        return <VisualIcon />;
      case 'layout':
        return <LayoutIcon />;
      case 'text':
        return <TextIcon />;
      case 'structural':
        return <StructureIcon />;
      default:
        return <AssessmentIcon />;
    }
  };

  const getSeverityColor = (score?: number): 'error' | 'warning' | 'success' => {
    if (!score) return 'error';
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (score: number): string => {
    if (score >= 0.9) return 'Very High';
    if (score >= 0.8) return 'High';
    if (score >= 0.7) return 'Moderate';
    if (score >= 0.6) return 'Low';
    return 'Very Low';
  };

  const getConfidenceColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Box>
      {/* Main Status Card */}
      <Card sx={{ mb: 3, border: 2, borderColor: is_compliant ? 'success.main' : 'error.main' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              {is_compliant ? (
                <CheckIcon sx={{ fontSize: 48, color: 'success.main', mr: 2 }} />
              ) : (
                <CancelIcon sx={{ fontSize: 48, color: 'error.main', mr: 2 }} />
              )}
              <Box>
                <Typography variant="h5" component="div">
                  Document is {is_compliant ? 'Compliant' : 'Non-Compliant'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {is_compliant
                    ? 'All validation checks passed successfully'
                    : `${corrections.length} issue${corrections.length !== 1 ? 's' : ''} found`}
                </Typography>
              </Box>
            </Box>
            
            {confidence_score !== undefined && (
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="primary">
                  {(confidence_score * 100).toFixed(1)}%
                </Typography>
                <Chip
                  label={getConfidenceLabel(confidence_score)}
                  color={getConfidenceColor(confidence_score)}
                  size="small"
                />
              </Box>
            )}
          </Box>

          {confidence_score !== undefined && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary">
                Confidence Score
              </Typography>
              <LinearProgress
                variant="determinate"
                value={confidence_score * 100}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: 'grey.200',
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 4,
                    backgroundColor: getConfidenceColor(confidence_score) + '.main',
                  },
                }}
              />
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Success Message */}
      {is_compliant && corrections.length === 0 && (
        <Alert severity="success" sx={{ mb: 3 }}>
          <AlertTitle>Perfect Match!</AlertTitle>
          Your document fully complies with the template requirements. No corrections needed.
        </Alert>
      )}

      {/* Corrections List */}
      {corrections.length > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <FixIcon sx={{ mr: 1 }} />
            Required Corrections
          </Typography>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            The following issues need to be addressed for full compliance:
          </Typography>

          <List>
            {corrections.map((correction, index) => (
              <React.Fragment key={index}>
                {index > 0 && <Divider />}
                <ListItem sx={{ py: 2 }}>
                  <ListItemIcon>
                    {getIcon(correction.type)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle1">
                          {correction.type.charAt(0).toUpperCase() + correction.type.slice(1)} Issue
                        </Typography>
                        {correction.score !== undefined && (
                          <Chip
                            label={`Score: ${(correction.score * 100).toFixed(0)}%`}
                            size="small"
                            color={getSeverityColor(correction.score)}
                            variant="outlined"
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2" component="div">
                          {correction.issue}
                        </Typography>
                        
                        {/* Additional details based on correction type */}
                        {correction.missing_fields && correction.missing_fields.length > 0 && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                              Missing fields:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                              {correction.missing_fields.map((field, i) => (
                                <Chip
                                  key={i}
                                  label={field}
                                  size="small"
                                  variant="outlined"
                                  color="error"
                                />
                              ))}
                            </Box>
                          </Box>
                        )}
                        
                        {correction.missing_blocks && correction.missing_blocks.length > 0 && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                              Missing layout blocks:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                              {correction.missing_blocks.map((block, i) => (
                                <Chip
                                  key={i}
                                  label={block}
                                  size="small"
                                  variant="outlined"
                                  color="warning"
                                />
                              ))}
                            </Box>
                          </Box>
                        )}
                        
                        {correction.details && (
                          <Accordion sx={{ mt: 1 }}>
                            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography variant="caption">View Details</Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              <pre style={{ margin: 0, fontSize: '0.75rem', overflow: 'auto' }}>
                                {JSON.stringify(correction.details, null, 2)}
                              </pre>
                            </AccordionDetails>
                          </Accordion>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
              </React.Fragment>
            ))}
          </List>

          {/* Summary Statistics */}
          <Box sx={{ mt: 3, p: 2, backgroundColor: 'grey.50', borderRadius: 1 }}>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="primary">
                    {corrections.length}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Total Issues
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="error">
                    {corrections.filter(c => c.score && c.score < 0.6).length}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Critical
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="warning.main">
                    {corrections.filter(c => c.score && c.score >= 0.6 && c.score < 0.8).length}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Warnings
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="success.main">
                    {corrections.filter(c => c.score && c.score >= 0.8).length}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Minor
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Box>
        </Paper>
      )}

      {/* Recommendations */}
      {!is_compliant && (
        <Alert severity="info" sx={{ mt: 3 }}>
          <AlertTitle>How to Fix</AlertTitle>
          <ol style={{ margin: 0, paddingLeft: 20 }}>
            <li>Review each correction item listed above</li>
            <li>Update your document to address the identified issues</li>
            <li>Ensure all required fields and sections are present</li>
            <li>Re-upload and validate the corrected document</li>
          </ol>
        </Alert>
      )}
    </Box>
  );
};

export default Results;
