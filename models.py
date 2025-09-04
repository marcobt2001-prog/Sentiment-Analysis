"""
Enterprise-grade Pydantic data models for consulting visual generation tool.

This module defines the core data structures using Pydantic v2 for robust validation,
serialization, and comprehensive audit trail capabilities required for production
consulting environments.
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Enumeration of validation statuses for data records."""
    PENDING = "pending"
    VALIDATED = "validated" 
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class SentimentType(str, Enum):
    """Enumeration of sentiment classifications."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ChartType(str, Enum):
    """Enumeration of supported chart types."""
    EMERGENT_THEMES = "emergent_themes"
    GROUP_SENTIMENT = "group_sentiment"
    RISK_MATRIX = "risk_matrix"
    CULTURE_ANALYSIS = "culture_analysis"
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class VerificationError(Exception):
    """Custom exception for verification process errors."""
    pass


class BaseDataRecord(BaseModel):
    """
    Base data record with comprehensive audit trail capabilities.
    
    Serves as the foundation for all data records in the system, providing
    essential tracking, validation, and traceability features required for
    enterprise-grade consulting applications.
    
    Attributes:
        record_id (UUID): Unique identifier for the record
        timestamp (datetime): Creation timestamp
        source_file (Optional[str]): Originating file path or identifier
        confidence_score (float): Data quality confidence score (0.0-1.0)
        validation_status (ValidationStatus): Current validation state
        metadata (Dict[str, Any]): Additional metadata for extensibility
        created_by (Optional[str]): User or system that created the record
        last_modified (datetime): Last modification timestamp
    """
    
    record_id: UUID = Field(default_factory=uuid4, description="Unique record identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    source_file: Optional[str] = Field(None, description="Source file path")
    confidence_score: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0,
        description="Data quality confidence score"
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.PENDING,
        description="Current validation status"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_by: Optional[str] = Field(None, description="Record creator")
    last_modified: datetime = Field(default_factory=datetime.now, description="Last modification")
    
    model_config = {
        "json_encoders": {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            Path: str
        },
        "validate_assignment": True,
        "arbitrary_types_allowed": True
    }
    
    @field_validator('source_file')
    @classmethod
    def validate_source_file(cls, v: Optional[str]) -> Optional[str]:
        """Validate source file path format."""
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("Source file must be a non-empty string")
            # Convert to Path to validate format
            try:
                Path(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid file path format: {e}")
        return v
    
    @model_validator(mode='after')
    def validate_record_integrity(self) -> 'BaseDataRecord':
        """Perform cross-field validation and integrity checks."""
        # Ensure last_modified is not before timestamp
        if self.last_modified < self.timestamp:
            self.last_modified = self.timestamp
        
        # Log record creation for audit trail
        logger.debug(f"Created BaseDataRecord {self.record_id} with confidence {self.confidence_score}")
        
        return self
    
    def to_audit_dict(self) -> Dict[str, Any]:
        """
        Generate comprehensive audit dictionary for compliance tracking.
        
        Returns:
            Dict[str, Any]: Complete audit information including all fields
        """
        return {
            "record_id": str(self.record_id),
            "timestamp": self.timestamp.isoformat(),
            "source_file": self.source_file,
            "confidence_score": self.confidence_score,
            "validation_status": self.validation_status.value,
            "created_by": self.created_by,
            "last_modified": self.last_modified.isoformat(),
            "metadata": self.metadata
        }
    
    def update_validation_status(self, status: ValidationStatus, updated_by: Optional[str] = None) -> None:
        """
        Update validation status with audit trail.
        
        Args:
            status (ValidationStatus): New validation status
            updated_by (Optional[str]): User performing the update
        """
        old_status = self.validation_status
        self.validation_status = status
        self.last_modified = datetime.now()
        
        if updated_by:
            self.metadata["last_updated_by"] = updated_by
        
        logger.info(f"Record {self.record_id} status changed from {old_status} to {status}")


class InterviewRecord(BaseDataRecord):
    """
    Specialized data record for interview content with comprehensive analysis.
    
    Extends BaseDataRecord to include interview-specific fields for content analysis,
    speaker identification, sentiment tracking, and thematic categorization essential
    for consulting insights and reporting.
    
    Attributes:
        paragraph_id (str): Unique paragraph identifier within the interview
        speaker_role (str): Role/position of the speaker
        department (Optional[str]): Speaker's department or division
        group (Optional[str]): Organizational group or team
        theme (str): Primary thematic category
        sentiment (SentimentType): Sentiment classification
        sentiment_score (float): Numerical sentiment confidence (-1.0 to 1.0)
        text (str): The actual interview content
        question_category (Optional[str]): Category of the question that prompted response
        keywords (List[str]): Extracted keywords for analysis
        word_count (int): Cached word count for performance
        contains_pii (bool): Flag indicating potential PII content
    """
    
    paragraph_id: str = Field(..., min_length=1, description="Paragraph identifier")
    speaker_role: str = Field(..., min_length=1, description="Speaker's role or position")
    department: Optional[str] = Field(None, description="Speaker's department")
    group: Optional[str] = Field(None, description="Organizational group")
    theme: str = Field(default="General", min_length=1, description="Primary thematic category")
    sentiment: SentimentType = Field(default=SentimentType.NEUTRAL, description="Sentiment classification")
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Numerical sentiment score"
    )
    text: str = Field(..., min_length=1, description="Interview content text")
    question_category: Optional[str] = Field(None, description="Question category")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    word_count: int = Field(default=0, ge=0, description="Text word count")
    contains_pii: bool = Field(default=False, description="PII content flag")
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v: str) -> str:
        """Validate and clean interview text content."""
        if not v.strip():
            raise ValueError("Interview text cannot be empty or only whitespace")
        
        # Basic PII detection patterns (simplified for example)
        pii_patterns = ['ssn', 'social security', 'phone:', 'email:', '@']
        text_lower = v.lower()
        contains_potential_pii = any(pattern in text_lower for pattern in pii_patterns)
        
        if contains_potential_pii:
            logger.warning("Potential PII detected in interview text")
        
        return v.strip()
    
    @field_validator('keywords')
    @classmethod 
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Validate and normalize keywords list."""
        if v is None:
            return []
        
        # Remove duplicates and empty strings, normalize case
        normalized = []
        seen = set()
        for keyword in v:
            if isinstance(keyword, str) and keyword.strip():
                clean_keyword = keyword.strip().lower()
                if clean_keyword not in seen:
                    normalized.append(clean_keyword)
                    seen.add(clean_keyword)
        
        return normalized
    
    @model_validator(mode='after')
    def calculate_derived_fields(self) -> 'InterviewRecord':
        """Calculate derived fields and perform additional validation."""
        # Update word count
        self.word_count = len(self.text.split())
        
        # Check for PII indicators
        pii_indicators = ['ssn', 'social security', 'phone', 'email', '@', 'address']
        text_lower = self.text.lower()
        self.contains_pii = any(indicator in text_lower for indicator in pii_indicators)
        
        # Validate sentiment score alignment
        if self.sentiment == SentimentType.POSITIVE and self.sentiment_score < 0:
            logger.warning(f"Sentiment mismatch in record {self.record_id}: positive sentiment with negative score")
        elif self.sentiment == SentimentType.NEGATIVE and self.sentiment_score > 0:
            logger.warning(f"Sentiment mismatch in record {self.record_id}: negative sentiment with positive score")
        
        # Call parent validator
        super().validate_record_integrity()
        
        return self
    
    def extract_themes(self) -> List[str]:
        """
        Extract potential themes from the text content.
        
        Returns:
            List[str]: List of identified themes
        """
        # Simplified theme extraction (in production, use NLP libraries)
        theme_keywords = {
            'digital_transformation': ['digital', 'technology', 'automation', 'ai', 'innovation'],
            'culture': ['culture', 'values', 'behavior', 'attitude', 'environment'],
            'leadership': ['leadership', 'management', 'direction', 'vision', 'strategy'],
            'change_management': ['change', 'transition', 'transformation', 'adaptation'],
            'customer_focus': ['customer', 'client', 'service', 'satisfaction', 'experience']
        }
        
        text_lower = self.text.lower()
        identified_themes = []
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                identified_themes.append(theme)
        
        return identified_themes
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis summary for reporting.
        
        Returns:
            Dict[str, Any]: Analysis summary with metrics and insights
        """
        return {
            "record_id": str(self.record_id),
            "speaker_info": {
                "role": self.speaker_role,
                "department": self.department,
                "group": self.group
            },
            "content_analysis": {
                "theme": self.theme,
                "sentiment": self.sentiment.value,
                "sentiment_score": self.sentiment_score,
                "word_count": self.word_count,
                "keywords": self.keywords,
                "identified_themes": self.extract_themes()
            },
            "data_quality": {
                "confidence_score": self.confidence_score,
                "validation_status": self.validation_status.value,
                "contains_pii": self.contains_pii
            },
            "metadata": {
                "question_category": self.question_category,
                "source_file": self.source_file,
                "timestamp": self.timestamp.isoformat()
            }
        }


class ChartData(BaseModel):
    """
    Comprehensive data structure for chart generation and validation.
    
    Contains all necessary data and metadata for generating professional
    consulting visualizations with full traceability and audit capabilities.
    
    Attributes:
        values (List[float]): Numerical data values for visualization
        labels (List[str]): Corresponding labels for data points
        source_ids (List[str]): Source record identifiers for traceability
        confidence_scores (List[float]): Per-data-point confidence scores
        chart_type (ChartType): Type of chart to generate
        metadata (Dict[str, Any]): Additional chart metadata
        chart_id (UUID): Unique chart identifier
        title (str): Chart title
        subtitle (Optional[str]): Optional chart subtitle
        x_axis_label (str): X-axis label
        y_axis_label (str): Y-axis label
        created_timestamp (datetime): Chart data creation time
        data_source_summary (Dict[str, Any]): Summary of data sources
    """
    
    values: List[float] = Field(..., min_length=1, description="Chart data values")
    labels: List[str] = Field(..., min_length=1, description="Data point labels")
    source_ids: List[str] = Field(..., min_length=1, description="Source record IDs")
    confidence_scores: List[float] = Field(..., min_length=1, description="Confidence scores")
    chart_type: ChartType = Field(..., description="Chart type specification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    chart_id: UUID = Field(default_factory=uuid4, description="Unique chart identifier")
    title: str = Field(..., min_length=1, description="Chart title")
    subtitle: Optional[str] = Field(None, description="Chart subtitle")
    x_axis_label: str = Field(default="Category", description="X-axis label")
    y_axis_label: str = Field(default="Value", description="Y-axis label")
    created_timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    data_source_summary: Dict[str, Any] = Field(default_factory=dict, description="Data source summary")
    
    @model_validator(mode='after')
    def validate_data_consistency(self) -> 'ChartData':
        """Validate consistency across all data arrays."""
        data_length = len(self.values)
        
        # Check all arrays have same length
        if not all(len(arr) == data_length for arr in [self.labels, self.source_ids, self.confidence_scores]):
            raise ValueError("All data arrays (values, labels, source_ids, confidence_scores) must have the same length")
        
        # Validate confidence scores
        for i, score in enumerate(self.confidence_scores):
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Confidence score at index {i} must be between 0.0 and 1.0, got {score}")
        
        # Generate data source summary
        self._generate_source_summary()
        
        logger.info(f"ChartData validated: {data_length} data points for {self.chart_type.value} chart")
        
        return self
    
    def _generate_source_summary(self) -> None:
        """Generate summary of data sources for audit trail."""
        unique_sources = set(self.source_ids)
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
        min_confidence = min(self.confidence_scores)
        max_confidence = max(self.confidence_scores)
        
        self.data_source_summary = {
            "total_data_points": len(self.values),
            "unique_source_count": len(unique_sources),
            "confidence_statistics": {
                "average": round(avg_confidence, 3),
                "minimum": round(min_confidence, 3),
                "maximum": round(max_confidence, 3)
            },
            "chart_type": self.chart_type.value,
            "generated_at": self.created_timestamp.isoformat()
        }
    
    def get_low_confidence_points(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Identify data points with confidence below threshold.
        
        Args:
            threshold (float): Confidence threshold for flagging
            
        Returns:
            List[Dict[str, Any]]: Low confidence data points with details
        """
        low_confidence = []
        for i, score in enumerate(self.confidence_scores):
            if score < threshold:
                low_confidence.append({
                    "index": i,
                    "label": self.labels[i],
                    "value": self.values[i],
                    "confidence": score,
                    "source_id": self.source_ids[i]
                })
        
        return low_confidence
    
    def validate_for_chart_type(self) -> bool:
        """
        Validate data compatibility with specified chart type.
        
        Returns:
            bool: True if data is compatible with chart type
            
        Raises:
            ValidationError: If data is incompatible with chart type
        """
        try:
            if self.chart_type == ChartType.PIE:
                if any(v < 0 for v in self.values):
                    raise ValidationError("Pie charts cannot contain negative values")
                
            elif self.chart_type == ChartType.RISK_MATRIX:
                if len(self.values) < 4:
                    raise ValidationError("Risk matrix requires at least 4 data points")
                    
            elif self.chart_type in [ChartType.EMERGENT_THEMES, ChartType.CULTURE_ANALYSIS]:
                if len(set(self.labels)) != len(self.labels):
                    raise ValidationError(f"{self.chart_type.value} requires unique labels")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Chart type validation failed: {e}")


class VerificationReport(BaseModel):
    """
    Comprehensive verification report for chart validation and audit trail.
    
    Documents the complete verification process including automated checks,
    validation results, identified issues, and recommendations for human review.
    
    Attributes:
        chart_id (UUID): Associated chart identifier
        validation_status (ValidationStatus): Overall validation result
        issues_found (List[str]): Detailed list of identified issues
        human_review_required (bool): Flag indicating need for human review
        timestamp (datetime): Verification timestamp
        verification_score (float): Overall verification confidence (0.0-1.0)
        automated_checks_passed (int): Number of automated checks that passed
        total_automated_checks (int): Total number of automated checks
        reviewer_notes (Optional[str]): Optional reviewer comments
        recommended_actions (List[str]): Recommended corrective actions
        compliance_flags (Dict[str, bool]): Compliance requirement flags
    """
    
    chart_id: UUID = Field(..., description="Associated chart ID")
    validation_status: ValidationStatus = Field(..., description="Validation result")
    issues_found: List[str] = Field(default_factory=list, description="Identified issues")
    human_review_required: bool = Field(default=False, description="Human review flag")
    timestamp: datetime = Field(default_factory=datetime.now, description="Verification timestamp")
    verification_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Verification confidence score"
    )
    automated_checks_passed: int = Field(default=0, ge=0, description="Passed automated checks")
    total_automated_checks: int = Field(default=0, ge=0, description="Total automated checks")
    reviewer_notes: Optional[str] = Field(None, description="Reviewer comments")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    compliance_flags: Dict[str, bool] = Field(default_factory=dict, description="Compliance flags")
    
    @field_validator('issues_found')
    @classmethod
    def validate_issues(cls, v: List[str]) -> List[str]:
        """Validate and clean issues list."""
        return [issue.strip() for issue in v if issue and issue.strip()]
    
    @model_validator(mode='after')
    def calculate_verification_metrics(self) -> 'VerificationReport':
        """Calculate verification metrics and determine review requirements."""
        # Calculate verification score based on checks passed
        if self.total_automated_checks > 0:
            self.verification_score = self.automated_checks_passed / self.total_automated_checks
        
        # Determine if human review is required
        if self.verification_score < 0.8 or len(self.issues_found) > 3:
            self.human_review_required = True
            
        # Set validation status based on results
        if self.verification_score >= 0.95 and not self.issues_found:
            self.validation_status = ValidationStatus.APPROVED
        elif self.verification_score >= 0.8 and len(self.issues_found) <= 2:
            self.validation_status = ValidationStatus.VALIDATED
        elif self.human_review_required:
            self.validation_status = ValidationStatus.REQUIRES_REVIEW
        else:
            self.validation_status = ValidationStatus.FAILED
        
        logger.info(f"Verification complete for chart {self.chart_id}: "
                   f"{self.validation_status.value} (score: {self.verification_score:.2f})")
        
        return self
    
    def add_issue(self, issue: str, severity: str = "medium") -> None:
        """
        Add a new issue to the verification report.
        
        Args:
            issue (str): Description of the issue
            severity (str): Severity level (low, medium, high, critical)
        """
        if issue.strip():
            formatted_issue = f"[{severity.upper()}] {issue.strip()}"
            self.issues_found.append(formatted_issue)
            
            # Update human review requirement for high/critical issues
            if severity.lower() in ['high', 'critical']:
                self.human_review_required = True
    
    def add_compliance_flag(self, requirement: str, passed: bool) -> None:
        """
        Add a compliance requirement result.
        
        Args:
            requirement (str): Compliance requirement name
            passed (bool): Whether the requirement was met
        """
        self.compliance_flags[requirement] = passed
        if not passed:
            self.add_issue(f"Compliance requirement not met: {requirement}", "high")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate executive summary of verification results.
        
        Returns:
            Dict[str, Any]: Comprehensive verification summary
        """
        return {
            "chart_id": str(self.chart_id),
            "verification_summary": {
                "status": self.validation_status.value,
                "score": self.verification_score,
                "human_review_required": self.human_review_required,
                "timestamp": self.timestamp.isoformat()
            },
            "quality_metrics": {
                "automated_checks": f"{self.automated_checks_passed}/{self.total_automated_checks}",
                "issues_count": len(self.issues_found),
                "compliance_score": sum(self.compliance_flags.values()) / max(len(self.compliance_flags), 1)
            },
            "issues": self.issues_found,
            "recommended_actions": self.recommended_actions,
            "compliance_status": self.compliance_flags,
            "reviewer_notes": self.reviewer_notes
        }


class FeedbackData(BaseModel):
    """
    Comprehensive feedback data structure for continuous improvement.
    
    Captures user feedback, corrections, and quality assessments to improve
    the visualization generation system through machine learning and process
    refinement.
    
    Attributes:
        chart_id (UUID): Associated chart identifier
        rating (int): User rating (1-5 scale)
        comments (Optional[str]): Detailed user comments
        corrections (Dict[str, Any]): Specific corrections or suggestions
        consultant_id (str): Identifier of the providing consultant
        timestamp (datetime): Feedback submission timestamp
        feedback_category (str): Category of feedback (quality, accuracy, design, etc.)
        actionable (bool): Whether feedback contains actionable items
        priority (str): Priority level for addressing feedback
        resolution_status (str): Current resolution status
    """
    
    chart_id: UUID = Field(..., description="Associated chart ID")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    comments: Optional[str] = Field(None, description="User comments")
    corrections: Dict[str, Any] = Field(default_factory=dict, description="Specific corrections")
    consultant_id: str = Field(..., min_length=1, description="Consultant identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")
    feedback_category: str = Field(default="general", description="Feedback category")
    actionable: bool = Field(default=False, description="Contains actionable items")
    priority: str = Field(default="medium", description="Priority level")
    resolution_status: str = Field(default="open", description="Resolution status")
    
    @field_validator('comments')
    @classmethod
    def validate_comments(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean user comments."""
        if v:
            cleaned = v.strip()
            if len(cleaned) > 5000:
                raise ValueError("Comments cannot exceed 5000 characters")
            return cleaned
        return v
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate priority level."""
        valid_priorities = {'low', 'medium', 'high', 'critical'}
        if v.lower() not in valid_priorities:
            raise ValueError(f"Priority must be one of {valid_priorities}")
        return v.lower()
    
    @model_validator(mode='after')
    def determine_actionability(self) -> 'FeedbackData':
        """Determine if feedback contains actionable items."""
        actionable_indicators = [
            'incorrect', 'wrong', 'should be', 'missing', 'add', 'remove',
            'change', 'fix', 'update', 'improve', 'suggestion'
        ]
        
        text_to_check = []
        if self.comments:
            text_to_check.append(self.comments.lower())
        
        # Check corrections dictionary
        for key, value in self.corrections.items():
            if isinstance(value, str):
                text_to_check.append(value.lower())
        
        combined_text = ' '.join(text_to_check)
        self.actionable = any(indicator in combined_text for indicator in actionable_indicators)
        
        # Set priority based on rating and actionability
        if self.rating <= 2 and self.actionable:
            self.priority = "high"
        elif self.rating <= 3 and self.actionable:
            self.priority = "medium"
        
        return self
    
    def extract_action_items(self) -> List[str]:
        """
        Extract specific action items from feedback.
        
        Returns:
            List[str]: List of actionable improvements
        """
        action_items = []
        
        # Extract from corrections
        for key, value in self.corrections.items():
            if isinstance(value, str) and value.strip():
                action_items.append(f"{key}: {value.strip()}")
        
        # Simple extraction from comments (in production, use NLP)
        if self.comments:
            sentences = self.comments.split('.')
            action_keywords = ['should', 'need to', 'must', 'please', 'suggest', 'recommend']
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in action_keywords):
                    action_items.append(sentence.strip())
        
        return action_items
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Generate structured feedback summary.
        
        Returns:
            Dict[str, Any]: Comprehensive feedback summary
        """
        return {
            "chart_id": str(self.chart_id),
            "consultant_id": self.consultant_id,
            "assessment": {
                "rating": self.rating,
                "category": self.feedback_category,
                "timestamp": self.timestamp.isoformat()
            },
            "content": {
                "comments": self.comments,
                "corrections_count": len(self.corrections),
                "actionable": self.actionable,
                "action_items": self.extract_action_items()
            },
            "management": {
                "priority": self.priority,
                "resolution_status": self.resolution_status
            }
        }