"""
Core data models for the consulting visual generation tool.

This module defines the primary data structures used throughout the application
for managing interview records, audit trails, and visualization configurations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataRecord:
    """
    Base data record with comprehensive audit trail capabilities.
    
    This class serves as the foundation for all data records in the system,
    providing essential tracking and traceability features required for
    enterprise-grade applications.
    
    Attributes:
        record_id (str): Unique identifier for the record, auto-generated
        timestamp (datetime): Creation timestamp, defaults to current time
        confidence_score (float): Data quality confidence score (0.0-1.0)
        source_file (Optional[str]): Originating file path or identifier
        metadata (Dict[str, Any]): Additional metadata for extensibility
    
    Raises:
        ValueError: If confidence_score is not between 0.0 and 1.0
    """
    record_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = field(default=1.0)
    source_file: Optional[str] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate data integrity after initialization."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")
        
        logger.debug(f"Created DataRecord {self.record_id} with confidence {self.confidence_score}")
    
    def to_audit_dict(self) -> Dict[str, Any]:
        """
        Generate audit trail dictionary for logging and compliance.
        
        Returns:
            Dict[str, Any]: Comprehensive audit information
        """
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "source_file": self.source_file,
            "metadata": self.metadata
        }


@dataclass(frozen=True)
class InterviewRecord(DataRecord):
    """
    Specialized data record for interview content with thematic analysis.
    
    Extends DataRecord to include interview-specific fields for content analysis,
    speaker identification, and sentiment tracking essential for consulting insights.
    
    Attributes:
        paragraph_id (str): Unique paragraph identifier within the interview
        speaker_role (str): Role/position of the speaker (e.g., "CEO", "Manager")
        theme (str): Primary thematic category for the content
        sentiment (str): Sentiment classification ("positive", "negative", "neutral")
        text (str): The actual interview content text
        keywords (List[str]): Extracted keywords for search and analysis
        weight (float): Relative importance weight for visualization
    
    Raises:
        ValueError: If sentiment is not one of the allowed values
        ValueError: If text is empty
        ValueError: If weight is negative
    """
    paragraph_id: str = field(default="")
    speaker_role: str = field(default="Unknown")
    theme: str = field(default="General")
    sentiment: str = field(default="neutral")
    text: str = field(default="")
    keywords: List[str] = field(default_factory=list)
    weight: float = field(default=1.0)
    
    VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
    
    def __post_init__(self) -> None:
        """Enhanced validation for interview-specific fields."""
        super().__post_init__()
        
        if self.sentiment not in self.VALID_SENTIMENTS:
            raise ValueError(f"Sentiment must be one of {self.VALID_SENTIMENTS}, got '{self.sentiment}'")
        
        if not self.text.strip():
            raise ValueError("Interview text cannot be empty")
        
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        
        logger.debug(f"Created InterviewRecord {self.record_id} for {self.speaker_role} with theme '{self.theme}'")
    
    def get_word_count(self) -> int:
        """Get word count for the interview text."""
        return len(self.text.split())
    
    def has_keyword(self, keyword: str) -> bool:
        """Check if the record contains a specific keyword (case-insensitive)."""
        keyword_lower = keyword.lower()
        return (keyword_lower in [k.lower() for k in self.keywords] or 
                keyword_lower in self.text.lower())


@dataclass
class VisualizationConfig:
    """
    Configuration specifications for chart generation and styling.
    
    Provides comprehensive configuration management for visualization parameters,
    ensuring consistent styling and professional presentation standards across
    all generated charts and reports.
    
    Attributes:
        chart_type (str): Type of chart to generate
        title (str): Chart title
        width (int): Chart width in pixels
        height (int): Chart height in pixels
        dpi (int): Resolution in dots per inch
        color_scheme (str): Color scheme identifier
        style_template (str): Style template name
        export_formats (List[str]): List of export format extensions
        font_family (str): Primary font family for text
        font_sizes (Dict[str, int]): Font sizes for different elements
        margins (Dict[str, float]): Chart margins configuration
        interactive (bool): Enable interactive features
    
    Raises:
        ValueError: If dimensions are non-positive
        ValueError: If DPI is below minimum professional standard
        ValueError: If unsupported export format is specified
    """
    chart_type: str = "bar"
    title: str = "Untitled Chart"
    width: int = 1200
    height: int = 800
    dpi: int = 300
    color_scheme: str = "professional"
    style_template: str = "consulting"
    export_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    font_family: str = "Arial"
    font_sizes: Dict[str, int] = field(default_factory=lambda: {
        "title": 16,
        "axis_label": 12,
        "axis_tick": 10,
        "legend": 11,
        "annotation": 9
    })
    margins: Dict[str, float] = field(default_factory=lambda: {
        "top": 0.15,
        "bottom": 0.15,
        "left": 0.15,
        "right": 0.15
    })
    interactive: bool = False
    
    SUPPORTED_CHART_TYPES = {
        "bar", "line", "scatter", "pie", "heatmap", "box", "violin", 
        "histogram", "area", "treemap", "sankey", "waterfall"
    }
    SUPPORTED_EXPORT_FORMATS = {"png", "pdf", "svg", "jpg", "eps"}
    MIN_DPI = 150
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Dimensions must be positive, got width={self.width}, height={self.height}")
        
        if self.dpi < self.MIN_DPI:
            raise ValueError(f"DPI must be at least {self.MIN_DPI} for professional quality, got {self.dpi}")
        
        if self.chart_type not in self.SUPPORTED_CHART_TYPES:
            raise ValueError(f"Unsupported chart type '{self.chart_type}'. "
                           f"Supported types: {sorted(self.SUPPORTED_CHART_TYPES)}")
        
        unsupported_formats = set(self.export_formats) - self.SUPPORTED_EXPORT_FORMATS
        if unsupported_formats:
            raise ValueError(f"Unsupported export formats: {unsupported_formats}. "
                           f"Supported formats: {sorted(self.SUPPORTED_EXPORT_FORMATS)}")
        
        logger.debug(f"Created VisualizationConfig for {self.chart_type} chart: {self.width}x{self.height}@{self.dpi}DPI")
    
    def get_figure_size(self) -> tuple[float, float]:
        """Calculate matplotlib figure size in inches."""
        return (self.width / self.dpi, self.height / self.dpi)
    
    def is_high_resolution(self) -> bool:
        """Check if configuration meets high-resolution standards."""
        return self.dpi >= 300 and self.width >= 1200 and self.height >= 800
    
    def to_matplotlib_params(self) -> Dict[str, Any]:
        """Convert to matplotlib-compatible parameter dictionary."""
        return {
            "figsize": self.get_figure_size(),
            "dpi": self.dpi,
            "facecolor": "white",
            "edgecolor": "none"
        }
    
    def validate_for_export(self, format_type: str) -> bool:
        """
        Validate configuration for specific export format.
        
        Args:
            format_type (str): Export format to validate against
            
        Returns:
            bool: True if configuration is suitable for the format
        """
        if format_type not in self.SUPPORTED_EXPORT_FORMATS:
            return False
        
        if format_type == "pdf" and not self.is_high_resolution():
            logger.warning("PDF export recommended with high resolution settings")
        
        return True