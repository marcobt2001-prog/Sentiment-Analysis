"""
Enterprise system configuration for consulting visual generation tool.

This module provides comprehensive configuration management including visualization
settings, professional styling, confidence thresholds, chart specifications, and
enterprise-grade logging configuration.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .models import ChartType, ValidationStatus


class LogLevel(str, Enum):
    """Enumeration of logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OutputFormat(str, Enum):
    """Enumeration of supported output formats."""
    PNG = "png"
    PDF = "pdf"
    SVG = "svg"
    JPEG = "jpeg"
    EPS = "eps"


@dataclass
class VisualizationSettings:
    """
    Professional visualization configuration settings.
    
    Attributes:
        dpi (int): Dots per inch for high-quality output
        chart_style (str): Professional chart styling template
        output_format (OutputFormat): Primary output format
        width (int): Default chart width in pixels
        height (int): Default chart height in pixels
        font_family (str): Primary font family
        font_size_base (int): Base font size in points
        line_width (float): Default line width
        figure_padding (float): Padding around figure elements
    """
    dpi: int = 300
    chart_style: str = "enterprise"
    output_format: OutputFormat = OutputFormat.PNG
    width: int = 1200
    height: int = 800
    font_family: str = "Arial"
    font_size_base: int = 12
    line_width: float = 2.0
    figure_padding: float = 0.15


@dataclass
class ProfessionalColorPalette:
    """
    Professional color palette for consulting presentations.
    
    Provides enterprise-standard colors that meet accessibility requirements
    and maintain professional appearance across all visualization types.
    
    Attributes:
        primary_blue (str): Primary corporate blue
        success_green (str): Success/positive indicator color
        warning_orange (str): Warning/attention color
        danger_red (str): Error/negative indicator color
        neutral_gray (str): Neutral/background color
        text_dark (str): Primary text color
        text_light (str): Secondary text color
        background_white (str): Clean background color
        accent_colors (List[str]): Additional accent colors
        gradient_colors (Dict[str, List[str]]): Gradient definitions
    """
    primary_blue: str = "#1f4788"
    success_green: str = "#70ad47"
    warning_orange: str = "#ffc000"
    danger_red: str = "#e74c3c"
    neutral_gray: str = "#95a5a6"
    text_dark: str = "#2c3e50"
    text_light: str = "#7f8c8d"
    background_white: str = "#ffffff"
    accent_colors: List[str] = field(default_factory=lambda: [
        "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#f39c12", "#34495e"
    ])
    gradient_colors: Dict[str, List[str]] = field(default_factory=lambda: {
        "blue_gradient": ["#ebf3fd", "#d6e7fa", "#3498db", "#1f4788"],
        "green_gradient": ["#eaf4e7", "#d5e9d0", "#70ad47", "#5a8f38"],
        "risk_gradient": ["#fff3cd", "#ffc000", "#e74c3c", "#c0392b"]
    })
    
    def get_color_by_sentiment(self, sentiment: str) -> str:
        """
        Get appropriate color based on sentiment.
        
        Args:
            sentiment (str): Sentiment type (positive, negative, neutral, mixed)
            
        Returns:
            str: Hex color code
        """
        sentiment_map = {
            "positive": self.success_green,
            "negative": self.danger_red,
            "neutral": self.neutral_gray,
            "mixed": self.warning_orange,
            "unknown": self.text_light
        }
        return sentiment_map.get(sentiment.lower(), self.primary_blue)
    
    def get_categorical_palette(self, num_categories: int) -> List[str]:
        """
        Generate a categorical color palette.
        
        Args:
            num_categories (int): Number of categories needed
            
        Returns:
            List[str]: List of hex color codes
        """
        base_colors = [
            self.primary_blue, self.success_green, self.warning_orange,
            self.danger_red, self.neutral_gray
        ] + self.accent_colors
        
        # Repeat colors if more categories than available colors
        palette = []
        for i in range(num_categories):
            palette.append(base_colors[i % len(base_colors)])
        
        return palette


@dataclass
class ConfidenceThresholds:
    """
    Confidence score thresholds for automated decision making.
    
    Attributes:
        auto_approve (float): Threshold for automatic approval
        human_review (float): Threshold requiring human review
        reject (float): Threshold for automatic rejection
        warning_threshold (float): Threshold for warnings
        quality_gate (float): Minimum quality threshold
    """
    auto_approve: float = 0.85
    human_review: float = 0.7
    reject: float = 0.5
    warning_threshold: float = 0.6
    quality_gate: float = 0.75
    
    def get_status_by_confidence(self, confidence: float) -> ValidationStatus:
        """
        Determine validation status based on confidence score.
        
        Args:
            confidence (float): Confidence score (0.0-1.0)
            
        Returns:
            ValidationStatus: Appropriate validation status
        """
        if confidence >= self.auto_approve:
            return ValidationStatus.APPROVED
        elif confidence >= self.human_review:
            return ValidationStatus.REQUIRES_REVIEW
        elif confidence >= self.reject:
            return ValidationStatus.PENDING
        else:
            return ValidationStatus.FAILED
    
    def is_acceptable_quality(self, confidence: float) -> bool:
        """Check if confidence meets minimum quality threshold."""
        return confidence >= self.quality_gate


@dataclass
class ChartSpecification:
    """
    Detailed specification for chart types.
    
    Attributes:
        chart_type (ChartType): Type of chart
        title_template (str): Template for chart titles
        required_fields (List[str]): Required data fields
        optional_fields (List[str]): Optional data fields
        min_data_points (int): Minimum required data points
        max_data_points (int): Maximum recommended data points
        color_scheme (str): Preferred color scheme
        layout_config (Dict[str, Any]): Layout configuration
        validation_rules (Dict[str, Any]): Validation rules
    """
    chart_type: ChartType
    title_template: str
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    min_data_points: int = 1
    max_data_points: int = 1000
    color_scheme: str = "categorical"
    layout_config: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class SystemConfig:
    """
    Comprehensive system configuration for enterprise consulting tool.
    
    Centralizes all configuration settings including visualization parameters,
    professional styling, confidence thresholds, chart specifications, file
    paths, and enterprise-grade logging configuration.
    
    Attributes:
        visualization_settings (VisualizationSettings): Visualization configuration
        color_palette (ProfessionalColorPalette): Professional color scheme
        confidence_thresholds (ConfidenceThresholds): Decision thresholds
        chart_specifications (Dict[ChartType, ChartSpecification]): Chart configs
        file_paths (Dict[str, Path]): System file paths
        logging_config (Dict[str, Any]): Logging configuration
        environment (str): Current environment (dev, staging, prod)
        audit_enabled (bool): Enable comprehensive audit logging
        cache_enabled (bool): Enable caching for performance
        max_concurrent_jobs (int): Maximum concurrent processing jobs
    """
    
    def __init__(
        self,
        environment: str = "production",
        config_file: Optional[Path] = None,
        override_settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize system configuration.
        
        Args:
            environment (str): Environment name (dev, staging, production)
            config_file (Optional[Path]): Path to configuration file
            override_settings (Optional[Dict[str, Any]]): Setting overrides
        """
        self.environment = environment
        self.audit_enabled = True
        self.cache_enabled = True
        self.max_concurrent_jobs = 4
        
        # Initialize core configuration components
        self.visualization_settings = VisualizationSettings()
        self.color_palette = ProfessionalColorPalette()
        self.confidence_thresholds = ConfidenceThresholds()
        
        # Initialize system paths
        self._initialize_file_paths()
        
        # Setup chart specifications
        self._initialize_chart_specifications()
        
        # Configure logging
        self._initialize_logging_config()
        
        # Apply any configuration overrides
        if override_settings:
            self._apply_overrides(override_settings)
        
        # Load from file if provided
        if config_file and config_file.exists():
            self._load_from_file(config_file)
        
        # Setup logging
        self._setup_logging()
        
        logging.info(f"SystemConfig initialized for {self.environment} environment")
    
    def _initialize_file_paths(self) -> None:
        """Initialize system file paths based on environment."""
        base_dir = Path.cwd()
        
        if self.environment == "production":
            self.file_paths = {
                "output": base_dir / "output",
                "temp": base_dir / "temp",
                "logs": base_dir / "logs",
                "cache": base_dir / "cache",
                "exports": base_dir / "exports",
                "config": base_dir / "config",
                "data": base_dir / "data"
            }
        else:
            self.file_paths = {
                "output": base_dir / f"output_{self.environment}",
                "temp": base_dir / f"temp_{self.environment}",
                "logs": base_dir / f"logs_{self.environment}",
                "cache": base_dir / f"cache_{self.environment}",
                "exports": base_dir / f"exports_{self.environment}",
                "config": base_dir / "config",
                "data": base_dir / "data"
            }
        
        # Ensure directories exist
        for path in self.file_paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _initialize_chart_specifications(self) -> None:
        """Initialize specifications for all supported chart types."""
        self.chart_specifications = {
            ChartType.EMERGENT_THEMES: ChartSpecification(
                chart_type=ChartType.EMERGENT_THEMES,
                title_template="Emergent Themes Analysis - {source}",
                required_fields=["theme", "frequency", "confidence"],
                optional_fields=["sentiment", "source_count"],
                min_data_points=3,
                max_data_points=15,
                color_scheme="categorical",
                layout_config={
                    "orientation": "horizontal",
                    "show_percentages": True,
                    "sort_by": "frequency"
                },
                validation_rules={
                    "unique_themes": True,
                    "min_confidence": 0.6,
                    "require_positive_values": True
                }
            ),
            
            ChartType.GROUP_SENTIMENT: ChartSpecification(
                chart_type=ChartType.GROUP_SENTIMENT,
                title_template="Sentiment Analysis by {grouping_field}",
                required_fields=["group", "sentiment_score", "sample_size"],
                optional_fields=["department", "confidence"],
                min_data_points=2,
                max_data_points=20,
                color_scheme="sentiment",
                layout_config={
                    "show_neutral_line": True,
                    "group_bars": True,
                    "show_sample_sizes": True
                },
                validation_rules={
                    "sentiment_range": (-1.0, 1.0),
                    "min_sample_size": 5,
                    "unique_groups": True
                }
            ),
            
            ChartType.RISK_MATRIX: ChartSpecification(
                chart_type=ChartType.RISK_MATRIX,
                title_template="Risk Assessment Matrix - {assessment_type}",
                required_fields=["risk_item", "probability", "impact", "risk_level"],
                optional_fields=["mitigation_status", "owner", "timeline"],
                min_data_points=1,
                max_data_points=50,
                color_scheme="risk",
                layout_config={
                    "matrix_size": (5, 5),
                    "show_risk_levels": True,
                    "quadrant_colors": True,
                    "bubble_size_by": "impact"
                },
                validation_rules={
                    "probability_range": (0.0, 1.0),
                    "impact_range": (0.0, 1.0),
                    "valid_risk_levels": ["low", "medium", "high", "critical"]
                }
            ),
            
            ChartType.CULTURE_ANALYSIS: ChartSpecification(
                chart_type=ChartType.CULTURE_ANALYSIS,
                title_template="Organizational Culture Analysis - {dimension}",
                required_fields=["culture_dimension", "current_state", "desired_state"],
                optional_fields=["gap_analysis", "priority", "department"],
                min_data_points=4,
                max_data_points=12,
                color_scheme="diverging",
                layout_config={
                    "show_gap_analysis": True,
                    "radar_chart": True,
                    "comparison_view": True
                },
                validation_rules={
                    "state_range": (1.0, 5.0),
                    "unique_dimensions": True,
                    "require_both_states": True
                }
            )
        }
    
    def _initialize_logging_config(self) -> None:
        """Initialize enterprise-grade logging configuration."""
        self.logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(funcName)s - %(message)s"
                },
                "audit": {
                    "format": "%(asctime)s [AUDIT] %(name)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": str(self.file_paths["logs"] / "application.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                },
                "audit": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "audit",
                    "filename": str(self.file_paths["logs"] / "audit.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10
                }
            },
            "loggers": {
                "": {  # Root logger
                    "level": "INFO" if self.environment == "production" else "DEBUG",
                    "handlers": ["console", "file"]
                },
                "audit": {
                    "level": "INFO",
                    "handlers": ["audit"],
                    "propagate": False
                },
                "matplotlib": {
                    "level": "WARNING"
                },
                "pydantic": {
                    "level": "WARNING"
                }
            }
        }
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides."""
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key == "visualization":
                for vis_key, vis_value in value.items():
                    if hasattr(self.visualization_settings, vis_key):
                        setattr(self.visualization_settings, vis_key, vis_value)
            elif key == "confidence":
                for conf_key, conf_value in value.items():
                    if hasattr(self.confidence_thresholds, conf_key):
                        setattr(self.confidence_thresholds, conf_key, conf_value)
    
    def _load_from_file(self, config_file: Path) -> None:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                self._apply_overrides(config_data)
                
        except ImportError:
            logging.warning("PyYAML not available, skipping config file loading")
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    def _setup_logging(self) -> None:
        """Configure Python logging system."""
        import logging.config
        
        try:
            logging.config.dictConfig(self.logging_config)
        except Exception as e:
            # Fallback to basic configuration
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            logging.warning(f"Failed to configure advanced logging: {e}")
    
    def get_chart_specification(self, chart_type: ChartType) -> ChartSpecification:
        """
        Get chart specification for a given chart type.
        
        Args:
            chart_type (ChartType): Chart type to get specification for
            
        Returns:
            ChartSpecification: Chart configuration
            
        Raises:
            ValueError: If chart type is not supported
        """
        if chart_type not in self.chart_specifications:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        return self.chart_specifications[chart_type]
    
    def get_output_path(self, filename: str, subdirectory: Optional[str] = None) -> Path:
        """
        Generate output file path with proper directory structure.
        
        Args:
            filename (str): Output filename
            subdirectory (Optional[str]): Optional subdirectory
            
        Returns:
            Path: Complete output file path
        """
        if subdirectory:
            output_dir = self.file_paths["output"] / subdirectory
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / filename
        
        return self.file_paths["output"] / filename
    
    def get_temp_path(self, filename: str) -> Path:
        """
        Generate temporary file path.
        
        Args:
            filename (str): Temporary filename
            
        Returns:
            Path: Complete temporary file path
        """
        return self.file_paths["temp"] / filename
    
    def get_cache_path(self, cache_key: str) -> Path:
        """
        Generate cache file path.
        
        Args:
            cache_key (str): Cache identifier
            
        Returns:
            Path: Complete cache file path
        """
        return self.file_paths["cache"] / f"{cache_key}.cache"
    
    def is_production_environment(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def get_matplotlib_style_config(self) -> Dict[str, Any]:
        """
        Generate matplotlib style configuration.
        
        Returns:
            Dict[str, Any]: Matplotlib rcParams configuration
        """
        return {
            "figure.figsize": (
                self.visualization_settings.width / self.visualization_settings.dpi,
                self.visualization_settings.height / self.visualization_settings.dpi
            ),
            "figure.dpi": self.visualization_settings.dpi,
            "savefig.dpi": self.visualization_settings.dpi,
            "figure.facecolor": self.color_palette.background_white,
            "axes.facecolor": self.color_palette.background_white,
            "text.color": self.color_palette.text_dark,
            "axes.labelcolor": self.color_palette.text_dark,
            "xtick.color": self.color_palette.text_dark,
            "ytick.color": self.color_palette.text_dark,
            "font.family": self.visualization_settings.font_family,
            "font.size": self.visualization_settings.font_size_base,
            "axes.titlesize": self.visualization_settings.font_size_base + 4,
            "axes.labelsize": self.visualization_settings.font_size_base,
            "xtick.labelsize": self.visualization_settings.font_size_base - 1,
            "ytick.labelsize": self.visualization_settings.font_size_base - 1,
            "legend.fontsize": self.visualization_settings.font_size_base - 1,
            "lines.linewidth": self.visualization_settings.line_width,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "legend.frameon": True,
            "legend.fancybox": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": self.visualization_settings.figure_padding
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate system configuration and return any issues found.
        
        Returns:
            List[str]: List of configuration issues, empty if valid
        """
        issues = []
        
        # Validate visualization settings
        if self.visualization_settings.dpi < 150:
            issues.append(f"DPI too low for professional output: {self.visualization_settings.dpi}")
        
        if self.visualization_settings.width < 800 or self.visualization_settings.height < 600:
            issues.append("Chart dimensions too small for professional output")
        
        # Validate confidence thresholds
        thresholds = [
            self.confidence_thresholds.reject,
            self.confidence_thresholds.human_review,
            self.confidence_thresholds.auto_approve
        ]
        
        if not all(0.0 <= t <= 1.0 for t in thresholds):
            issues.append("Confidence thresholds must be between 0.0 and 1.0")
        
        if not (thresholds[0] <= thresholds[1] <= thresholds[2]):
            issues.append("Confidence thresholds must be in ascending order")
        
        # Validate file paths
        for path_name, path in self.file_paths.items():
            if not path.exists():
                issues.append(f"Required directory does not exist: {path_name} ({path})")
        
        return issues
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"SystemConfig("
            f"environment={self.environment}, "
            f"dpi={self.visualization_settings.dpi}, "
            f"audit_enabled={self.audit_enabled})"
        )