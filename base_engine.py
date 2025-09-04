"""
Abstract base visualization engine for enterprise chart generation.

This module provides the foundational abstract class for all visualization engines,
ensuring consistent interfaces, professional styling, comprehensive validation,
and enterprise-grade metadata export capabilities.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import pandas as pd
import numpy as np

from ..core.config import SystemConfig
from ..core.models import ChartData, ValidationError
from ..core.verification import VerificationEngine, ValidationResult

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")


class VisualizationError(Exception):
    """Custom exception for visualization-related errors."""
    pass


class ChartGenerationError(VisualizationError):
    """Exception raised during chart generation process."""
    pass


class StyleConfigurationError(VisualizationError):
    """Exception raised during style configuration."""
    pass


class BaseVisualizationEngine(ABC):
    """
    Abstract base class for all enterprise visualization engines.
    
    Provides the foundational framework for chart generation with comprehensive
    validation, professional styling, metadata export, and audit trail capabilities.
    All concrete visualization implementations must inherit from this class.
    
    Attributes:
        config (SystemConfig): System-wide configuration instance
        verification_engine (VerificationEngine): Data verification engine
        generation_metadata (Dict[str, Any]): Chart generation metadata
        style_applied (bool): Flag indicating if professional styling is applied
        current_chart_data (Optional[ChartData]): Currently processed chart data
    """
    
    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        verification_engine: Optional[VerificationEngine] = None
    ):
        """
        Initialize the base visualization engine.
        
        Args:
            config (Optional[SystemConfig]): System configuration instance
            verification_engine (Optional[VerificationEngine]): Verification engine instance
        """
        self.config = config or SystemConfig()
        self.verification_engine = verification_engine or VerificationEngine(self.config)
        
        # Initialize metadata tracking
        self.generation_metadata: Dict[str, Any] = {
            "engine_class": self.__class__.__name__,
            "initialization_timestamp": datetime.now().isoformat(),
            "config_version": "1.0",
            "style_configuration": {},
            "generation_history": []
        }
        
        self.style_applied = False
        self.current_chart_data: Optional[ChartData] = None
        
        # Setup professional styling
        try:
            self.setup_professional_styling()
            self.style_applied = True
        except Exception as e:
            logger.warning(f"Failed to setup professional styling: {e}")
            self.style_applied = False
        
        audit_logger.info(f"{self.__class__.__name__} initialized")
        logger.info(f"BaseVisualizationEngine ready with {len(self.config.chart_specifications)} chart specifications")
    
    @abstractmethod
    def generate_chart(self, spec: ChartData) -> str:
        """
        Generate a professional chart based on the provided specification.
        
        This is the primary method that must be implemented by all concrete
        visualization engines. It should create a chart file and return the
        file path to the generated visualization.
        
        Args:
            spec (ChartData): Complete chart data specification including values,
                            labels, styling preferences, and metadata
                            
        Returns:
            str: File path to the generated chart image
            
        Raises:
            ChartGenerationError: If chart generation fails for any reason
            ValidationError: If the input specification is invalid
            VisualizationError: For any other visualization-related errors
            
        Note:
            Implementations must:
            1. Validate input data using validate_input_data()
            2. Apply professional styling
            3. Generate the chart using appropriate libraries
            4. Save with high-quality settings (300 DPI)
            5. Return the absolute file path to the generated chart
        """
        pass
    
    def validate_input_data(self, data: ChartData) -> bool:
        """
        Comprehensive validation of input chart data.
        
        Performs enterprise-grade validation including data integrity checks,
        chart type compatibility, business rule compliance, and source
        traceability verification to ensure reliable chart generation.
        
        Args:
            data (ChartData): Chart data to validate
            
        Returns:
            bool: True if data passes all validation checks
            
        Raises:
            ValidationError: If critical validation failures are detected
            VisualizationError: If validation process encounters system errors
        """
        validation_start = datetime.now()
        
        try:
            logger.info(f"Starting validation for chart {data.chart_id} ({data.chart_type.value})")
            
            # Store current chart data for reference
            self.current_chart_data = data
            
            # Perform comprehensive verification using the verification engine
            verification_report = self.verification_engine.validate_chart_data(data)
            
            # Check if validation passed
            validation_passed = verification_report.validation_status.value in [
                "approved", "validated", "pending"
            ]
            
            if not validation_passed:
                error_message = f"Data validation failed: {', '.join(verification_report.issues_found)}"
                logger.error(f"Validation failed for chart {data.chart_id}: {error_message}")
                raise ValidationError(error_message)
            
            # Additional engine-specific validations
            engine_validation_result = self._perform_engine_specific_validation(data)
            
            if not engine_validation_result.passed:
                error_message = f"Engine-specific validation failed: {', '.join(engine_validation_result.issues)}"
                logger.error(f"Engine validation failed for chart {data.chart_id}: {error_message}")
                raise ValidationError(error_message)
            
            # Record validation success
            validation_duration = (datetime.now() - validation_start).total_seconds()
            
            self.generation_metadata["generation_history"].append({
                "operation": "validate_input_data",
                "timestamp": validation_start.isoformat(),
                "chart_id": str(data.chart_id),
                "chart_type": data.chart_type.value,
                "validation_passed": True,
                "verification_score": verification_report.verification_score,
                "duration_seconds": validation_duration,
                "data_points": len(data.values),
                "unique_sources": len(set(data.source_ids))
            })
            
            audit_logger.info(f"Data validation successful for chart {data.chart_id} "
                             f"(score: {verification_report.verification_score:.3f}, "
                             f"duration: {validation_duration:.3f}s)")
            
            logger.info(f"Validation completed successfully for chart {data.chart_id}")
            
            return True
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Validation process failed for chart {data.chart_id if data else 'unknown'}: {e}")
            raise VisualizationError(f"Data validation process failed: {e}")
    
    def _perform_engine_specific_validation(self, data: ChartData) -> ValidationResult:
        """
        Perform visualization engine specific validation checks.
        
        Args:
            data (ChartData): Chart data to validate
            
        Returns:
            ValidationResult: Engine-specific validation results
        """
        issues = []
        score = 1.0
        
        # Check matplotlib compatibility
        try:
            # Test that we can create basic plot elements
            fig, ax = plt.subplots(figsize=(1, 1))
            plt.close(fig)
        except Exception as e:
            issues.append(f"Matplotlib compatibility issue: {e}")
            score -= 0.5
        
        # Validate chart dimensions
        if hasattr(data, 'metadata'):
            width = data.metadata.get('width', self.config.visualization_settings.width)
            height = data.metadata.get('height', self.config.visualization_settings.height)
            
            if width < 400 or height < 300:
                issues.append(f"Chart dimensions too small: {width}x{height}")
                score -= 0.3
            
            if width > 5000 or height > 5000:
                issues.append(f"Chart dimensions too large: {width}x{height}")
                score -= 0.2
        
        # Check for potential rendering issues
        if len(data.labels) > 50:
            issues.append("Large number of labels may cause rendering issues")
            score -= 0.1
        
        # Validate color requirements
        if data.chart_type.value in ['pie', 'bar'] and len(data.values) > 20:
            issues.append("Many data points may require custom color palette")
            score -= 0.1
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            recommendations=["Review chart specifications"] if issues else []
        )
    
    def export_chart_metadata(self, chart_id: str, output_path: str) -> None:
        """
        Export comprehensive chart metadata for audit and documentation.
        
        Generates detailed metadata including chart specifications, generation
        parameters, data quality metrics, validation results, and system
        configuration for enterprise audit trails and documentation.
        
        Args:
            chart_id (str): Unique identifier of the chart
            output_path (str): File path for metadata export
            
        Raises:
            VisualizationError: If metadata export fails
        """
        try:
            export_timestamp = datetime.now()
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Compile comprehensive metadata
            metadata = {
                "export_info": {
                    "timestamp": export_timestamp.isoformat(),
                    "chart_id": chart_id,
                    "engine_class": self.__class__.__name__,
                    "version": "1.0"
                },
                "chart_specification": self._get_chart_specification_metadata(),
                "generation_metadata": self.generation_metadata,
                "system_configuration": self._get_system_config_metadata(),
                "data_quality_metrics": self._get_data_quality_metadata(),
                "validation_results": self._get_validation_metadata(),
                "styling_information": self._get_styling_metadata()
            }
            
            # Write metadata to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
            
            # Log export activity
            audit_logger.info(f"Chart metadata exported for {chart_id}: {output_file}")
            
            # Update generation history
            self.generation_metadata["generation_history"].append({
                "operation": "export_chart_metadata",
                "timestamp": export_timestamp.isoformat(),
                "chart_id": chart_id,
                "output_path": str(output_file),
                "file_size_bytes": output_file.stat().st_size if output_file.exists() else 0
            })
            
            logger.info(f"Chart metadata exported successfully: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export chart metadata for {chart_id}: {e}")
            raise VisualizationError(f"Metadata export failed: {e}")
    
    def _get_chart_specification_metadata(self) -> Dict[str, Any]:
        """Get chart specification metadata."""
        if not self.current_chart_data:
            return {"status": "No current chart data available"}
        
        return {
            "chart_id": str(self.current_chart_data.chart_id),
            "chart_type": self.current_chart_data.chart_type.value,
            "title": self.current_chart_data.title,
            "subtitle": self.current_chart_data.subtitle,
            "data_points_count": len(self.current_chart_data.values),
            "unique_sources": len(set(self.current_chart_data.source_ids)),
            "x_axis_label": self.current_chart_data.x_axis_label,
            "y_axis_label": self.current_chart_data.y_axis_label,
            "created_timestamp": self.current_chart_data.created_timestamp.isoformat(),
            "metadata": self.current_chart_data.metadata
        }
    
    def _get_system_config_metadata(self) -> Dict[str, Any]:
        """Get system configuration metadata."""
        return {
            "environment": self.config.environment,
            "visualization_settings": {
                "dpi": self.config.visualization_settings.dpi,
                "chart_style": self.config.visualization_settings.chart_style,
                "output_format": self.config.visualization_settings.output_format.value,
                "width": self.config.visualization_settings.width,
                "height": self.config.visualization_settings.height,
                "font_family": self.config.visualization_settings.font_family
            },
            "color_palette": {
                "primary_blue": self.config.color_palette.primary_blue,
                "success_green": self.config.color_palette.success_green,
                "warning_orange": self.config.color_palette.warning_orange,
                "danger_red": self.config.color_palette.danger_red
            },
            "confidence_thresholds": {
                "auto_approve": self.config.confidence_thresholds.auto_approve,
                "human_review": self.config.confidence_thresholds.human_review,
                "reject": self.config.confidence_thresholds.reject
            },
            "audit_enabled": self.config.audit_enabled,
            "cache_enabled": self.config.cache_enabled
        }
    
    def _get_data_quality_metadata(self) -> Dict[str, Any]:
        """Get data quality metrics metadata."""
        if not self.current_chart_data:
            return {"status": "No current chart data available"}
        
        data = self.current_chart_data
        confidence_scores = data.confidence_scores
        values = data.values
        
        return {
            "confidence_metrics": {
                "average": sum(confidence_scores) / len(confidence_scores),
                "minimum": min(confidence_scores),
                "maximum": max(confidence_scores),
                "standard_deviation": np.std(confidence_scores) if len(confidence_scores) > 1 else 0.0
            },
            "value_metrics": {
                "count": len(values),
                "minimum": min(values),
                "maximum": max(values),
                "average": sum(values) / len(values),
                "standard_deviation": np.std(values) if len(values) > 1 else 0.0,
                "range": max(values) - min(values)
            },
            "source_metrics": {
                "total_sources": len(data.source_ids),
                "unique_sources": len(set(data.source_ids)),
                "source_diversity": len(set(data.source_ids)) / len(data.source_ids)
            }
        }
    
    def _get_validation_metadata(self) -> Dict[str, Any]:
        """Get validation results metadata."""
        return self.verification_engine.get_verification_statistics()
    
    def _get_styling_metadata(self) -> Dict[str, Any]:
        """Get styling configuration metadata."""
        return {
            "style_applied": self.style_applied,
            "matplotlib_backend": plt.get_backend(),
            "rcParams_sample": {
                "figure.dpi": plt.rcParams.get("figure.dpi", "not set"),
                "font.family": plt.rcParams.get("font.family", "not set"),
                "font.size": plt.rcParams.get("font.size", "not set"),
                "axes.grid": plt.rcParams.get("axes.grid", "not set")
            },
            "color_palette_applied": True,
            "professional_standards": {
                "dpi_300": self.config.visualization_settings.dpi >= 300,
                "vector_format_capable": True,
                "high_quality_fonts": True
            }
        }
    
    def setup_professional_styling(self) -> None:
        """
        Configure matplotlib with professional styling for enterprise presentations.
        
        Applies comprehensive styling including 300 DPI resolution, professional
        color schemes, appropriate fonts, grid styling, and export settings
        suitable for consulting and business presentations.
        
        Raises:
            StyleConfigurationError: If styling configuration fails
        """
        try:
            logger.info("Configuring professional matplotlib styling")
            
            # Get style configuration from system config
            style_config = self.config.get_matplotlib_style_config()
            
            # Apply matplotlib style configuration
            plt.rcParams.update(style_config)
            
            # Set professional color cycle
            color_palette = self.config.color_palette.get_categorical_palette(10)
            plt.rcParams['axes.prop_cycle'] = plt.cycler('color', color_palette)
            
            # Configure figure and axes styling
            plt.rcParams.update({
                # High-quality rendering
                'figure.dpi': self.config.visualization_settings.dpi,
                'savefig.dpi': self.config.visualization_settings.dpi,
                'figure.autolayout': True,
                
                # Professional typography
                'font.family': [self.config.visualization_settings.font_family],
                'font.size': self.config.visualization_settings.font_size_base,
                'font.weight': 'normal',
                
                # Clean appearance
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.left': True,
                'axes.spines.bottom': True,
                
                # Grid styling
                'axes.grid': True,
                'axes.grid.axis': 'y',
                'grid.color': '#E0E0E0',
                'grid.linewidth': 0.5,
                'grid.alpha': 0.7,
                
                # Professional colors
                'axes.edgecolor': self.config.color_palette.neutral_gray,
                'axes.labelcolor': self.config.color_palette.text_dark,
                'text.color': self.config.color_palette.text_dark,
                'xtick.color': self.config.color_palette.text_dark,
                'ytick.color': self.config.color_palette.text_dark,
                
                # Legend styling
                'legend.frameon': True,
                'legend.framealpha': 0.9,
                'legend.facecolor': 'white',
                'legend.edgecolor': self.config.color_palette.neutral_gray,
                'legend.shadow': False,
                'legend.fancybox': True,
                
                # Export settings
                'savefig.bbox': 'tight',
                'savefig.pad_inches': self.config.visualization_settings.figure_padding,
                'savefig.transparent': False,
                'savefig.facecolor': 'white',
                'savefig.edgecolor': 'none'
            })
            
            # Store applied styling configuration
            self.generation_metadata["style_configuration"] = {
                "timestamp": datetime.now().isoformat(),
                "dpi": self.config.visualization_settings.dpi,
                "font_family": self.config.visualization_settings.font_family,
                "color_palette": color_palette[:5],  # Store first 5 colors for reference
                "style_template": self.config.visualization_settings.chart_style
            }
            
            logger.info(f"Professional styling configured: "
                       f"DPI={self.config.visualization_settings.dpi}, "
                       f"font={self.config.visualization_settings.font_family}, "
                       f"colors={len(color_palette)} colors")
            
        except Exception as e:
            logger.error(f"Failed to setup professional styling: {e}")
            raise StyleConfigurationError(f"Professional styling configuration failed: {e}")
    
    def _generate_output_filename(
        self, 
        chart_data: ChartData, 
        extension: str = None,
        include_timestamp: bool = True
    ) -> str:
        """
        Generate standardized output filename for charts.
        
        Args:
            chart_data (ChartData): Chart data for filename generation
            extension (str): File extension (defaults to config setting)
            include_timestamp (bool): Include timestamp in filename
            
        Returns:
            str: Generated filename
        """
        if extension is None:
            extension = self.config.visualization_settings.output_format.value
        
        # Sanitize chart title for filename
        safe_title = "".join(c for c in chart_data.title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
        
        # Build filename components
        filename_parts = [
            safe_title,
            chart_data.chart_type.value,
            str(chart_data.chart_id)[:8]  # First 8 characters of UUID
        ]
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_parts.append(timestamp)
        
        filename = "_".join(filename_parts) + f".{extension}"
        
        return filename
    
    def _save_chart_with_metadata(
        self, 
        figure, 
        chart_data: ChartData, 
        output_path: str
    ) -> str:
        """
        Save chart with comprehensive metadata and professional quality settings.
        
        Args:
            figure: Matplotlib figure object
            chart_data (ChartData): Chart data for metadata
            output_path (str): Output file path
            
        Returns:
            str: Path to saved file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine save parameters based on file format
            save_kwargs = {
                'dpi': self.config.visualization_settings.dpi,
                'bbox_inches': 'tight',
                'pad_inches': self.config.visualization_settings.figure_padding,
                'facecolor': 'white',
                'edgecolor': 'none'
            }
            
            # Format-specific metadata
            if output_file.suffix.lower() == '.pdf':
                save_kwargs['metadata'] = {
                    'Title': chart_data.title,
                    'Subject': f'{chart_data.chart_type.value} Chart',
                    'Creator': f'{self.__class__.__name__}',
                    'CreationDate': datetime.now(),
                    'Keywords': f"chart,{chart_data.chart_type.value},enterprise"
                }
            
            # Save the figure
            figure.savefig(output_file, **save_kwargs)
            
            # Log save operation
            file_size = output_file.stat().st_size if output_file.exists() else 0
            
            audit_logger.info(f"Chart saved: {output_file} "
                             f"({file_size} bytes, {self.config.visualization_settings.dpi} DPI)")
            
            # Update generation history
            self.generation_metadata["generation_history"].append({
                "operation": "save_chart",
                "timestamp": datetime.now().isoformat(),
                "chart_id": str(chart_data.chart_id),
                "output_path": str(output_file),
                "file_size_bytes": file_size,
                "dpi": self.config.visualization_settings.dpi,
                "format": output_file.suffix.lower()
            })
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save chart: {e}")
            raise VisualizationError(f"Chart save operation failed: {e}")
    
    def cleanup_resources(self) -> None:
        """Clean up visualization resources and temporary files."""
        try:
            # Close all matplotlib figures
            plt.close('all')
            
            # Clear current chart data
            self.current_chart_data = None
            
            # Log cleanup
            logger.debug(f"Resources cleaned up for {self.__class__.__name__}")
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive generation statistics and metrics.
        
        Returns:
            Dict[str, Any]: Generation statistics including performance metrics
        """
        history = self.generation_metadata.get("generation_history", [])
        
        # Calculate operation statistics
        operations = {}
        total_duration = 0.0
        
        for entry in history:
            operation = entry.get("operation", "unknown")
            duration = entry.get("duration_seconds", 0.0)
            
            if operation not in operations:
                operations[operation] = {"count": 0, "total_duration": 0.0}
            
            operations[operation]["count"] += 1
            operations[operation]["total_duration"] += duration
            total_duration += duration
        
        # Calculate averages
        for op_stats in operations.values():
            if op_stats["count"] > 0:
                op_stats["average_duration"] = op_stats["total_duration"] / op_stats["count"]
        
        return {
            "engine_class": self.__class__.__name__,
            "initialization_timestamp": self.generation_metadata.get("initialization_timestamp"),
            "total_operations": len(history),
            "total_duration_seconds": total_duration,
            "operations_breakdown": operations,
            "style_applied": self.style_applied,
            "current_chart_active": self.current_chart_data is not None,
            "verification_statistics": self.verification_engine.get_verification_statistics()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic resource cleanup."""
        self.cleanup_resources()
        
        if exc_type is not None:
            logger.error(f"Exception in visualization engine: {exc_type.__name__}: {exc_val}")
            audit_logger.error(f"Visualization error in {self.__class__.__name__}: {exc_val}")
        
        return False  # Don't suppress exceptions