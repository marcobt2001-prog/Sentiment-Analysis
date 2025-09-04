"""
Enterprise verification engine for chart validation and feedback management.

This module provides comprehensive verification capabilities including data
validation, source traceability, confidence threshold enforcement, feedback
capture, and CSV export functionality for enterprise audit and compliance.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import pandas as pd

from .config import SystemConfig
from .models import (
    ChartData, 
    VerificationReport, 
    FeedbackData, 
    ValidationStatus,
    ValidationError,
    VerificationError
)

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")


class ValidationResult:
    """
    Result container for validation operations.
    
    Attributes:
        passed (bool): Whether validation passed
        score (float): Validation confidence score (0.0-1.0)
        issues (List[str]): List of identified issues
        recommendations (List[str]): Recommended actions
        metadata (Dict[str, Any]): Additional validation metadata
    """
    
    def __init__(
        self,
        passed: bool,
        score: float = 0.0,
        issues: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.passed = passed
        self.score = max(0.0, min(1.0, score))  # Clamp to valid range
        self.issues = issues or []
        self.recommendations = recommendations or []
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"ValidationResult({status}, score={self.score:.3f}, issues={len(self.issues)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class VerificationEngine:
    """
    Enterprise-grade verification engine for chart validation and quality assurance.
    
    Provides comprehensive validation capabilities including data integrity checks,
    source traceability verification, confidence threshold enforcement, feedback
    management, and audit trail generation for enterprise consulting environments.
    
    Attributes:
        config (SystemConfig): System configuration instance
        feedback_storage (List[FeedbackData]): In-memory feedback storage
        verification_cache (Dict[str, VerificationReport]): Cached verification results
        audit_trail (List[Dict[str, Any]]): Comprehensive audit trail
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the verification engine.
        
        Args:
            config (Optional[SystemConfig]): System configuration instance
        """
        self.config = config or SystemConfig()
        self.feedback_storage: List[FeedbackData] = []
        self.verification_cache: Dict[str, VerificationReport] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Initialize verification metrics tracking
        self.verification_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "human_reviews_required": 0,
            "average_confidence": 0.0
        }
        
        audit_logger.info("VerificationEngine initialized")
        logger.info(f"VerificationEngine ready with {len(self.config.chart_specifications)} chart types")
    
    def validate_chart_data(self, data: ChartData) -> VerificationReport:
        """
        Comprehensive validation of chart data with detailed reporting.
        
        Performs extensive validation including data integrity checks, format
        validation, chart type compatibility, confidence analysis, and business
        rule enforcement to ensure enterprise-grade chart quality.
        
        Args:
            data (ChartData): Chart data to validate
            
        Returns:
            VerificationReport: Comprehensive validation report
            
        Raises:
            VerificationError: If critical validation process fails
        """
        validation_start = datetime.now()
        
        try:
            # Initialize verification report
            report = VerificationReport(
                chart_id=data.chart_id,
                validation_status=ValidationStatus.PENDING,
                timestamp=validation_start
            )
            
            audit_logger.info(f"Starting validation for chart {data.chart_id} ({data.chart_type.value})")
            
            # Track total number of checks
            total_checks = 0
            passed_checks = 0
            
            # 1. Basic data integrity validation
            integrity_result = self._validate_data_integrity(data)
            total_checks += 1
            if integrity_result.passed:
                passed_checks += 1
            else:
                for issue in integrity_result.issues:
                    report.add_issue(issue, "high")
            
            # 2. Chart type specific validation
            chart_type_result = self._validate_chart_type_compatibility(data)
            total_checks += 1
            if chart_type_result.passed:
                passed_checks += 1
            else:
                for issue in chart_type_result.issues:
                    report.add_issue(issue, "medium")
            
            # 3. Confidence score analysis
            confidence_result = self._validate_confidence_scores(data)
            total_checks += 1
            if confidence_result.passed:
                passed_checks += 1
            else:
                for issue in confidence_result.issues:
                    report.add_issue(issue, "medium")
            
            # 4. Source traceability verification
            traceability_result = self._validate_source_traceability(data)
            total_checks += 1
            if traceability_result.passed:
                passed_checks += 1
            else:
                for issue in traceability_result.issues:
                    report.add_issue(issue, "high")
            
            # 5. Business rule compliance
            compliance_result = self._validate_business_rules(data)
            total_checks += 1
            if compliance_result.passed:
                passed_checks += 1
            else:
                for issue in compliance_result.issues:
                    report.add_issue(issue, "critical")
            
            # 6. Data quality assessment
            quality_result = self._assess_data_quality(data)
            total_checks += 1
            if quality_result.passed:
                passed_checks += 1
            else:
                for issue in quality_result.issues:
                    report.add_issue(issue, "low")
            
            # Update report with check results
            report.total_automated_checks = total_checks
            report.automated_checks_passed = passed_checks
            
            # Add compliance flags
            report.add_compliance_flag("data_integrity", integrity_result.passed)
            report.add_compliance_flag("chart_compatibility", chart_type_result.passed)
            report.add_compliance_flag("confidence_thresholds", confidence_result.passed)
            report.add_compliance_flag("source_traceability", traceability_result.passed)
            report.add_compliance_flag("business_rules", compliance_result.passed)
            report.add_compliance_flag("data_quality", quality_result.passed)
            
            # Generate recommendations
            self._generate_recommendations(report, data)
            
            # Cache the verification result
            self.verification_cache[str(data.chart_id)] = report
            
            # Update statistics
            self.verification_stats["total_validations"] += 1
            if report.validation_status in [ValidationStatus.APPROVED, ValidationStatus.VALIDATED]:
                self.verification_stats["passed_validations"] += 1
            else:
                self.verification_stats["failed_validations"] += 1
            
            if report.human_review_required:
                self.verification_stats["human_reviews_required"] += 1
            
            # Update average confidence
            avg_confidence = sum(data.confidence_scores) / len(data.confidence_scores)
            current_avg = self.verification_stats["average_confidence"]
            total_validations = self.verification_stats["total_validations"]
            self.verification_stats["average_confidence"] = (
                (current_avg * (total_validations - 1) + avg_confidence) / total_validations
            )
            
            # Add to audit trail
            self.audit_trail.append({
                "timestamp": validation_start.isoformat(),
                "operation": "validate_chart_data",
                "chart_id": str(data.chart_id),
                "chart_type": data.chart_type.value,
                "validation_status": report.validation_status.value,
                "verification_score": report.verification_score,
                "issues_count": len(report.issues_found),
                "human_review_required": report.human_review_required,
                "duration_seconds": (datetime.now() - validation_start).total_seconds()
            })
            
            audit_logger.info(f"Validation completed for chart {data.chart_id}: "
                             f"{report.validation_status.value} (score: {report.verification_score:.3f})")
            
            return report
            
        except Exception as e:
            logger.error(f"Validation failed for chart {data.chart_id}: {e}")
            audit_logger.error(f"Validation error for chart {data.chart_id}: {e}")
            raise VerificationError(f"Chart validation failed: {e}")
    
    def _validate_data_integrity(self, data: ChartData) -> ValidationResult:
        """Validate basic data integrity and consistency."""
        issues = []
        score = 1.0
        
        # Check array length consistency (already validated by Pydantic, but double-check)
        arrays = [data.values, data.labels, data.source_ids, data.confidence_scores]
        lengths = [len(arr) for arr in arrays]
        
        if len(set(lengths)) > 1:
            issues.append(f"Inconsistent array lengths: {lengths}")
            score -= 0.5
        
        # Check for null/empty values
        if any(v is None for v in data.values):
            issues.append("Data contains null values")
            score -= 0.3
        
        if any(not label.strip() for label in data.labels):
            issues.append("Data contains empty labels")
            score -= 0.2
        
        # Check for duplicate labels (might be valid for some chart types)
        if len(data.labels) != len(set(data.labels)) and data.chart_type.value not in ['line', 'scatter']:
            issues.append("Data contains duplicate labels")
            score -= 0.1
        
        # Check for extreme outliers
        if len(data.values) > 1:
            values_array = [v for v in data.values if v is not None]
            if values_array:
                mean_val = sum(values_array) / len(values_array)
                std_val = (sum((x - mean_val) ** 2 for x in values_array) / len(values_array)) ** 0.5
                
                if std_val > 0:
                    outliers = [v for v in values_array if abs(v - mean_val) > 3 * std_val]
                    if len(outliers) > len(values_array) * 0.1:  # More than 10% outliers
                        issues.append(f"High number of statistical outliers detected: {len(outliers)}")
                        score -= 0.1
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            recommendations=["Review data collection process", "Validate data sources"] if issues else []
        )
    
    def _validate_chart_type_compatibility(self, data: ChartData) -> ValidationResult:
        """Validate data compatibility with specified chart type."""
        issues = []
        score = 1.0
        
        try:
            chart_spec = self.config.get_chart_specification(data.chart_type)
            
            # Check minimum data points
            if len(data.values) < chart_spec.min_data_points:
                issues.append(f"Insufficient data points: {len(data.values)} < {chart_spec.min_data_points}")
                score -= 0.5
            
            # Check maximum data points
            if len(data.values) > chart_spec.max_data_points:
                issues.append(f"Too many data points: {len(data.values)} > {chart_spec.max_data_points}")
                score -= 0.2
            
            # Apply chart-specific validation rules
            validation_rules = chart_spec.validation_rules
            
            if "require_positive_values" in validation_rules and validation_rules["require_positive_values"]:
                negative_values = [v for v in data.values if v < 0]
                if negative_values:
                    issues.append(f"Chart type requires positive values, found {len(negative_values)} negative values")
                    score -= 0.3
            
            if "unique_labels" in validation_rules and validation_rules["unique_labels"]:
                if len(data.labels) != len(set(data.labels)):
                    issues.append("Chart type requires unique labels")
                    score -= 0.4
            
            if "sentiment_range" in validation_rules:
                min_val, max_val = validation_rules["sentiment_range"]
                out_of_range = [v for v in data.values if not (min_val <= v <= max_val)]
                if out_of_range:
                    issues.append(f"Values outside sentiment range {min_val}-{max_val}: {len(out_of_range)} values")
                    score -= 0.3
            
            if "min_confidence" in validation_rules:
                min_confidence = validation_rules["min_confidence"]
                low_confidence = [c for c in data.confidence_scores if c < min_confidence]
                if low_confidence:
                    issues.append(f"Low confidence scores below {min_confidence}: {len(low_confidence)} values")
                    score -= 0.2
            
        except ValueError as e:
            issues.append(f"Unsupported chart type: {e}")
            score = 0.0
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            recommendations=["Review chart type selection", "Adjust data preparation"] if issues else []
        )
    
    def _validate_confidence_scores(self, data: ChartData) -> ValidationResult:
        """Validate confidence scores against system thresholds."""
        issues = []
        score = 1.0
        
        thresholds = self.config.confidence_thresholds
        
        # Check individual confidence scores
        low_confidence_count = sum(1 for c in data.confidence_scores if c < thresholds.reject)
        review_required_count = sum(1 for c in data.confidence_scores if c < thresholds.human_review)
        
        if low_confidence_count > 0:
            issues.append(f"{low_confidence_count} data points below rejection threshold ({thresholds.reject})")
            score -= 0.4
        
        if review_required_count > len(data.confidence_scores) * 0.5:
            issues.append(f"Over 50% of data points require human review (threshold: {thresholds.human_review})")
            score -= 0.3
        
        # Calculate overall confidence
        avg_confidence = sum(data.confidence_scores) / len(data.confidence_scores)
        
        if avg_confidence < thresholds.quality_gate:
            issues.append(f"Average confidence {avg_confidence:.3f} below quality gate {thresholds.quality_gate}")
            score -= 0.5
        
        # Check confidence score distribution
        confidence_std = (sum((c - avg_confidence) ** 2 for c in data.confidence_scores) / len(data.confidence_scores)) ** 0.5
        
        if confidence_std > 0.3:  # High variability in confidence
            issues.append(f"High variability in confidence scores (std: {confidence_std:.3f})")
            score -= 0.2
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            recommendations=["Review data sources", "Improve data collection"] if issues else [],
            metadata={"average_confidence": avg_confidence, "confidence_std": confidence_std}
        )
    
    def _validate_source_traceability(self, data: ChartData) -> ValidationResult:
        """Validate source traceability and data lineage."""
        issues = []
        score = 1.0
        
        # Check for empty source IDs
        empty_sources = sum(1 for source_id in data.source_ids if not source_id.strip())
        if empty_sources > 0:
            issues.append(f"{empty_sources} data points have empty source IDs")
            score -= 0.5
        
        # Check source ID format (should be valid identifiers)
        invalid_sources = []
        for source_id in data.source_ids:
            if source_id.strip():
                # Basic format validation (can be enhanced based on requirements)
                if len(source_id) < 3 or not source_id.replace('-', '').replace('_', '').isalnum():
                    invalid_sources.append(source_id)
        
        if invalid_sources:
            issues.append(f"{len(invalid_sources)} invalid source ID formats")
            score -= 0.3
        
        # Check for source diversity (avoid single source dominance)
        unique_sources = set(data.source_ids)
        source_diversity = len(unique_sources) / len(data.source_ids)
        
        if source_diversity < 0.1:  # Less than 10% diversity
            issues.append(f"Low source diversity: {len(unique_sources)} unique sources for {len(data.source_ids)} data points")
            score -= 0.2
        
        # Validate data source summary consistency
        if data.data_source_summary:
            expected_count = data.data_source_summary.get("unique_source_count", 0)
            if expected_count != len(unique_sources):
                issues.append("Data source summary inconsistent with actual unique source count")
                score -= 0.1
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            recommendations=["Improve source ID formatting", "Diversify data sources"] if issues else [],
            metadata={"unique_sources": len(unique_sources), "source_diversity": source_diversity}
        )
    
    def _validate_business_rules(self, data: ChartData) -> ValidationResult:
        """Validate against business rules and compliance requirements."""
        issues = []
        score = 1.0
        
        # Rule 1: Title must be meaningful and not generic
        generic_titles = ["untitled", "chart", "graph", "visualization", "data"]
        if any(generic in data.title.lower() for generic in generic_titles):
            issues.append("Chart title appears to be generic or placeholder")
            score -= 0.2
        
        # Rule 2: Chart must have proper axis labels for applicable types
        chart_types_requiring_labels = ["bar", "line", "scatter", "group_sentiment", "risk_matrix"]
        if data.chart_type.value in chart_types_requiring_labels:
            if data.x_axis_label.lower() in ["category", "x", "x-axis"]:
                issues.append("X-axis label appears to be generic")
                score -= 0.1
            
            if data.y_axis_label.lower() in ["value", "y", "y-axis"]:
                issues.append("Y-axis label appears to be generic")
                score -= 0.1
        
        # Rule 3: Risk matrix specific rules
        if data.chart_type == data.chart_type.RISK_MATRIX:
            # Values should be probabilities (0-1) or impacts (0-1)
            invalid_risk_values = [v for v in data.values if not (0.0 <= v <= 1.0)]
            if invalid_risk_values:
                issues.append(f"Risk matrix values must be between 0-1, found {len(invalid_risk_values)} invalid values")
                score -= 0.4
        
        # Rule 4: Sentiment analysis rules
        if data.chart_type == data.chart_type.GROUP_SENTIMENT:
            # Values should be sentiment scores (-1 to 1)
            invalid_sentiment = [v for v in data.values if not (-1.0 <= v <= 1.0)]
            if invalid_sentiment:
                issues.append(f"Sentiment scores must be between -1 and 1, found {len(invalid_sentiment)} invalid values")
                score -= 0.4
        
        # Rule 5: Culture analysis rules
        if data.chart_type == data.chart_type.CULTURE_ANALYSIS:
            # Values should be in scale range (typically 1-5)
            invalid_culture_values = [v for v in data.values if not (1.0 <= v <= 5.0)]
            if invalid_culture_values:
                issues.append(f"Culture analysis values must be between 1-5, found {len(invalid_culture_values)} invalid values")
                score -= 0.4
        
        # Rule 6: Minimum confidence for business use
        business_min_confidence = 0.6
        low_business_confidence = sum(1 for c in data.confidence_scores if c < business_min_confidence)
        if low_business_confidence > len(data.confidence_scores) * 0.25:  # More than 25%
            issues.append(f"Too many data points below business confidence threshold: {low_business_confidence}")
            score -= 0.3
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            recommendations=["Review business requirements", "Improve data labeling"] if issues else []
        )
    
    def _assess_data_quality(self, data: ChartData) -> ValidationResult:
        """Assess overall data quality and completeness."""
        issues = []
        score = 1.0
        quality_metrics = {}
        
        # Completeness assessment
        completeness_score = 1.0
        if not data.subtitle:
            completeness_score -= 0.1
        if not data.metadata:
            completeness_score -= 0.1
        
        quality_metrics["completeness"] = completeness_score
        
        # Consistency assessment
        consistency_score = 1.0
        
        # Check label-value alignment
        if len(data.labels) != len(data.values):
            consistency_score -= 0.5
        
        # Check confidence-data alignment
        high_conf_low_val = sum(1 for i, (conf, val) in enumerate(zip(data.confidence_scores, data.values)) 
                               if conf > 0.9 and abs(val) < 0.1)
        if high_conf_low_val > len(data.values) * 0.2:
            issues.append("Inconsistency: high confidence with low values")
            consistency_score -= 0.2
        
        quality_metrics["consistency"] = consistency_score
        
        # Accuracy assessment (based on confidence scores)
        avg_confidence = sum(data.confidence_scores) / len(data.confidence_scores)
        accuracy_score = avg_confidence
        quality_metrics["accuracy"] = accuracy_score
        
        # Timeliness assessment (data freshness)
        data_age_hours = (datetime.now() - data.created_timestamp).total_seconds() / 3600
        if data_age_hours > 168:  # More than a week old
            issues.append(f"Data is {data_age_hours:.1f} hours old")
            score -= 0.1
        
        timeliness_score = max(0.0, 1.0 - (data_age_hours / (24 * 30)))  # Decay over 30 days
        quality_metrics["timeliness"] = timeliness_score
        
        # Overall quality score
        overall_quality = (completeness_score + consistency_score + accuracy_score + timeliness_score) / 4
        score = min(score, overall_quality)
        
        if overall_quality < 0.7:
            issues.append(f"Overall data quality score below threshold: {overall_quality:.3f}")
        
        return ValidationResult(
            passed=overall_quality >= 0.7,
            score=max(0.0, score),
            issues=issues,
            recommendations=["Improve data collection processes"] if issues else [],
            metadata=quality_metrics
        )
    
    def _generate_recommendations(self, report: VerificationReport, data: ChartData) -> None:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        if report.verification_score < 0.7:
            recommendations.append("Review and improve data collection methodology")
        
        if report.human_review_required:
            recommendations.append("Schedule human expert review before publication")
        
        if len(report.issues_found) > 5:
            recommendations.append("Consider data preprocessing to address multiple quality issues")
        
        # Chart-specific recommendations
        if data.chart_type == data.chart_type.RISK_MATRIX and any("risk" in issue.lower() for issue in report.issues_found):
            recommendations.append("Validate risk assessment methodology and probability calculations")
        
        if data.chart_type == data.chart_type.GROUP_SENTIMENT and any("sentiment" in issue.lower() for issue in report.issues_found):
            recommendations.append("Review sentiment analysis algorithm and training data")
        
        # Confidence-based recommendations
        avg_confidence = sum(data.confidence_scores) / len(data.confidence_scores)
        if avg_confidence < self.config.confidence_thresholds.human_review:
            recommendations.append("Increase sample size or improve data sources to boost confidence")
        
        report.recommended_actions.extend(recommendations)
    
    def check_source_traceability(self, chart_data: ChartData) -> bool:
        """
        Simplified source traceability check.
        
        Args:
            chart_data (ChartData): Chart data to verify
            
        Returns:
            bool: True if source traceability is acceptable
        """
        try:
            # Check that all data points have valid source IDs
            if not all(source_id.strip() for source_id in chart_data.source_ids):
                return False
            
            # Check that we have reasonable source diversity
            unique_sources = len(set(chart_data.source_ids))
            total_points = len(chart_data.source_ids)
            
            # Require at least 10% source diversity or minimum 2 sources
            min_diversity = max(2, total_points * 0.1)
            
            return unique_sources >= min_diversity
            
        except Exception as e:
            logger.error(f"Source traceability check failed: {e}")
            return False
    
    def enforce_confidence_thresholds(self, data: ChartData) -> ValidationResult:
        """
        Enforce confidence thresholds and determine approval status.
        
        Args:
            data (ChartData): Chart data to evaluate
            
        Returns:
            ValidationResult: Threshold enforcement result
        """
        try:
            thresholds = self.config.confidence_thresholds
            avg_confidence = sum(data.confidence_scores) / len(data.confidence_scores)
            
            issues = []
            recommendations = []
            
            # Count scores in each threshold range
            rejected_count = sum(1 for score in data.confidence_scores if score < thresholds.reject)
            review_count = sum(1 for score in data.confidence_scores if score < thresholds.human_review)
            approved_count = sum(1 for score in data.confidence_scores if score >= thresholds.auto_approve)
            
            # Determine overall status
            if rejected_count > 0:
                issues.append(f"{rejected_count} data points below rejection threshold ({thresholds.reject})")
                recommendations.append("Remove or improve low-confidence data points")
            
            if review_count > len(data.confidence_scores) * 0.3:  # More than 30% need review
                issues.append("High proportion of data points require human review")
                recommendations.append("Consider additional data validation steps")
            
            # Overall assessment
            if avg_confidence >= thresholds.auto_approve:
                status = "auto_approved"
                passed = True
            elif avg_confidence >= thresholds.human_review:
                status = "requires_review"
                passed = True
                recommendations.append("Schedule human review before final approval")
            elif avg_confidence >= thresholds.reject:
                status = "conditional"
                passed = False
                recommendations.append("Improve data quality before resubmission")
            else:
                status = "rejected"
                passed = False
                recommendations.append("Substantial data quality improvements required")
            
            audit_logger.info(f"Confidence threshold enforcement for chart {data.chart_id}: "
                             f"{status} (avg confidence: {avg_confidence:.3f})")
            
            return ValidationResult(
                passed=passed,
                score=avg_confidence,
                issues=issues,
                recommendations=recommendations,
                metadata={
                    "status": status,
                    "average_confidence": avg_confidence,
                    "rejected_count": rejected_count,
                    "review_count": review_count,
                    "approved_count": approved_count
                }
            )
            
        except Exception as e:
            logger.error(f"Confidence threshold enforcement failed: {e}")
            return ValidationResult(
                passed=False,
                score=0.0,
                issues=[f"Threshold enforcement error: {e}"],
                recommendations=["Review system configuration and data format"]
            )
    
    def capture_feedback(self, feedback: FeedbackData) -> None:
        """
        Capture and store user feedback for continuous improvement.
        
        Args:
            feedback (FeedbackData): User feedback data
        """
        try:
            # Validate feedback data
            if not isinstance(feedback, FeedbackData):
                raise ValueError("Invalid feedback data type")
            
            # Store feedback
            self.feedback_storage.append(feedback)
            
            # Log for audit trail
            audit_logger.info(f"Feedback captured for chart {feedback.chart_id} "
                             f"by {feedback.consultant_id}: rating={feedback.rating}")
            
            # Add to audit trail
            self.audit_trail.append({
                "timestamp": feedback.timestamp.isoformat(),
                "operation": "capture_feedback",
                "chart_id": str(feedback.chart_id),
                "consultant_id": feedback.consultant_id,
                "rating": feedback.rating,
                "feedback_category": feedback.feedback_category,
                "actionable": feedback.actionable,
                "priority": feedback.priority
            })
            
            logger.info(f"Feedback captured successfully for chart {feedback.chart_id}")
            
        except Exception as e:
            logger.error(f"Failed to capture feedback: {e}")
            raise VerificationError(f"Feedback capture failed: {e}")
    
    def generate_csv_export(self, chart_data: ChartData, output_path: str) -> str:
        """
        Generate comprehensive CSV export with full audit trail.
        
        Args:
            chart_data (ChartData): Chart data to export
            output_path (str): Output file path
            
        Returns:
            str: Path to the generated CSV file
            
        Raises:
            VerificationError: If export generation fails
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare export data
            export_data = []
            
            for i, (value, label, source_id, confidence) in enumerate(
                zip(chart_data.values, chart_data.labels, chart_data.source_ids, chart_data.confidence_scores)
            ):
                row = {
                    "chart_id": str(chart_data.chart_id),
                    "chart_type": chart_data.chart_type.value,
                    "chart_title": chart_data.title,
                    "data_point_index": i,
                    "label": label,
                    "value": value,
                    "source_id": source_id,
                    "confidence_score": confidence,
                    "x_axis_label": chart_data.x_axis_label,
                    "y_axis_label": chart_data.y_axis_label,
                    "created_timestamp": chart_data.created_timestamp.isoformat(),
                    "export_timestamp": datetime.now().isoformat()
                }
                
                # Add metadata fields
                for key, meta_value in chart_data.metadata.items():
                    row[f"metadata_{key}"] = str(meta_value)
                
                export_data.append(row)
            
            # Write CSV file
            if export_data:
                fieldnames = list(export_data[0].keys())
                
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(export_data)
            
            # Add summary sheet if using Excel format
            if output_file.suffix.lower() in ['.xlsx', '.xls']:
                self._add_excel_summary_sheet(output_file, chart_data)
            
            # Log export activity
            audit_logger.info(f"CSV export generated for chart {chart_data.chart_id}: {output_file}")
            
            # Add to audit trail
            self.audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "operation": "generate_csv_export",
                "chart_id": str(chart_data.chart_id),
                "output_path": str(output_file),
                "data_points_exported": len(export_data),
                "file_size_bytes": output_file.stat().st_size if output_file.exists() else 0
            })
            
            logger.info(f"CSV export completed: {output_file} ({len(export_data)} rows)")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"CSV export failed for chart {chart_data.chart_id}: {e}")
            raise VerificationError(f"CSV export failed: {e}")
    
    def _add_excel_summary_sheet(self, file_path: Path, chart_data: ChartData) -> None:
        """Add summary sheet to Excel export."""
        try:
            # This would require openpyxl, adding basic structure
            summary_data = {
                "Chart Information": {
                    "Chart ID": str(chart_data.chart_id),
                    "Chart Type": chart_data.chart_type.value,
                    "Title": chart_data.title,
                    "Subtitle": chart_data.subtitle or "N/A",
                    "Created": chart_data.created_timestamp.isoformat(),
                    "Data Points": len(chart_data.values),
                    "Unique Sources": len(set(chart_data.source_ids))
                },
                "Quality Metrics": {
                    "Average Confidence": sum(chart_data.confidence_scores) / len(chart_data.confidence_scores),
                    "Min Confidence": min(chart_data.confidence_scores),
                    "Max Confidence": max(chart_data.confidence_scores),
                    "Min Value": min(chart_data.values),
                    "Max Value": max(chart_data.values),
                    "Value Range": max(chart_data.values) - min(chart_data.values)
                }
            }
            
            # In a full implementation, would write this to Excel
            logger.debug(f"Excel summary prepared for {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to add Excel summary sheet: {e}")
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive verification statistics.
        
        Returns:
            Dict[str, Any]: Verification statistics and metrics
        """
        stats = self.verification_stats.copy()
        
        # Add calculated metrics
        total_validations = stats["total_validations"]
        if total_validations > 0:
            stats["pass_rate"] = stats["passed_validations"] / total_validations
            stats["failure_rate"] = stats["failed_validations"] / total_validations
            stats["human_review_rate"] = stats["human_reviews_required"] / total_validations
        else:
            stats["pass_rate"] = 0.0
            stats["failure_rate"] = 0.0
            stats["human_review_rate"] = 0.0
        
        # Add feedback statistics
        stats["total_feedback_entries"] = len(self.feedback_storage)
        if self.feedback_storage:
            stats["average_rating"] = sum(f.rating for f in self.feedback_storage) / len(self.feedback_storage)
            stats["actionable_feedback_rate"] = sum(1 for f in self.feedback_storage if f.actionable) / len(self.feedback_storage)
        else:
            stats["average_rating"] = 0.0
            stats["actionable_feedback_rate"] = 0.0
        
        # Add cache statistics
        stats["cached_verifications"] = len(self.verification_cache)
        stats["audit_trail_entries"] = len(self.audit_trail)
        
        return stats
    
    def export_audit_trail(self, output_path: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> str:
        """
        Export comprehensive audit trail for compliance reporting.
        
        Args:
            output_path (str): Output file path
            start_date (Optional[datetime]): Start date filter
            end_date (Optional[datetime]): End date filter
            
        Returns:
            str: Path to exported audit trail file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Filter audit trail by date range if specified
            filtered_trail = self.audit_trail
            if start_date or end_date:
                filtered_trail = []
                for entry in self.audit_trail:
                    entry_date = datetime.fromisoformat(entry["timestamp"])
                    if start_date and entry_date < start_date:
                        continue
                    if end_date and entry_date > end_date:
                        continue
                    filtered_trail.append(entry)
            
            # Export as JSON for comprehensive data preservation
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_timestamp": datetime.now().isoformat(),
                    "total_entries": len(filtered_trail),
                    "date_range": {
                        "start": start_date.isoformat() if start_date else None,
                        "end": end_date.isoformat() if end_date else None
                    },
                    "audit_entries": filtered_trail,
                    "verification_statistics": self.get_verification_statistics()
                }, f, indent=2, default=str)
            
            logger.info(f"Audit trail exported: {output_file} ({len(filtered_trail)} entries)")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Audit trail export failed: {e}")
            raise VerificationError(f"Audit trail export failed: {e}")