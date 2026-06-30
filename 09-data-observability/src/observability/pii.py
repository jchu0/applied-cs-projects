"""PII detection for data assets."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from observability.collectors import MetadataCollector
from observability.models import TableMetadata

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ZIP_CODE = "zip_code"
    DATE_OF_BIRTH = "date_of_birth"


@dataclass
class PIIDetection:
    """Result of PII detection."""

    table_id: str
    column_name: str
    name_indicators: List[str]
    data_indicators: List[Tuple[str, float]]
    confidence: float
    pii_types: List[str] = field(default_factory=list)


class PIIDetector:
    """Detect PII in column names and data."""

    def __init__(self):
        self.patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?1?\d{9,15}$",
            "ssn": r"^\d{3}-\d{2}-\d{4}$",
            "credit_card": r"^\d{13,16}$",
            "ip_address": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
            "zip_code": r"^\d{5}(-\d{4})?$",
            "date_of_birth": r"^\d{4}-\d{2}-\d{2}$",
        }

        self.suspicious_names = [
            "ssn", "social_security", "social_security_number",
            "password", "pwd", "passwd", "secret", "token",
            "credit_card", "card_number", "cc_number", "cvv", "pin",
            "email", "email_address", "e_mail",
            "phone", "phone_number", "mobile", "cell",
            "address", "street", "city", "state", "zip", "postal",
            "dob", "birth_date", "date_of_birth", "birthdate",
            "first_name", "last_name", "full_name", "name",
            "ip_address", "ip", "user_agent",
            "location", "lat", "lng", "latitude", "longitude",
            "bank_account", "account_number", "routing_number",
            "passport", "drivers_license", "license_number",
        ]

        self.pii_type_map = {
            "ssn": "SSN",
            "social_security": "SSN",
            "credit_card": "Credit Card",
            "card_number": "Credit Card",
            "email": "Email",
            "phone": "Phone",
            "mobile": "Phone",
            "address": "Address",
            "name": "Name",
            "dob": "Date of Birth",
            "birth_date": "Date of Birth",
            "ip": "IP Address",
            "bank_account": "Financial",
            "passport": "Government ID",
            "drivers_license": "Government ID",
        }

        # Custom patterns added at runtime
        self.custom_patterns: Dict[str, Tuple[str, float]] = {}

    def detect(self, text: str) -> Dict[Union[PIIType, str], Dict[str, Any]]:
        """Detect PII in text and return dict of PIIType -> detection info."""
        if not text:
            return {}

        result = {}

        # More flexible patterns for detection in arbitrary text
        detection_patterns = {
            PIIType.EMAIL: r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            PIIType.PHONE: r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}',
            PIIType.SSN: r'\d{3}-\d{2}-\d{4}',
            PIIType.CREDIT_CARD: r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|\d{13,16}',
            PIIType.IP_ADDRESS: r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        }

        for pii_type, pattern in detection_patterns.items():
            if re.search(pattern, text):
                # Higher confidence for exact/structured matches
                confidence = 0.95 if pii_type in [PIIType.EMAIL, PIIType.SSN] else 0.9
                result[pii_type] = {"confidence": confidence, "match": re.search(pattern, text).group()}

        # Check custom patterns
        for name, (pattern, confidence) in self.custom_patterns.items():
            if re.search(pattern, text):
                result[name] = {"confidence": confidence, "match": re.search(pattern, text).group()}

        return result

    def scan_column(self, column_name: str, values: List[Any]) -> Dict[str, Any]:
        """Scan a column of values for PII."""
        if not values:
            return {
                "pii_detected": False,
                "pii_type": None,
                "pii_percentage": 0.0,
                "sample_matches": []
            }

        pii_counts: Dict[Union[PIIType, str], int] = {}
        sample_matches = []

        for value in values:
            if value is None:
                continue
            detections = self.detect(str(value))
            if detections:
                for pii_type in detections:
                    pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1
                if len(sample_matches) < 5:
                    sample_matches.append(str(value))

        if not pii_counts:
            return {
                "pii_detected": False,
                "pii_type": None,
                "pii_percentage": 0.0,
                "sample_matches": []
            }

        # Find the most common PII type
        most_common_type = max(pii_counts, key=pii_counts.get)
        pii_percentage = (pii_counts[most_common_type] / len(values)) * 100

        return {
            "pii_detected": True,
            "pii_type": most_common_type,
            "pii_percentage": pii_percentage,
            "sample_matches": sample_matches
        }

    def scan_table(self, columns_or_table_id, collector=None, sample_size=1000):
        """Scan table for PII. Supports both dict input and async collector API."""
        # Sync version for dict input (used by tests)
        if isinstance(columns_or_table_id, dict):
            result = {}
            for column_name, values in columns_or_table_id.items():
                result[column_name] = self.scan_column(column_name, values)
            return result
        # Async version with collector - handled by _scan_table_async
        raise ValueError("Use _scan_table_async for collector-based scanning")

    async def _scan_table_async(
        self,
        table_id: str,
        collector: MetadataCollector,
        sample_size: int = 1000,
    ) -> List[PIIDetection]:
        """Async version: Scan a table for PII using collector."""
        schema = await collector.collect_schema(table_id)
        detections = []

        for column in schema.columns:
            # Check column name
            name_indicators = self.detect_in_column_name(column.name)

            # Sample data
            sample = await collector.get_column_sample(
                table_id, column.name, sample_size
            )
            data_indicators = self.detect_in_sample(column.name, sample)

            if name_indicators or data_indicators:
                confidence = self._calculate_confidence(
                    name_indicators, data_indicators
                )
                pii_types = self._get_pii_types(
                    column.name, name_indicators, data_indicators
                )

                detections.append(PIIDetection(
                    table_id=table_id,
                    column_name=column.name,
                    name_indicators=name_indicators,
                    data_indicators=data_indicators,
                    confidence=confidence,
                    pii_types=pii_types,
                ))

        return detections

    def mask(self, text: str) -> str:
        """Mask PII in text."""
        if not text:
            return text

        result = text

        # Mask patterns
        mask_patterns = [
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]'),
            (r'\d{3}-\d{2}-\d{4}', '[SSN]'),
            (r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', '[CREDIT_CARD]'),
            (r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', '[PHONE]'),
            (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP]'),
        ]

        for pattern, replacement in mask_patterns:
            result = re.sub(pattern, replacement, result)

        return result

    def add_pattern(self, name: str, pattern: str, confidence: float = 0.9) -> None:
        """Add a custom PII pattern."""
        self.custom_patterns[name] = (pattern, confidence)

    def column_name_suspicion(self, column_name: str) -> float:
        """Return suspicion score (0-1) based on column name."""
        name_lower = column_name.lower()
        score = 0.0

        high_suspicion = ["ssn", "social_security", "password", "credit_card", "cvv", "pin"]
        medium_suspicion = ["email", "phone", "mobile", "address", "dob", "birth"]
        low_suspicion = ["name", "user", "ip", "location"]

        for term in high_suspicion:
            if term in name_lower:
                score = max(score, 0.9)

        for term in medium_suspicion:
            if term in name_lower:
                score = max(score, 0.7)

        for term in low_suspicion:
            if term in name_lower:
                score = max(score, 0.5)

        return score

    def get_recommendations(self, scan_result: Dict[str, Any]) -> List[str]:
        """Get recommendations for handling detected PII."""
        recommendations = []

        if not scan_result.get("pii_detected"):
            return recommendations

        pii_type = scan_result.get("pii_type")

        if pii_type == PIIType.SSN:
            recommendations.append("Consider encrypting SSN data at rest and in transit")
            recommendations.append("Implement strict access controls for SSN data")
            recommendations.append("Mask SSN in non-production environments")
        elif pii_type == PIIType.CREDIT_CARD:
            recommendations.append("Ensure PCI-DSS compliance for credit card data")
            recommendations.append("Tokenize credit card numbers where possible")
            recommendations.append("Encrypt credit card data")
        elif pii_type == PIIType.EMAIL:
            recommendations.append("Consider hashing email addresses for analytics")
            recommendations.append("Mask email addresses in logs and error messages")
        elif pii_type == PIIType.PHONE:
            recommendations.append("Mask phone numbers in non-production environments")
        else:
            recommendations.append("Review data handling policies for PII")
            recommendations.append("Consider masking or encrypting this data")

        return recommendations

    def detect_in_column_name(self, column_name: str) -> List[str]:
        """Detect PII indicators in column name."""
        detections = []
        name_lower = column_name.lower()

        # Check for exact matches
        for suspicious in self.suspicious_names:
            if suspicious in name_lower:
                detections.append(f"Column name contains '{suspicious}'")

        # Check for common patterns
        if "_id" in name_lower and any(
            pii in name_lower for pii in ["user", "customer", "person", "employee"]
        ):
            detections.append("Column appears to be a person identifier")

        return detections

    def detect_in_sample(
        self, column_name: str, sample: List[Any]
    ) -> List[Tuple[str, float]]:
        """Detect PII patterns in data sample."""
        detections = []

        if not sample:
            return detections

        for pii_type, pattern in self.patterns.items():
            matches = sum(
                1 for value in sample
                if value is not None and re.match(pattern, str(value))
            )
            ratio = matches / len(sample)

            if ratio > 0.5:  # More than 50% match
                detections.append((pii_type, ratio))

        return detections

    def _calculate_confidence(
        self,
        name_indicators: List[str],
        data_indicators: List[Tuple[str, float]],
    ) -> float:
        """Calculate confidence score for PII detection."""
        score = 0.0

        # Name indicators contribute to confidence
        if name_indicators:
            score += min(0.5, len(name_indicators) * 0.2)

        # Data pattern matches contribute more
        if data_indicators:
            max_ratio = max(ratio for _, ratio in data_indicators)
            score += max_ratio * 0.5

        return min(1.0, score)

    def _get_pii_types(
        self,
        column_name: str,
        name_indicators: List[str],
        data_indicators: List[Tuple[str, float]],
    ) -> List[str]:
        """Determine PII types from indicators."""
        pii_types = set()

        # From name indicators
        name_lower = column_name.lower()
        for key, pii_type in self.pii_type_map.items():
            if key in name_lower:
                pii_types.add(pii_type)

        # From data indicators
        for indicator_type, _ in data_indicators:
            if indicator_type == "email":
                pii_types.add("Email")
            elif indicator_type == "phone":
                pii_types.add("Phone")
            elif indicator_type == "ssn":
                pii_types.add("SSN")
            elif indicator_type == "credit_card":
                pii_types.add("Credit Card")
            elif indicator_type == "ip_address":
                pii_types.add("IP Address")
            elif indicator_type == "zip_code":
                pii_types.add("Address")
            elif indicator_type == "date_of_birth":
                pii_types.add("Date of Birth")

        return list(pii_types)

    def scan_schema_only(self, schema: TableMetadata) -> List[PIIDetection]:
        """Scan schema without data access."""
        detections = []

        for column in schema.columns:
            name_indicators = self.detect_in_column_name(column.name)

            if name_indicators:
                pii_types = self._get_pii_types(column.name, name_indicators, [])

                detections.append(PIIDetection(
                    table_id=schema.table_id,
                    column_name=column.name,
                    name_indicators=name_indicators,
                    data_indicators=[],
                    confidence=min(0.5, len(name_indicators) * 0.2),
                    pii_types=pii_types,
                ))

        return detections


class PIIRegistry:
    """Registry for tracking PII across data assets."""

    def __init__(self):
        self._detections: dict[str, List[PIIDetection]] = {}

    def register_detections(
        self, table_id: str, detections: List[PIIDetection]
    ) -> None:
        """Register PII detections for a table."""
        self._detections[table_id] = detections

    def get_table_pii(self, table_id: str) -> List[PIIDetection]:
        """Get PII detections for a table."""
        return self._detections.get(table_id, [])

    def get_all_pii(self) -> dict[str, List[PIIDetection]]:
        """Get all PII detections."""
        return self._detections.copy()

    def get_high_risk_tables(self, confidence_threshold: float = 0.7) -> List[str]:
        """Get tables with high-confidence PII."""
        high_risk = []

        for table_id, detections in self._detections.items():
            if any(d.confidence >= confidence_threshold for d in detections):
                high_risk.append(table_id)

        return high_risk

    def get_pii_summary(self) -> dict:
        """Get summary of PII across all tables."""
        summary = {
            "total_tables_scanned": len(self._detections),
            "tables_with_pii": 0,
            "total_pii_columns": 0,
            "pii_types": {},
        }

        for detections in self._detections.values():
            if detections:
                summary["tables_with_pii"] += 1
                summary["total_pii_columns"] += len(detections)

                for detection in detections:
                    for pii_type in detection.pii_types:
                        summary["pii_types"][pii_type] = (
                            summary["pii_types"].get(pii_type, 0) + 1
                        )

        return summary
