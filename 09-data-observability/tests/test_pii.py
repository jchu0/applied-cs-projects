"""Tests for PII detection module."""

import pytest
from observability.pii import PIIDetector, PIIType


class TestPIIDetector:
    """Tests for PIIDetector class."""

    @pytest.fixture
    def detector(self):
        """Create PII detector."""
        return PIIDetector()

    def test_detect_email(self, detector):
        """Test email detection."""
        result = detector.detect("user@example.com")
        assert PIIType.EMAIL in result
        assert result[PIIType.EMAIL]["confidence"] > 0.9

    def test_detect_phone_us(self, detector):
        """Test US phone number detection."""
        result = detector.detect("(555) 123-4567")
        assert PIIType.PHONE in result

    def test_detect_phone_international(self, detector):
        """Test international phone detection."""
        result = detector.detect("+1-555-123-4567")
        assert PIIType.PHONE in result

    def test_detect_ssn(self, detector):
        """Test SSN detection."""
        result = detector.detect("123-45-6789")
        assert PIIType.SSN in result
        assert result[PIIType.SSN]["confidence"] > 0.9

    def test_detect_credit_card(self, detector):
        """Test credit card detection."""
        result = detector.detect("4111-1111-1111-1111")
        assert PIIType.CREDIT_CARD in result

    def test_detect_credit_card_no_dashes(self, detector):
        """Test credit card without dashes."""
        result = detector.detect("4111111111111111")
        assert PIIType.CREDIT_CARD in result

    def test_detect_ip_address(self, detector):
        """Test IP address detection."""
        result = detector.detect("192.168.1.1")
        assert PIIType.IP_ADDRESS in result

    def test_detect_no_pii(self, detector):
        """Test non-PII text."""
        result = detector.detect("Hello, this is a normal message")
        assert len(result) == 0

    def test_detect_multiple_pii(self, detector):
        """Test detecting multiple PII types."""
        text = "Contact: user@example.com, Phone: 555-123-4567"
        result = detector.detect(text)
        assert PIIType.EMAIL in result
        assert PIIType.PHONE in result

    def test_scan_column(self, detector):
        """Test scanning a column of values."""
        values = [
            "john@example.com",
            "jane@example.org",
            "normal text",
            "bob@test.com",
            "no email here"
        ]
        result = detector.scan_column("email", values)
        assert result["pii_detected"] is True
        assert result["pii_type"] == PIIType.EMAIL
        assert result["pii_percentage"] == 60.0  # 3 out of 5

    def test_scan_column_no_pii(self, detector):
        """Test scanning column with no PII."""
        values = ["apple", "banana", "cherry", "date", "elderberry"]
        result = detector.scan_column("fruits", values)
        assert result["pii_detected"] is False
        assert result["pii_percentage"] == 0.0

    def test_scan_table(self, detector):
        """Test scanning entire table."""
        columns = {
            "user_email": ["john@example.com", "jane@example.org"],
            "user_name": ["John Doe", "Jane Doe"],
            "phone": ["555-1234", "555-5678"],
        }
        result = detector.scan_table(columns)
        assert "user_email" in result
        assert result["user_email"]["pii_detected"] is True

    def test_mask_pii(self, detector):
        """Test masking PII in text."""
        text = "Email: user@example.com, SSN: 123-45-6789"
        masked = detector.mask(text)
        assert "user@example.com" not in masked
        assert "123-45-6789" not in masked
        assert "[EMAIL]" in masked or "***" in masked

    def test_confidence_levels(self, detector):
        """Test confidence levels for detection."""
        # High confidence
        result = detector.detect("user@example.com")
        assert result[PIIType.EMAIL]["confidence"] >= 0.9

        # Medium confidence - ambiguous pattern
        result = detector.detect("123456789")  # Could be anything
        # Should have lower confidence or not detect

    def test_custom_patterns(self, detector):
        """Test adding custom PII patterns."""
        detector.add_pattern(
            "EMPLOYEE_ID",
            r"EMP-\d{6}",
            confidence=0.95
        )
        result = detector.detect("Employee: EMP-123456")
        assert "EMPLOYEE_ID" in result

    def test_detect_by_column_name(self, detector):
        """Test detection based on column name heuristics."""
        # Columns with PII-like names should have higher suspicion
        suspicious_names = ["email", "ssn", "social_security", "credit_card", "phone_number"]
        for name in suspicious_names:
            suspicion = detector.column_name_suspicion(name)
            assert suspicion > 0.5

    def test_non_suspicious_column_names(self, detector):
        """Test non-suspicious column names."""
        safe_names = ["created_at", "row_count", "product_id", "category"]
        for name in safe_names:
            suspicion = detector.column_name_suspicion(name)
            assert suspicion < 0.5


class TestPIIType:
    """Tests for PIIType enum."""

    def test_all_pii_types(self):
        """Test all PII types are defined."""
        assert PIIType.EMAIL is not None
        assert PIIType.PHONE is not None
        assert PIIType.SSN is not None
        assert PIIType.CREDIT_CARD is not None
        assert PIIType.IP_ADDRESS is not None

    def test_pii_type_values(self):
        """Test PII type string values."""
        assert PIIType.EMAIL.value == "email"
        assert PIIType.SSN.value == "ssn"


class TestPIIScanResults:
    """Tests for PII scan result handling."""

    @pytest.fixture
    def detector(self):
        """Create PII detector."""
        return PIIDetector()

    def test_scan_result_format(self, detector):
        """Test scan result format."""
        result = detector.scan_column(
            "test",
            ["user@example.com", "normal"]
        )
        assert "pii_detected" in result
        assert "pii_type" in result
        assert "pii_percentage" in result
        assert "sample_matches" in result

    def test_recommendations(self, detector):
        """Test PII handling recommendations."""
        result = detector.scan_column(
            "ssn",
            ["123-45-6789", "987-65-4321"]
        )
        recommendations = detector.get_recommendations(result)
        assert len(recommendations) > 0
        assert any("mask" in r.lower() or "encrypt" in r.lower() for r in recommendations)
