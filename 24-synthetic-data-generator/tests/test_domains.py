"""Tests for domain validation."""

import pytest

from syntheticdata.domains import (
    DomainType,
    DomainConfig,
    LegalDomainValidator,
    MedicalDomainValidator,
    TechnicalDomainValidator,
    FinancialDomainValidator,
    CustomDomainValidator,
    DomainRegistry,
    validate_for_domain,
    get_domain_config,
)
from syntheticdata.schemas import RAGExample, InstructionExample, DifficultyLevel


@pytest.fixture
def legal_validator():
    """Create legal domain validator."""
    return LegalDomainValidator()


@pytest.fixture
def medical_validator():
    """Create medical domain validator."""
    return MedicalDomainValidator()


@pytest.fixture
def technical_validator():
    """Create technical domain validator."""
    return TechnicalDomainValidator()


@pytest.fixture
def financial_validator():
    """Create financial domain validator."""
    return FinancialDomainValidator()


class TestLegalDomainValidator:
    """Tests for LegalDomainValidator."""

    def test_valid_legal_content(self, legal_validator):
        """Test validation of proper legal content."""
        example = RAGExample(
            id="test1",
            question="What is contract law?",
            answer="Contract law governs agreements between parties. For specific legal advice, please consult an attorney. See 17 U.S.C. § 101 for definitions.",
            context="Legal textbook on contracts",
            domain="legal",
        )

        is_valid, issues = legal_validator.validate(example)
        assert is_valid
        assert len(issues) == 0

    def test_prohibited_terms(self, legal_validator):
        """Test detection of prohibited legal terms."""
        example = RAGExample(
            id="test2",
            question="Will I win my case?",
            answer="You will definitely win this case. It's guaranteed to succeed.",
            context="Legal advice",
            domain="legal",
        )

        is_valid, issues = legal_validator.validate(example)
        assert not is_valid
        assert any("definitely" in issue.lower() for issue in issues)

    def test_missing_disclaimer(self, legal_validator):
        """Test detection of missing disclaimers."""
        example = RAGExample(
            id="test3",
            question="How do I file a lawsuit?",
            answer="To file a lawsuit, you need to first identify the correct court that has jurisdiction over your case. Then prepare your complaint with all relevant facts and legal claims. " * 5,
            context="Legal procedure guide",
            domain="legal",
        )

        is_valid, issues = legal_validator.validate(example)
        assert not is_valid
        assert any("disclaimer" in issue.lower() for issue in issues)


class TestMedicalDomainValidator:
    """Tests for MedicalDomainValidator."""

    def test_valid_medical_content(self, medical_validator):
        """Test validation of proper medical content."""
        example = RAGExample(
            id="test1",
            question="What are symptoms of diabetes?",
            answer="Common symptoms include increased thirst, frequent urination, and fatigue. Please consult your healthcare provider for proper diagnosis.",
            context="Medical reference",
            domain="medical",
        )

        is_valid, issues = medical_validator.validate(example)
        assert is_valid
        assert len(issues) == 0

    def test_prohibited_medical_terms(self, medical_validator):
        """Test detection of prohibited medical terms."""
        example = RAGExample(
            id="test2",
            question="Does this product work?",
            answer="This is a miracle cure that is guaranteed to work 100% of the time.",
            context="Product claim",
            domain="medical",
        )

        is_valid, issues = medical_validator.validate(example)
        assert not is_valid
        assert len(issues) >= 2  # Multiple prohibited terms

    def test_dangerous_advice_detection(self, medical_validator):
        """Test detection of dangerous medical advice."""
        example = RAGExample(
            id="test3",
            question="What should I do about my medication?",
            answer="You should increase your dose if you're not feeling better. You can also mix your medications for faster results.",
            context="Medical advice",
            domain="medical",
        )

        is_valid, issues = medical_validator.validate(example)
        assert not is_valid
        assert any("dangerous" in issue.lower() for issue in issues)


class TestTechnicalDomainValidator:
    """Tests for TechnicalDomainValidator."""

    def test_valid_technical_content(self, technical_validator):
        """Test validation of proper technical content."""
        example = InstructionExample(
            id="test1",
            instruction="Write a function to calculate factorial",
            output="```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```",
            domain="technical",
        )

        is_valid, issues = technical_validator.validate(example)
        assert is_valid
        assert len(issues) == 0

    def test_unsafe_code_patterns(self, technical_validator):
        """Test detection of unsafe code patterns."""
        example = InstructionExample(
            id="test2",
            instruction="Execute user input",
            output="You can use eval(user_input) to execute user code. Set password = 'admin123' for testing.",
            domain="technical",
        )

        is_valid, issues = technical_validator.validate(example)
        assert not is_valid
        assert any("unsafe" in issue.lower() for issue in issues)


class TestFinancialDomainValidator:
    """Tests for FinancialDomainValidator."""

    def test_valid_financial_content(self, financial_validator):
        """Test validation of proper financial content."""
        example = RAGExample(
            id="test1",
            question="How should I invest?",
            answer="Diversification is important for managing risk. This is not financial advice - please consult a financial advisor for personalized guidance.",
            context="Investment guide",
            domain="financial",
        )

        is_valid, issues = financial_validator.validate(example)
        assert is_valid
        assert len(issues) == 0

    def test_prohibited_financial_terms(self, financial_validator):
        """Test detection of prohibited financial terms."""
        example = RAGExample(
            id="test2",
            question="Is this investment safe?",
            answer="This is guaranteed returns with risk-free investing. You can't lose with this strategy.",
            context="Investment pitch",
            domain="financial",
        )

        is_valid, issues = financial_validator.validate(example)
        assert not is_valid
        # Should catch multiple prohibited terms

    def test_problematic_claims(self, financial_validator):
        """Test detection of problematic financial claims."""
        example = RAGExample(
            id="test3",
            question="What returns can I expect?",
            answer="This stock will definitely rise 500%. You can double your money in a week.",
            context="Stock tip",
            domain="financial",
        )

        is_valid, issues = financial_validator.validate(example)
        assert not is_valid
        assert any("claim" in issue.lower() for issue in issues)


class TestCustomDomainValidator:
    """Tests for CustomDomainValidator."""

    def test_custom_domain_validation(self):
        """Test custom domain with user-defined rules."""
        config = DomainConfig(
            domain_type=DomainType.CUSTOM,
            name="Gaming",
            description="Video game content",
            required_terms=["gameplay", "player"],
            prohibited_terms=["cheat", "hack"],
            max_response_length=1000,
        )

        validator = CustomDomainValidator(config)

        # Valid content
        valid_example = RAGExample(
            id="test1",
            question="How do I play?",
            answer="The player uses gameplay mechanics to progress through levels.",
            context="Game guide",
            domain="gaming",
        )
        is_valid, issues = validator.validate(valid_example)
        assert is_valid

        # Invalid content
        invalid_example = RAGExample(
            id="test2",
            question="How do I win?",
            answer="Use this cheat code to hack the game.",
            context="Game guide",
            domain="gaming",
        )
        is_valid, issues = validator.validate(invalid_example)
        assert not is_valid


class TestDomainRegistry:
    """Tests for DomainRegistry."""

    def test_get_builtin_validators(self):
        """Test getting built-in validators."""
        registry = DomainRegistry()

        for domain in ["legal", "medical", "technical", "financial"]:
            validator = registry.get_validator(domain)
            assert validator is not None

    def test_register_custom_domain(self):
        """Test registering custom domain."""
        registry = DomainRegistry()

        config = DomainConfig(
            domain_type=DomainType.CUSTOM,
            name="Education",
            description="Educational content",
        )
        registry.register_custom_domain("education", config)

        validator = registry.get_validator("education")
        assert validator is not None

    def test_list_domains(self):
        """Test listing all domains."""
        registry = DomainRegistry()
        domains = registry.list_domains()

        assert "legal" in domains
        assert "medical" in domains
        assert "technical" in domains
        assert "financial" in domains


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_validate_for_domain(self):
        """Test validate_for_domain function."""
        example = RAGExample(
            id="test1",
            question="Test question",
            answer="Test answer with medical advice. Please consult your healthcare provider.",
            context="Test",
            domain="medical",
        )

        is_valid, issues = validate_for_domain(example, "medical")
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_get_domain_config(self):
        """Test get_domain_config function."""
        for domain in ["legal", "medical", "technical", "financial"]:
            config = get_domain_config(domain)
            assert config is not None
            assert config.name

        # Unknown domain returns general config
        config = get_domain_config("unknown")
        assert config.domain_type == DomainType.GENERAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
