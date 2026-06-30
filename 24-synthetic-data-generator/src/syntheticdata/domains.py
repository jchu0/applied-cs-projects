"""Domain-specific validation and configuration."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DomainType(Enum):
    """Supported domain types."""
    GENERAL = "general"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    CUSTOM = "custom"


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""

    domain_type: DomainType
    name: str
    description: str = ""

    # Terminology and vocabulary
    required_terms: list[str] = field(default_factory=list)
    prohibited_terms: list[str] = field(default_factory=list)

    # Quality thresholds
    min_accuracy_score: float = 0.8
    min_relevance_score: float = 0.7
    max_hallucination_rate: float = 0.1

    # Format requirements
    require_citations: bool = False
    require_disclaimers: bool = False
    max_response_length: int = 2048

    # Domain-specific patterns
    required_patterns: list[str] = field(default_factory=list)
    prohibited_patterns: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict = field(default_factory=dict)


class DomainValidator(ABC):
    """Base class for domain validators."""

    def __init__(self, config: DomainConfig):
        self.config = config

    @abstractmethod
    def validate(self, example) -> tuple[bool, list[str]]:
        """Validate example against domain requirements.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        pass

    def check_terminology(self, text: str) -> list[str]:
        """Check for required and prohibited terms."""
        issues = []
        text_lower = text.lower()

        # Check required terms
        for term in self.config.required_terms:
            if term.lower() not in text_lower:
                issues.append(f"Missing required term: {term}")

        # Check prohibited terms
        for term in self.config.prohibited_terms:
            if term.lower() in text_lower:
                issues.append(f"Contains prohibited term: {term}")

        return issues

    def check_patterns(self, text: str) -> list[str]:
        """Check for required and prohibited patterns."""
        issues = []

        # Check required patterns
        for pattern in self.config.required_patterns:
            if not re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Missing required pattern: {pattern}")

        # Check prohibited patterns
        for pattern in self.config.prohibited_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Contains prohibited pattern: {pattern}")

        return issues

    def check_length(self, text: str) -> list[str]:
        """Check response length."""
        if len(text) > self.config.max_response_length:
            return [f"Response exceeds max length ({len(text)} > {self.config.max_response_length})"]
        return []


class LegalDomainValidator(DomainValidator):
    """Validator for legal domain content."""

    def __init__(self, config: DomainConfig = None):
        if config is None:
            config = self._default_config()
        super().__init__(config)

    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            domain_type=DomainType.LEGAL,
            name="Legal",
            description="Legal documents, contracts, and regulations",
            required_terms=[],
            prohibited_terms=[
                "guarantee",
                "definitely",
                "always",
                "never",
            ],
            min_accuracy_score=0.9,
            min_relevance_score=0.8,
            max_hallucination_rate=0.05,
            require_citations=True,
            require_disclaimers=True,
            max_response_length=4096,
            required_patterns=[],
            prohibited_patterns=[
                r"(?i)i am not a lawyer",  # Should not include in training data
            ],
        )

    def validate(self, example) -> tuple[bool, list[str]]:
        """Validate legal domain example."""
        issues = []

        # Get text content
        if hasattr(example, 'answer'):
            text = example.answer
        elif hasattr(example, 'output'):
            text = example.output
        else:
            text = str(example)

        # Check terminology
        issues.extend(self.check_terminology(text))

        # Check patterns
        issues.extend(self.check_patterns(text))

        # Check length
        issues.extend(self.check_length(text))

        # Legal-specific checks
        if self.config.require_disclaimers:
            disclaimer_patterns = [
                r"(?i)consult.*attorney",
                r"(?i)legal advice",
                r"(?i)not.*substitute.*professional",
                r"(?i)for informational purposes",
            ]
            has_disclaimer = any(
                re.search(p, text) for p in disclaimer_patterns
            )
            if not has_disclaimer and len(text) > 200:
                issues.append("Legal content should include appropriate disclaimers")

        # Check for citation-like patterns
        if self.config.require_citations:
            citation_patterns = [
                r"\d+\s+U\.S\.C\.",  # US Code
                r"\d+\s+C\.F\.R\.",  # CFR
                r"\d+\s+[A-Z][a-z]+\.\s+\d+",  # Case citations
                r"§\s*\d+",  # Section references
            ]
            has_citations = any(
                re.search(p, text) for p in citation_patterns
            )
            # Only warn, don't require for all content
            if not has_citations and len(text) > 500:
                pass  # Optional warning

        return len(issues) == 0, issues


class MedicalDomainValidator(DomainValidator):
    """Validator for medical domain content."""

    def __init__(self, config: DomainConfig = None):
        if config is None:
            config = self._default_config()
        super().__init__(config)

    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            domain_type=DomainType.MEDICAL,
            name="Medical",
            description="Medical and healthcare content",
            required_terms=[],
            prohibited_terms=[
                "cure",
                "guaranteed",
                "miracle",
                "100%",
            ],
            min_accuracy_score=0.95,
            min_relevance_score=0.85,
            max_hallucination_rate=0.02,
            require_citations=True,
            require_disclaimers=True,
            max_response_length=4096,
            required_patterns=[],
            prohibited_patterns=[
                r"(?i)stop taking.*medication",
                r"(?i)instead of.*doctor",
            ],
        )

    def validate(self, example) -> tuple[bool, list[str]]:
        """Validate medical domain example."""
        issues = []

        # Get text content
        if hasattr(example, 'answer'):
            text = example.answer
        elif hasattr(example, 'output'):
            text = example.output
        else:
            text = str(example)

        # Check terminology
        issues.extend(self.check_terminology(text))

        # Check patterns
        issues.extend(self.check_patterns(text))

        # Check length
        issues.extend(self.check_length(text))

        # Medical-specific checks
        if self.config.require_disclaimers:
            disclaimer_patterns = [
                r"(?i)consult.*healthcare",
                r"(?i)consult.*physician",
                r"(?i)medical advice",
                r"(?i)not.*substitute.*professional",
                r"(?i)seek medical attention",
            ]
            has_disclaimer = any(
                re.search(p, text) for p in disclaimer_patterns
            )
            if not has_disclaimer and len(text) > 200:
                issues.append("Medical content should include healthcare disclaimers")

        # Check for dangerous advice patterns
        dangerous_patterns = [
            (r"(?i)diagnose.*yourself", "Self-diagnosis advice"),
            (r"(?i)increase.*dose", "Dosage modification advice"),
            (r"(?i)mix.*medication", "Medication mixing advice"),
        ]

        for pattern, description in dangerous_patterns:
            if re.search(pattern, text):
                issues.append(f"Contains potentially dangerous advice: {description}")

        return len(issues) == 0, issues


class TechnicalDomainValidator(DomainValidator):
    """Validator for technical/engineering domain content."""

    def __init__(self, config: DomainConfig = None):
        if config is None:
            config = self._default_config()
        super().__init__(config)

    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            domain_type=DomainType.TECHNICAL,
            name="Technical",
            description="Programming, engineering, and technical documentation",
            required_terms=[],
            prohibited_terms=[],
            min_accuracy_score=0.85,
            min_relevance_score=0.8,
            max_hallucination_rate=0.1,
            require_citations=False,
            require_disclaimers=False,
            max_response_length=8192,
            required_patterns=[],
            prohibited_patterns=[
                r"(?i)hack.*into",
                r"(?i)bypass.*security",
            ],
        )

    def validate(self, example) -> tuple[bool, list[str]]:
        """Validate technical domain example."""
        issues = []

        # Get text content
        if hasattr(example, 'answer'):
            text = example.answer
        elif hasattr(example, 'output'):
            text = example.output
        else:
            text = str(example)

        # Check terminology
        issues.extend(self.check_terminology(text))

        # Check patterns
        issues.extend(self.check_patterns(text))

        # Check length
        issues.extend(self.check_length(text))

        # Technical-specific checks
        # Check for code blocks if discussing code
        code_indicators = [
            "function", "class", "def ", "import ", "const ", "var ", "let "
        ]
        has_code_topic = any(ind in text.lower() for ind in code_indicators)
        has_code_block = "```" in text or "    " in text  # Markdown or indented

        if has_code_topic and not has_code_block and len(text) > 300:
            # This is just informational, not an error
            pass

        # Check for deprecated/unsafe patterns
        unsafe_patterns = [
            (r"eval\s*\(", "Use of eval()"),
            (r"(?i)disable.*ssl", "Disabling SSL"),
            (r"password\s*=\s*['\"]", "Hardcoded password"),
        ]

        for pattern, description in unsafe_patterns:
            if re.search(pattern, text):
                issues.append(f"Contains potentially unsafe pattern: {description}")

        return len(issues) == 0, issues


class FinancialDomainValidator(DomainValidator):
    """Validator for financial domain content."""

    def __init__(self, config: DomainConfig = None):
        if config is None:
            config = self._default_config()
        super().__init__(config)

    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            domain_type=DomainType.FINANCIAL,
            name="Financial",
            description="Financial advice, investing, and market analysis",
            required_terms=[],
            prohibited_terms=[
                "guaranteed returns",
                "risk-free",
                "can't lose",
                "100% safe",
            ],
            min_accuracy_score=0.9,
            min_relevance_score=0.8,
            max_hallucination_rate=0.05,
            require_citations=True,
            require_disclaimers=True,
            max_response_length=4096,
            required_patterns=[],
            prohibited_patterns=[
                r"(?i)insider.*information",
                r"(?i)pump.*dump",
            ],
        )

    def validate(self, example) -> tuple[bool, list[str]]:
        """Validate financial domain example."""
        issues = []

        # Get text content
        if hasattr(example, 'answer'):
            text = example.answer
        elif hasattr(example, 'output'):
            text = example.output
        else:
            text = str(example)

        # Check terminology
        issues.extend(self.check_terminology(text))

        # Check patterns
        issues.extend(self.check_patterns(text))

        # Check length
        issues.extend(self.check_length(text))

        # Financial-specific checks
        if self.config.require_disclaimers:
            disclaimer_patterns = [
                r"(?i)not.*financial advice",
                r"(?i)consult.*financial advisor",
                r"(?i)past performance",
                r"(?i)for informational purposes",
                r"(?i)do your own research",
            ]
            has_disclaimer = any(
                re.search(p, text) for p in disclaimer_patterns
            )
            if not has_disclaimer and len(text) > 200:
                issues.append("Financial content should include appropriate disclaimers")

        # Check for specific financial claims
        claim_patterns = [
            (r"(?i)will\s+(definitely|certainly)\s+rise", "Definitive market predictions"),
            (r"(?i)\d+%.*return", "Specific return promises"),
            (r"(?i)double your money", "Unrealistic return claims"),
        ]

        for pattern, description in claim_patterns:
            if re.search(pattern, text):
                issues.append(f"Contains problematic financial claim: {description}")

        return len(issues) == 0, issues


class CustomDomainValidator(DomainValidator):
    """Validator for custom user-defined domains."""

    def __init__(self, config: DomainConfig):
        super().__init__(config)

    def validate(self, example) -> tuple[bool, list[str]]:
        """Validate against custom domain configuration."""
        issues = []

        # Get text content
        if hasattr(example, 'answer'):
            text = example.answer
        elif hasattr(example, 'output'):
            text = example.output
        else:
            text = str(example)

        # Run all standard checks
        issues.extend(self.check_terminology(text))
        issues.extend(self.check_patterns(text))
        issues.extend(self.check_length(text))

        return len(issues) == 0, issues


class DomainRegistry:
    """Registry for domain validators."""

    def __init__(self):
        self._validators: dict[str, type[DomainValidator]] = {
            DomainType.LEGAL.value: LegalDomainValidator,
            DomainType.MEDICAL.value: MedicalDomainValidator,
            DomainType.TECHNICAL.value: TechnicalDomainValidator,
            DomainType.FINANCIAL.value: FinancialDomainValidator,
        }
        self._custom_configs: dict[str, DomainConfig] = {}

    def register_custom_domain(self, name: str, config: DomainConfig):
        """Register a custom domain configuration."""
        self._custom_configs[name] = config

    def register(self, name: str):
        """Decorator to register a custom domain validator class."""
        def decorator(validator_cls):
            self._validators[name] = validator_cls
            return validator_cls
        return decorator

    def get_validator(self, domain: str, config: DomainConfig = None) -> DomainValidator:
        """Get validator for a domain."""
        if domain in self._validators:
            return self._validators[domain](config)
        elif domain in self._custom_configs:
            return CustomDomainValidator(self._custom_configs[domain])
        elif config:
            return CustomDomainValidator(config)
        else:
            # Return a permissive validator for unknown domains
            return CustomDomainValidator(DomainConfig(
                domain_type=DomainType.GENERAL,
                name="General",
                description="General purpose content",
            ))

    def list_domains(self) -> list[str]:
        """List all registered domains."""
        return list(self._validators.keys()) + list(self._custom_configs.keys())


# Singleton registry
domain_registry = DomainRegistry()


def validate_for_domain(
    example,
    domain: str,
    config: DomainConfig = None,
) -> tuple[bool, list[str]]:
    """Validate an example for a specific domain.

    Args:
        example: The example to validate
        domain: Domain name (legal, medical, technical, financial, or custom)
        config: Optional custom domain configuration

    Returns:
        Tuple of (is_valid, list of issues)
    """
    validator = domain_registry.get_validator(domain, config)
    return validator.validate(example)


def get_domain_config(domain: str) -> DomainConfig:
    """Get default configuration for a domain."""
    validators = {
        "legal": LegalDomainValidator,
        "medical": MedicalDomainValidator,
        "technical": TechnicalDomainValidator,
        "financial": FinancialDomainValidator,
    }

    if domain in validators:
        validator = validators[domain]()
        return validator.config
    else:
        return DomainConfig(
            domain_type=DomainType.GENERAL,
            name="General",
            description="General purpose content",
        )
