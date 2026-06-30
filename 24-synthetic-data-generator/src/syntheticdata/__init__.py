"""
Synthetic Data Generator for RAG + Fine-Tuning

A production-grade synthetic data generation pipeline for creating
high-quality training data for RAG systems and LLM fine-tuning.
"""

from .schemas import (
    DataType,
    DifficultyLevel,
    RAGExample,
    InstructionExample,
    ConversationExample,
    PreferenceExample,
    GenerationConfig,
)
from .generator import SyntheticDataGenerator
from .quality import QualityScorer
from .curriculum import CurriculumSampler, DifficultyScorer, CurriculumManager
from .dataset import DatasetManager, Dataset
from .provider import MockProvider, MockModelProvider

# Aliases for backwards compatibility
GeneratorConfig = GenerationConfig
DatasetBuilder = DatasetManager
QualityChecker = QualityScorer
from .templates import PromptTemplateLibrary, DomainPromptTemplates
from .domains import (
    DomainType,
    DomainConfig,
    DomainRegistry,
    LegalDomainValidator,
    MedicalDomainValidator,
    TechnicalDomainValidator,
    FinancialDomainValidator,
    CustomDomainValidator,
    validate_for_domain,
    get_domain_config,
)

# Optional API imports
try:
    from .api import create_api, DVCIntegration
except ImportError:
    # FastAPI not installed
    create_api = None
    DVCIntegration = None

__version__ = "0.1.0"
__all__ = [
    # Schemas
    "DataType",
    "DifficultyLevel",
    "RAGExample",
    "InstructionExample",
    "ConversationExample",
    "PreferenceExample",
    "GenerationConfig",
    "GeneratorConfig",  # Alias
    # Core
    "SyntheticDataGenerator",
    "QualityScorer",
    "QualityChecker",  # Alias
    "CurriculumSampler",
    "CurriculumManager",
    "DifficultyScorer",
    "DatasetManager",
    "DatasetBuilder",  # Alias
    "Dataset",
    "MockProvider",
    "MockModelProvider",
    "PromptTemplateLibrary",
    "DomainPromptTemplates",
    # Domains
    "DomainType",
    "DomainConfig",
    "DomainRegistry",
    "LegalDomainValidator",
    "MedicalDomainValidator",
    "TechnicalDomainValidator",
    "FinancialDomainValidator",
    "CustomDomainValidator",
    "validate_for_domain",
    "get_domain_config",
    # API (optional)
    "create_api",
    "DVCIntegration",
]
