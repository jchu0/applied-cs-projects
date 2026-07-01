# Project 24: Synthetic Data Generator - Setup Guide

## Overview
LLM-based synthetic data generation system for creating high-quality training datasets.

## Prerequisites
- Python 3.9+
- pip or conda
- OpenAI API key or Anthropic API key (for LLM generation)
- 8GB+ RAM recommended

## Installation

### 1. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n syntheticdata python=3.10
conda activate syntheticdata
```

### 2. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### 3. Download NLP Models (Optional)
```bash
# Download spaCy model if needed
python -m spacy download en_core_web_sm
```

## Configuration

### 1. Environment Variables
Create a `.env` file in the project root:

```bash
# LLM Provider Configuration
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Generation Settings
DEFAULT_TEMPERATURE=0.8
DEFAULT_MAX_TOKENS=1000

# Output Directory
OUTPUT_DIR=./output

# DVC Remote (optional)
DVC_REMOTE=s3://your-bucket/datasets
```

### 2. Provider Setup
Configure your LLM provider in code:

```python
from syntheticdata import ModelProvider

# OpenAI
provider = ModelProvider(
    provider_type="openai",
    api_key="your-key",
    model="gpt-4"
)

# Anthropic
provider = ModelProvider(
    provider_type="anthropic",
    api_key="your-key",
    model="claude-3-opus-20240229"
)
```

## Usage

### Basic Generation
```python
from syntheticdata import SyntheticDataGenerator, GenerationConfig
from syntheticdata import DataType, DifficultyLevel

# Create configuration
config = GenerationConfig(
    data_type=DataType.RAG_QA,
    num_samples=100,
    temperature=0.8,
    difficulty_distribution={
        DifficultyLevel.EASY: 0.3,
        DifficultyLevel.MEDIUM: 0.4,
        DifficultyLevel.HARD: 0.3
    }
)

# Initialize generator
generator = SyntheticDataGenerator(
    model_provider=provider,
    config=config
)

# Generate data
examples = await generator.generate_batch(num_samples=10)
```

### Running the API Server
```bash
# Start the FastAPI server
uvicorn syntheticdata.api:app --host 0.0.0.0 --port 8000 --reload

# Or use the CLI
python -m syntheticdata.api
```

### API Examples
```bash
# Generate synthetic data
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "rag_qa",
    "num_samples": 10,
    "domain": "medical",
    "temperature": 0.8
  }'

# Check job status
curl "http://localhost:8000/jobs/{job_id}/status"

# Get results
curl "http://localhost:8000/jobs/{job_id}/result"
```

## Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=syntheticdata --cov-report=html

# Run specific test file
pytest tests/test_generator.py
```

## Data Versioning with DVC (Optional)
```bash
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d myremote s3://mybucket/datasets

# Track datasets
dvc add output/dataset.jsonl

# Commit and push
git add output/dataset.jsonl.dvc .dvc/config
git commit -m "Add dataset"
dvc push
```

## Common Issues

### Issue: API key not found
**Solution**: Ensure `.env` file is in the project root and contains valid API keys.

### Issue: Out of memory during generation
**Solution**: Reduce batch size or use smaller models:
```python
config = GenerationConfig(batch_size=5)  # Reduce from default
```

### Issue: Low quality outputs
**Solution**: Adjust quality threshold and temperature:
```python
config = GenerationConfig(
    min_quality_score=0.8,
    temperature=0.7  # Lower for more deterministic
)
```

## Project Structure
```
24-synthetic-data-generator/
├── src/syntheticdata/
│   ├── generator.py       # Main generation engine
│   ├── quality.py         # Quality scoring
│   ├── domains.py         # Domain-specific configs
│   ├── api.py            # FastAPI application
│   └── ...
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── SETUP.md
```

## Next Steps
1. Configure your LLM provider
2. Define domain-specific templates
3. Run quality evaluation
4. Export data for training
5. Set up continuous generation pipeline

## Resources
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [DVC Documentation](https://dvc.org/doc)
