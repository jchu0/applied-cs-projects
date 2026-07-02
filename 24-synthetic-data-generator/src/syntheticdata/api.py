"""FastAPI production API for synthetic data generation."""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .schemas import (
    DataType,
    DifficultyLevel,
    RAGExample,
    InstructionExample,
    GenerationConfig,
)
from .generator import SyntheticDataGenerator
from .quality import QualityScorer, AutoCurationPipeline, HallucinationDetector
from .dataset import DatasetManager
from .domains import validate_for_domain, get_domain_config


# Pydantic models for API
class GenerationRequest(BaseModel):
    """Request for synthetic data generation."""

    data_type: str = Field(default="rag_qa", description="Type of data to generate")
    num_samples: int = Field(default=10, ge=1, le=1000, description="Number of samples")
    domain: Optional[str] = Field(default=None, description="Domain (legal, medical, technical, financial)")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Generation temperature")
    min_quality_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum quality score")
    difficulty_distribution: Optional[dict[str, float]] = Field(
        default=None,
        description="Distribution of difficulty levels",
    )
    source_data: list[dict] = Field(default=[], description="Source data for generation")


class GenerationResponse(BaseModel):
    """Response from generation job."""

    job_id: str
    status: str
    message: str
    created_at: str


class JobStatusResponse(BaseModel):
    """Status of a generation job."""

    job_id: str
    status: str
    progress: float
    num_generated: int
    num_total: int
    errors: list[str]
    created_at: str
    completed_at: Optional[str]


class JobResultResponse(BaseModel):
    """Result of a completed generation job."""

    job_id: str
    status: str
    examples: list[dict]
    statistics: dict
    quality_report: dict


class QualityScoreRequest(BaseModel):
    """Request for quality scoring."""

    examples: list[dict]
    domain: Optional[str] = None


class QualityScoreResponse(BaseModel):
    """Response from quality scoring."""

    scores: list[float]
    average_score: float
    passed: int
    failed: int


class CurationRequest(BaseModel):
    """Request for auto-curation."""

    examples: list[dict]
    min_quality_score: float = 0.7
    max_hallucination_score: float = 0.3
    domain: Optional[str] = None


class CurationResponse(BaseModel):
    """Response from auto-curation."""

    accepted: list[dict]
    rejected: list[dict]
    acceptance_rate: float
    rejection_reasons: dict


class DatasetSaveRequest(BaseModel):
    """Request to save dataset."""

    examples: list[dict]
    name: str
    format: str = "jsonl"
    use_dvc: bool = False


class DatasetSaveResponse(BaseModel):
    """Response from dataset save."""

    filepath: str
    num_examples: int
    format: str
    dvc_tracked: bool


class ExportRequest(BaseModel):
    """Request to export dataset for training."""

    examples: list[dict]
    format: str = "sharegpt"
    filename: Optional[str] = None


class ExportResponse(BaseModel):
    """Response from dataset export."""

    filepath: str
    format: str
    num_examples: int


class DomainValidationRequest(BaseModel):
    """Request for domain validation."""

    examples: list[dict]
    domain: str


class DomainValidationResponse(BaseModel):
    """Response from domain validation."""

    results: list[dict]
    passed: int
    failed: int
    pass_rate: float


# Job storage (in production, use Redis or database)
jobs: dict[str, dict] = {}


def create_api(
    model_provider=None,
    output_dir: str = "./output",
):
    """Create FastAPI application.

    Args:
        model_provider: Model provider for generation (required for actual generation)
        output_dir: Directory for saving datasets

    Returns:
        FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    except ImportError:
        raise ImportError("FastAPI required. Install with: pip install fastapi uvicorn")

    from .security import build_auth_dependency, install_hardening

    app = FastAPI(
        title="Synthetic Data Generator API",
        description="Production API for generating high-quality synthetic training data",
        version="0.1.0",
        dependencies=[Depends(build_auth_dependency())],
    )

    # Rate limiting + request timeout middleware (opt-in via env).
    install_hardening(app)

    # Initialize components
    dataset_manager = DatasetManager(output_dir)

    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "name": "Synthetic Data Generator API",
            "version": "0.1.0",
            "status": "running",
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.post("/generate", response_model=GenerationResponse)
    async def generate_data(
        request: GenerationRequest,
        background_tasks: BackgroundTasks,
    ):
        """Start a synthetic data generation job."""
        if model_provider is None:
            raise HTTPException(
                status_code=503,
                detail="Model provider not configured",
            )

        job_id = str(uuid.uuid4())

        # Create job record
        jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "num_generated": 0,
            "num_total": request.num_samples,
            "errors": [],
            "examples": [],
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }

        # Start background task
        background_tasks.add_task(
            run_generation_job,
            job_id,
            request,
            model_provider,
        )

        return GenerationResponse(
            job_id=job_id,
            status="pending",
            message=f"Generation job started for {request.num_samples} samples",
            created_at=jobs[job_id]["created_at"],
        )

    @app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
    async def get_job_status(job_id: str):
        """Get status of a generation job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            num_generated=job["num_generated"],
            num_total=job["num_total"],
            errors=job["errors"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
        )

    @app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
    async def get_job_result(job_id: str):
        """Get result of a completed generation job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        if job["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job not completed. Status: {job['status']}",
            )

        return JobResultResponse(
            job_id=job_id,
            status=job["status"],
            examples=job["examples"],
            statistics=job.get("statistics", {}),
            quality_report=job.get("quality_report", {}),
        )

    @app.post("/score", response_model=QualityScoreResponse)
    async def score_quality(request: QualityScoreRequest):
        """Score quality of examples."""
        if model_provider is None:
            raise HTTPException(
                status_code=503,
                detail="Model provider not configured",
            )

        scorer = QualityScorer(model_provider)
        scores = []

        for ex_dict in request.examples:
            # Convert to example object
            example = dict_to_example(ex_dict)
            if example:
                score = await scorer.score(example)
                scores.append(score)
            else:
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        threshold = 0.7
        passed = sum(1 for s in scores if s >= threshold)

        return QualityScoreResponse(
            scores=scores,
            average_score=avg_score,
            passed=passed,
            failed=len(scores) - passed,
        )

    @app.post("/curate", response_model=CurationResponse)
    async def curate_data(request: CurationRequest):
        """Auto-curate examples."""
        if model_provider is None:
            raise HTTPException(
                status_code=503,
                detail="Model provider not configured",
            )

        scorer = QualityScorer(model_provider)
        hallucination_detector = HallucinationDetector(model_provider)
        pipeline = AutoCurationPipeline(
            quality_scorer=scorer,
            hallucination_detector=hallucination_detector,
            min_quality_score=request.min_quality_score,
            max_hallucination_score=request.max_hallucination_score,
        )

        # Convert dicts to examples
        examples = [dict_to_example(ex) for ex in request.examples]
        examples = [ex for ex in examples if ex is not None]

        # Run curation
        accepted, rejected = await pipeline.curate(examples)

        # Convert back to dicts
        accepted_dicts = [example_to_dict(ex) for ex in accepted]
        rejected_dicts = [example_to_dict(ex) for ex in rejected]

        # Get rejection reasons from statistics
        stats = pipeline.get_statistics()

        return CurationResponse(
            accepted=accepted_dicts,
            rejected=rejected_dicts,
            acceptance_rate=len(accepted) / len(examples) if examples else 0.0,
            rejection_reasons=stats.get("rejection_reasons", {}),
        )

    @app.post("/validate", response_model=DomainValidationResponse)
    async def validate_domain(request: DomainValidationRequest):
        """Validate examples against domain requirements."""
        results = []
        passed = 0
        failed = 0

        for ex_dict in request.examples:
            example = dict_to_example(ex_dict)
            if example:
                is_valid, issues = validate_for_domain(example, request.domain)
                results.append({
                    "id": ex_dict.get("id", "unknown"),
                    "valid": is_valid,
                    "issues": issues,
                })
                if is_valid:
                    passed += 1
                else:
                    failed += 1
            else:
                results.append({
                    "id": ex_dict.get("id", "unknown"),
                    "valid": False,
                    "issues": ["Could not parse example"],
                })
                failed += 1

        return DomainValidationResponse(
            results=results,
            passed=passed,
            failed=failed,
            pass_rate=passed / len(request.examples) if request.examples else 0.0,
        )

    @app.post("/dataset/save", response_model=DatasetSaveResponse)
    async def save_dataset(request: DatasetSaveRequest):
        """Save dataset to file."""
        # Convert dicts to examples if needed
        examples = request.examples

        filepath = dataset_manager.save_dataset(
            examples=examples,
            name=request.name,
            format=request.format,
        )

        return DatasetSaveResponse(
            filepath=str(filepath),
            num_examples=len(examples),
            format=request.format,
            dvc_tracked=request.use_dvc,
        )

    @app.post("/dataset/export", response_model=ExportResponse)
    async def export_dataset(request: ExportRequest):
        """Export dataset for training."""
        # Convert dicts to examples
        examples = [dict_to_example(ex) for ex in request.examples]
        examples = [ex for ex in examples if ex is not None]

        output_path = None
        if request.filename:
            output_path = Path(output_dir) / request.filename

        filepath = dataset_manager.export_for_training(
            examples=examples,
            format=request.format,
            output_path=output_path,
        )

        return ExportResponse(
            filepath=str(filepath),
            format=request.format,
            num_examples=len(examples),
        )

    @app.get("/datasets")
    async def list_datasets():
        """List all saved datasets."""
        return dataset_manager.list_datasets()

    @app.get("/domains")
    async def list_domains():
        """List available domains and their configurations."""
        domains = ["legal", "medical", "technical", "financial"]
        configs = {}

        for domain in domains:
            config = get_domain_config(domain)
            configs[domain] = {
                "name": config.name,
                "description": config.description,
                "min_accuracy_score": config.min_accuracy_score,
                "max_hallucination_rate": config.max_hallucination_rate,
                "require_citations": config.require_citations,
                "require_disclaimers": config.require_disclaimers,
            }

        return configs

    return app


async def run_generation_job(
    job_id: str,
    request: GenerationRequest,
    model_provider,
):
    """Run generation job in background."""
    try:
        jobs[job_id]["status"] = "running"

        # Create configuration
        difficulty_dist = None
        if request.difficulty_distribution:
            difficulty_dist = {
                DifficultyLevel[k.upper()]: v
                for k, v in request.difficulty_distribution.items()
            }

        config = GenerationConfig(
            data_type=DataType(request.data_type),
            num_samples=request.num_samples,
            domain=request.domain,
            temperature=request.temperature,
            min_quality_score=request.min_quality_score,
            difficulty_distribution=difficulty_dist or {
                DifficultyLevel.EASY: 0.2,
                DifficultyLevel.MEDIUM: 0.4,
                DifficultyLevel.HARD: 0.3,
                DifficultyLevel.EXPERT: 0.1,
            },
        )

        # Create generator
        generator = SyntheticDataGenerator(
            model_provider=model_provider,
            config=config,
        )

        # Generate in batches
        batch_size = min(10, request.num_samples)
        all_examples = []

        for i in range(0, request.num_samples, batch_size):
            current_batch_size = min(batch_size, request.num_samples - i)

            try:
                examples = await generator.generate_batch(
                    num_samples=current_batch_size,
                    source_data=request.source_data if request.source_data else None,
                )
                all_examples.extend(examples)

                # Update progress
                jobs[job_id]["num_generated"] = len(all_examples)
                jobs[job_id]["progress"] = len(all_examples) / request.num_samples

            except Exception as e:
                jobs[job_id]["errors"].append(f"Batch {i}: {str(e)}")

        # Convert to dicts
        jobs[job_id]["examples"] = [example_to_dict(ex) for ex in all_examples]

        # Compute statistics
        jobs[job_id]["statistics"] = {
            "total_generated": len(all_examples),
            "success_rate": len(all_examples) / request.num_samples if request.num_samples > 0 else 0,
        }

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["errors"].append(str(e))
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


def dict_to_example(data: dict):
    """Convert dictionary to example object."""
    if "question" in data and "answer" in data:
        return RAGExample(
            id=data.get("id", str(uuid.uuid4())[:16]),
            question=data["question"],
            answer=data["answer"],
            context=data.get("context", ""),
            difficulty=DifficultyLevel[data.get("difficulty", "MEDIUM").upper()]
            if isinstance(data.get("difficulty"), str)
            else DifficultyLevel.MEDIUM,
            domain=data.get("domain"),
            metadata=data.get("metadata", {}),
        )
    elif "instruction" in data and "output" in data:
        return InstructionExample(
            id=data.get("id", str(uuid.uuid4())[:16]),
            instruction=data["instruction"],
            input=data.get("input", ""),
            output=data["output"],
            difficulty=DifficultyLevel[data.get("difficulty", "MEDIUM").upper()]
            if isinstance(data.get("difficulty"), str)
            else DifficultyLevel.MEDIUM,
            domain=data.get("domain"),
            metadata=data.get("metadata", {}),
        )
    return None


def example_to_dict(example) -> dict:
    """Convert example object to dictionary."""
    if hasattr(example, "to_dict"):
        return example.to_dict()
    elif hasattr(example, "__dataclass_fields__"):
        from dataclasses import asdict
        d = asdict(example)
        if "difficulty" in d and hasattr(d["difficulty"], "name"):
            d["difficulty"] = d["difficulty"].name
        return d
    return {}


# DVC integration utilities
class DVCIntegration:
    """Integration with DVC for data versioning."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)

    def add(self, filepath: str) -> bool:
        """Add file to DVC tracking."""
        import subprocess

        try:
            result = subprocess.run(
                ["dvc", "add", filepath],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def push(self, remote: str = None) -> bool:
        """Push tracked files to remote."""
        import subprocess

        cmd = ["dvc", "push"]
        if remote:
            cmd.extend(["-r", remote])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def pull(self, remote: str = None) -> bool:
        """Pull tracked files from remote."""
        import subprocess

        cmd = ["dvc", "pull"]
        if remote:
            cmd.extend(["-r", remote])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def checkout(self, version: str = None) -> bool:
        """Checkout specific version of data."""
        import subprocess

        cmd = ["dvc", "checkout"]
        if version:
            # First checkout git to specific version
            subprocess.run(
                ["git", "checkout", version],
                cwd=self.repo_path,
                capture_output=True,
            )

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False
