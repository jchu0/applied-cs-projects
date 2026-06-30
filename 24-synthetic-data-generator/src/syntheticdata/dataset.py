"""Dataset management with versioning and export."""

import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np

from .schemas import RAGExample, InstructionExample, ConversationExample


class Dataset:
    """Simple container for a collection of examples with metadata."""

    def __init__(self, samples: list = None, metadata: dict = None):
        self.samples = samples or []
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def to_list(self) -> list:
        """Convert samples to list of dicts."""
        result = []
        for sample in self.samples:
            if hasattr(sample, 'to_dict'):
                result.append(sample.to_dict())
            else:
                result.append(sample)
        return result

    def export_json(self, path: Path) -> Path:
        """Export dataset to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_list(), f, indent=2)
        return path

    def export_jsonl(self, path: Path) -> Path:
        """Export dataset to JSONL file."""
        path = Path(path)
        with open(path, "w") as f:
            for item in self.to_list():
                f.write(json.dumps(item) + "\n")
        return path

    def export_parquet(self, path: Path) -> Path:
        """Export dataset to Parquet file."""
        path = Path(path)
        try:
            import pandas as pd
            df = pd.DataFrame(self.to_list())
            df.to_parquet(path)
        except ImportError:
            raise ImportError("pandas and pyarrow required for parquet format")
        return path

    def export_csv(self, path: Path) -> Path:
        """Export dataset to CSV file."""
        path = Path(path)
        try:
            import pandas as pd
            df = pd.DataFrame(self.to_list())
            df.to_csv(path, index=False)
        except ImportError:
            raise ImportError("pandas required for CSV format")
        return path


class DatasetManager:
    """Manage generated datasets with versioning and curation."""

    def __init__(
        self,
        output_dir: str,
        use_dvc: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_dvc = use_dvc
        self._samples: list = []

    def add_samples(self, samples: list) -> None:
        """Add samples to the internal collection."""
        self._samples.extend(samples)

    def get_samples(self) -> list:
        """Get all collected samples."""
        return self._samples

    def clear_samples(self) -> None:
        """Clear all collected samples."""
        self._samples = []

    def save_checkpoint(self, name: str) -> Path:
        """Save current samples as a checkpoint."""
        return self.save_dataset(self._samples, name, format="jsonl")

    def load_checkpoint(self, name: str) -> list:
        """Load samples from a checkpoint."""
        filepath = self.output_dir / f"{name}.jsonl"
        if filepath.exists():
            self._samples = self.load_dataset(filepath)
        return self._samples

    def build(self) -> "Dataset":
        """Build a Dataset from collected samples."""
        return Dataset(samples=self._samples.copy(), metadata={})

    def save_dataset(
        self,
        examples: list,
        name: str,
        format: str = "jsonl",
    ) -> Path:
        """Save dataset to file."""
        filepath = self.output_dir / f"{name}.{format}"

        if format == "jsonl":
            with open(filepath, "w") as f:
                for ex in examples:
                    f.write(json.dumps(self._to_dict(ex)) + "\n")
        elif format == "json":
            with open(filepath, "w") as f:
                json.dump([self._to_dict(ex) for ex in examples], f, indent=2)
        elif format == "parquet":
            try:
                import pandas as pd
                df = pd.DataFrame([self._to_dict(ex) for ex in examples])
                df.to_parquet(filepath)
            except ImportError:
                raise ImportError("pandas and pyarrow required for parquet format")
        else:
            raise ValueError(f"Unknown format: {format}")

        # Version with DVC if enabled
        if self.use_dvc:
            import subprocess
            subprocess.run(["dvc", "add", str(filepath)], check=False)

        return filepath

    def load_dataset(self, filepath: Path) -> list[dict]:
        """Load dataset from file."""
        filepath = Path(filepath)
        examples = []

        if filepath.suffix == ".jsonl":
            with open(filepath) as f:
                for line in f:
                    examples.append(json.loads(line))
        elif filepath.suffix == ".json":
            with open(filepath) as f:
                examples = json.load(f)
        elif filepath.suffix == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(filepath)
                examples = df.to_dict("records")
            except ImportError:
                raise ImportError("pandas required for parquet format")

        return examples

    def list_datasets(self) -> list[dict]:
        """List all datasets in output directory."""
        datasets = []
        for path in self.output_dir.glob("*"):
            if path.suffix in [".jsonl", ".json", ".parquet"]:
                stats = path.stat()
                datasets.append({
                    "name": path.stem,
                    "format": path.suffix[1:],
                    "size_bytes": stats.st_size,
                    "modified": stats.st_mtime,
                })
        return datasets

    def deduplicate(
        self,
        examples: list,
        similarity_threshold: float = 0.9,
    ) -> list:
        """Remove duplicate or near-duplicate examples."""
        seen_hashes = set()
        unique = []

        for ex in examples:
            # Hash based on content
            if isinstance(ex, RAGExample):
                content = f"{ex.question}|{ex.answer}"
            elif isinstance(ex, InstructionExample):
                content = f"{ex.instruction}|{ex.output}"
            elif isinstance(ex, ConversationExample):
                content = str(ex.messages)
            else:
                content = str(ex)

            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(ex)

        return unique

    def check_bias(
        self,
        examples: list,
        dimensions: list[str] = None,
    ) -> dict:
        """Check for biases in dataset."""
        dimensions = dimensions or ["difficulty", "length", "domain"]
        report = {}

        # Difficulty distribution
        if "difficulty" in dimensions:
            diff_counts = defaultdict(int)
            for ex in examples:
                if hasattr(ex, 'difficulty'):
                    diff_counts[ex.difficulty.name] += 1
            report["difficulty_distribution"] = dict(diff_counts)

        # Length distribution
        if "length" in dimensions:
            lengths = []
            for ex in examples:
                if isinstance(ex, RAGExample):
                    lengths.append(len(ex.answer))
                elif isinstance(ex, InstructionExample):
                    lengths.append(len(ex.output))
                elif isinstance(ex, ConversationExample):
                    lengths.append(sum(len(m["content"]) for m in ex.messages))

            if lengths:
                report["length_stats"] = {
                    "mean": float(np.mean(lengths)),
                    "std": float(np.std(lengths)),
                    "min": int(np.min(lengths)),
                    "max": int(np.max(lengths)),
                }

        # Domain distribution
        if "domain" in dimensions:
            domain_counts = defaultdict(int)
            for ex in examples:
                domain = getattr(ex, 'domain', None) or "general"
                domain_counts[domain] += 1
            report["domain_distribution"] = dict(domain_counts)

        return report

    def export_for_training(
        self,
        examples: list,
        format: str = "sharegpt",
        output_path: Path = None,
    ) -> Path:
        """Export in format suitable for training."""
        output_path = output_path or self.output_dir / f"train_{format}.json"

        if format == "sharegpt":
            # ShareGPT format for many fine-tuning tools
            converted = []
            for ex in examples:
                if isinstance(ex, InstructionExample):
                    input_text = ex.instruction
                    if ex.input:
                        input_text += f"\n\n{ex.input}"
                    converted.append({
                        "conversations": [
                            {"from": "human", "value": input_text},
                            {"from": "gpt", "value": ex.output},
                        ]
                    })
                elif isinstance(ex, ConversationExample):
                    convs = []
                    for msg in ex.messages:
                        role = "human" if msg["role"] == "user" else "gpt"
                        convs.append({"from": role, "value": msg["content"]})
                    converted.append({"conversations": convs})
                elif isinstance(ex, RAGExample):
                    converted.append({
                        "conversations": [
                            {"from": "human", "value": f"Context: {ex.context}\n\nQuestion: {ex.question}"},
                            {"from": "gpt", "value": ex.answer},
                        ]
                    })

        elif format == "alpaca":
            # Alpaca format
            converted = []
            for ex in examples:
                if isinstance(ex, InstructionExample):
                    converted.append({
                        "instruction": ex.instruction,
                        "input": ex.input,
                        "output": ex.output,
                    })
                elif isinstance(ex, RAGExample):
                    converted.append({
                        "instruction": ex.question,
                        "input": ex.context,
                        "output": ex.answer,
                    })

        elif format == "openai":
            # OpenAI fine-tuning format
            converted = []
            for ex in examples:
                if isinstance(ex, InstructionExample):
                    converted.append({
                        "messages": [
                            {"role": "user", "content": f"{ex.instruction}\n\n{ex.input}"},
                            {"role": "assistant", "content": ex.output},
                        ]
                    })
                elif isinstance(ex, ConversationExample):
                    msgs = []
                    if ex.system_prompt:
                        msgs.append({"role": "system", "content": ex.system_prompt})
                    for m in ex.messages:
                        msgs.append({"role": m["role"], "content": m["content"]})
                    converted.append({"messages": msgs})

        else:
            raise ValueError(f"Unknown export format: {format}")

        with open(output_path, "w") as f:
            json.dump(converted, f, indent=2)

        return output_path

    def merge_datasets(
        self,
        datasets: list[Path],
        output_name: str,
        dedupe: bool = True,
    ) -> Path:
        """Merge multiple datasets into one."""
        all_examples = []

        for dataset_path in datasets:
            examples = self.load_dataset(dataset_path)
            all_examples.extend(examples)

        if dedupe:
            # Simple hash-based deduplication
            seen = set()
            unique = []
            for ex in all_examples:
                ex_hash = hashlib.md5(json.dumps(ex, sort_keys=True).encode()).hexdigest()
                if ex_hash not in seen:
                    seen.add(ex_hash)
                    unique.append(ex)
            all_examples = unique

        return self.save_dataset(all_examples, output_name)

    def _to_dict(self, example) -> dict:
        """Convert example to dictionary."""
        if hasattr(example, 'to_dict'):
            return example.to_dict()
        elif hasattr(example, "__dataclass_fields__"):
            from dataclasses import asdict
            d = asdict(example)
            # Convert enums
            if "difficulty" in d and hasattr(d["difficulty"], "name"):
                d["difficulty"] = d["difficulty"].name
            return d
        return example
