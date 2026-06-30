"""Curriculum learning and difficulty-based sampling."""

import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict, Any

from .schemas import DifficultyLevel, RAGExample, InstructionExample
from .provider import ModelProvider


@dataclass
class CurriculumLevel:
    """A single level in the curriculum."""
    name: str
    difficulty: float = 0.5
    topics: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    mastery_threshold: float = 0.8
    min_attempts: int = 1


class CurriculumManager:
    """Manage curriculum levels and progression."""

    def __init__(self):
        self._levels: Dict[str, CurriculumLevel] = {}
        self._level_order: List[str] = []
        self._completed: set = set()
        self._current_index: int = 0

    def add_level(
        self,
        name: str,
        difficulty: float = 0.5,
        topics: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Add a curriculum level."""
        level = CurriculumLevel(
            name=name,
            difficulty=difficulty,
            topics=topics or [],
            prerequisites=prerequisites or [],
            tags=tags or [],
        )
        self._levels[name] = level
        self._level_order.append(name)

    def add_level_object(self, level: CurriculumLevel) -> None:
        """Add a CurriculumLevel object directly."""
        self._levels[level.name] = level
        self._level_order.append(level.name)

    def num_levels(self) -> int:
        """Get number of levels."""
        return len(self._levels)

    def current_level(self) -> Optional[CurriculumLevel]:
        """Get current level."""
        if not self._level_order:
            return None
        return self._levels[self._level_order[self._current_index]]

    def get_level(self, name: str) -> Optional[CurriculumLevel]:
        """Get level by name."""
        return self._levels.get(name)

    def get_all_levels(self) -> List[CurriculumLevel]:
        """Get all levels in order."""
        return [self._levels[name] for name in self._level_order]

    def advance(self) -> None:
        """Advance to next level."""
        if self._current_index < len(self._level_order) - 1:
            self._current_index += 1

    def can_access_level(self, name: str) -> bool:
        """Check if a level can be accessed based on prerequisites."""
        level = self._levels.get(name)
        if not level:
            return False
        return all(prereq in self._completed for prereq in level.prerequisites)

    def mark_completed(self, name: str) -> None:
        """Mark a level as completed."""
        self._completed.add(name)

    def is_mastered(self, name: str, tracker: 'ProgressTracker') -> bool:
        """Check if a level is mastered based on tracker stats."""
        level = self._levels.get(name)
        if not level:
            return False
        stats = tracker.get_level_stats(name)
        if stats["attempts"] < level.min_attempts:
            return False
        return stats["average_score"] >= level.mastery_threshold

    def export_json(self) -> str:
        """Export curriculum to JSON."""
        data = {
            "levels": [
                {
                    "name": level.name,
                    "difficulty": level.difficulty,
                    "topics": level.topics,
                    "prerequisites": level.prerequisites,
                    "tags": level.tags,
                }
                for level in self.get_all_levels()
            ]
        }
        return json.dumps(data)

    def import_json(self, json_str: str) -> None:
        """Import curriculum from JSON."""
        data = json.loads(json_str)
        for level_data in data.get("levels", []):
            self.add_level(
                name=level_data["name"],
                difficulty=level_data.get("difficulty", 0.5),
                topics=level_data.get("topics", []),
                prerequisites=level_data.get("prerequisites", []),
                tags=level_data.get("tags", []),
            )

    def get_spaced_repetition_scheduler(self) -> 'SpacedRepetitionScheduler':
        """Get a spaced repetition scheduler."""
        return SpacedRepetitionScheduler()

    def generate_personalized_path(self, profile: Dict[str, Any]) -> List[str]:
        """Generate a personalized curriculum path based on student profile."""
        learning_style = profile.get("learning_style", "")
        path = []
        for name in self._level_order:
            level = self._levels[name]
            if learning_style in level.tags or not level.tags:
                path.append(name)
        return path


class ProgressTracker:
    """Track student progress through curriculum."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self._attempts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record_attempt(self, level: str, score: float, time_spent: int) -> None:
        """Record an attempt at a level."""
        self._attempts[level].append({
            "score": score,
            "time_spent": time_spent,
        })

    def get_level_stats(self, level: str) -> Dict[str, Any]:
        """Get statistics for a level."""
        attempts = self._attempts.get(level, [])
        if not attempts:
            return {
                "attempts": 0,
                "average_score": 0.0,
                "best_score": 0.0,
                "total_time": 0,
            }
        scores = [a["score"] for a in attempts]
        times = [a["time_spent"] for a in attempts]
        return {
            "attempts": len(attempts),
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "total_time": sum(times),
        }


class AdaptiveCurriculum:
    """Adaptive curriculum based on performance."""

    def __init__(self, advance_threshold: float = 0.85, review_threshold: float = 0.5):
        self._performance: List[float] = []
        self.advance_threshold = advance_threshold
        self.review_threshold = review_threshold

    def record_performance(self, level: str, score: float) -> None:
        """Record performance score."""
        self._performance.append(score)

    def get_recommendation(self) -> str:
        """Get recommendation based on recent performance."""
        if not self._performance:
            return "continue"
        # Use last 2 scores for more responsive feedback
        recent = self._performance[-2:] if len(self._performance) >= 2 else self._performance
        avg = sum(recent) / len(recent)
        if avg >= self.advance_threshold:
            return "advance"
        elif avg < self.review_threshold:
            return "review"
        return "continue"


class SpacedRepetitionScheduler:
    """Spaced repetition scheduler for review."""

    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}

    def add_item(self, item_id: str, initial_interval: int = 1) -> None:
        """Add an item to the scheduler."""
        self._items[item_id] = {
            "interval": initial_interval,
            "last_reviewed": 0,
            "ease_factor": 2.5,
        }

    def get_due_items(self) -> List[str]:
        """Get items due for review."""
        return list(self._items.keys())

    def update_item(self, item_id: str, performance: float) -> None:
        """Update item based on performance."""
        if item_id not in self._items:
            return
        item = self._items[item_id]
        if performance >= 0.8:
            item["interval"] = int(item["interval"] * item["ease_factor"])
        elif performance < 0.6:
            item["interval"] = max(1, item["interval"] // 2)

    def get_interval(self, item_id: str) -> int:
        """Get current interval for an item."""
        return self._items.get(item_id, {}).get("interval", 1)


class CurriculumAnalytics:
    """Analytics for curriculum data."""

    def __init__(self):
        self._trackers: List[ProgressTracker] = []

    def add_tracker(self, tracker: ProgressTracker) -> None:
        """Add a progress tracker."""
        self._trackers.append(tracker)

    def generate_insights(self) -> Dict[str, Any]:
        """Generate insights from curriculum data."""
        all_scores = []
        level_scores = defaultdict(list)

        for tracker in self._trackers:
            for level, attempts in tracker._attempts.items():
                for attempt in attempts:
                    all_scores.append(attempt["score"])
                    level_scores[level].append(attempt["score"])

        avg_progress = sum(all_scores) / len(all_scores) if all_scores else 0

        # Find struggling levels (below 70% average)
        struggling = [
            level for level, scores in level_scores.items()
            if sum(scores) / len(scores) < 0.7
        ]

        # Find top performers
        tracker_avgs = []
        for tracker in self._trackers:
            scores = []
            for attempts in tracker._attempts.values():
                scores.extend(a["score"] for a in attempts)
            if scores:
                tracker_avgs.append((tracker.student_id, sum(scores) / len(scores)))

        tracker_avgs.sort(key=lambda x: x[1], reverse=True)
        top_performers = [t[0] for t in tracker_avgs[:3]]

        return {
            "average_progress": avg_progress,
            "struggling_levels": struggling,
            "top_performers": top_performers,
        }


class CurriculumSampler:
    """Sample examples with curriculum learning (easy to hard)."""

    def __init__(
        self,
        initial_difficulty: float = 0.2,
        difficulty_increase_rate: float = 0.1,
        warmup_steps: int = 100,
    ):
        self.current_difficulty = initial_difficulty
        self.increase_rate = difficulty_increase_rate
        self.warmup_steps = warmup_steps
        self.step = 0

    def sample_batch(
        self,
        examples: list,
        batch_size: int,
    ) -> list:
        """Sample batch based on current curriculum stage."""
        # Group by difficulty
        by_difficulty = defaultdict(list)
        for ex in examples:
            by_difficulty[ex.difficulty.value].append(ex)

        # Compute sampling weights
        weights = self._compute_weights()

        # Sample from each difficulty level
        batch = []
        for diff_value, weight in weights.items():
            count = int(batch_size * weight)
            available = by_difficulty.get(diff_value, [])

            if available:
                sampled = np.random.choice(
                    available,
                    size=min(count, len(available)),
                    replace=False
                ).tolist()
                batch.extend(sampled)

        # Fill remaining with medium difficulty
        while len(batch) < batch_size:
            medium = by_difficulty.get(DifficultyLevel.MEDIUM.value, [])
            if medium:
                batch.append(np.random.choice(medium))
            else:
                break

        # Update curriculum
        self._step()

        return batch[:batch_size]

    def _compute_weights(self) -> dict:
        """Compute sampling weights based on current difficulty."""
        weights = {}

        for diff in DifficultyLevel:
            # Higher difficulty -> need higher current_difficulty
            threshold = diff.value / 4.0
            weight = 1 / (1 + np.exp(-10 * (self.current_difficulty - threshold)))
            weights[diff.value] = weight

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _step(self):
        """Update curriculum difficulty."""
        self.step += 1

        if self.step > self.warmup_steps:
            self.current_difficulty = min(
                1.0,
                self.current_difficulty + self.increase_rate
            )

    def get_difficulty_for_step(self, step: int) -> float:
        """Get target difficulty for a given step."""
        if step <= self.warmup_steps:
            return self.current_difficulty
        else:
            return min(
                1.0,
                self.current_difficulty + (step - self.warmup_steps) * self.increase_rate
            )

    def reset(self):
        """Reset curriculum to initial state."""
        self.step = 0
        self.current_difficulty = 0.2


class DifficultyScorer:
    """Score difficulty of examples using LLM."""

    def __init__(self, model_provider: ModelProvider):
        self.model = model_provider

    async def score_difficulty(
        self,
        example: Union[RAGExample, InstructionExample],
    ) -> DifficultyLevel:
        """Estimate difficulty level of example."""
        if isinstance(example, RAGExample):
            content = f"Question: {example.question}\nAnswer: {example.answer}"
        else:
            content = f"Instruction: {example.instruction}\nOutput: {example.output}"

        prompt = f"""Rate the difficulty level of this example.

{content}

Consider:
- Complexity of language
- Amount of reasoning required
- Domain expertise needed
- Length and detail of response

Output only one of: EASY, MEDIUM, HARD, EXPERT"""

        try:
            response = await self.model.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
            )

            difficulty_map = {
                "EASY": DifficultyLevel.EASY,
                "MEDIUM": DifficultyLevel.MEDIUM,
                "HARD": DifficultyLevel.HARD,
                "EXPERT": DifficultyLevel.EXPERT,
            }

            return difficulty_map.get(
                response.strip().upper(),
                DifficultyLevel.MEDIUM
            )
        except Exception:
            return DifficultyLevel.MEDIUM

    async def batch_score(
        self,
        examples: list,
    ) -> list[DifficultyLevel]:
        """Score difficulty for a batch of examples."""
        import asyncio
        tasks = [self.score_difficulty(ex) for ex in examples]
        return await asyncio.gather(*tasks)


class BalancedBatchSampler:
    """Sample balanced batches across difficulty levels and domains."""

    def __init__(
        self,
        difficulty_balance: bool = True,
        domain_balance: bool = True,
    ):
        self.difficulty_balance = difficulty_balance
        self.domain_balance = domain_balance

    def sample(
        self,
        examples: list,
        batch_size: int,
    ) -> list:
        """Sample a balanced batch."""
        if not examples:
            return []

        # Group examples
        groups = defaultdict(list)

        for ex in examples:
            key = []
            if self.difficulty_balance:
                key.append(ex.difficulty.name)
            if self.domain_balance and hasattr(ex, 'domain'):
                key.append(ex.domain or 'general')

            groups[tuple(key)].append(ex)

        # Sample equally from each group
        if not groups:
            return examples[:batch_size]

        per_group = max(1, batch_size // len(groups))
        batch = []

        for key, group_examples in groups.items():
            sampled = np.random.choice(
                group_examples,
                size=min(per_group, len(group_examples)),
                replace=False
            ).tolist()
            batch.extend(sampled)

        # Fill remaining randomly
        remaining = batch_size - len(batch)
        if remaining > 0:
            all_remaining = [ex for ex in examples if ex not in batch]
            if all_remaining:
                additional = np.random.choice(
                    all_remaining,
                    size=min(remaining, len(all_remaining)),
                    replace=False
                ).tolist()
                batch.extend(additional)

        return batch[:batch_size]
