"""Tests for curriculum-based data generation."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from syntheticdata.curriculum import (
    CurriculumManager,
    CurriculumLevel,
    ProgressTracker,
    AdaptiveCurriculum,
    CurriculumAnalytics,
)


class TestCurriculumManager:
    """Test curriculum management functionality."""

    def test_curriculum_initialization(self):
        """Test curriculum manager initialization."""
        manager = CurriculumManager()
        assert manager.num_levels() == 0
        assert manager.current_level() is None

    def test_add_curriculum_levels(self):
        """Test adding curriculum levels."""
        manager = CurriculumManager()

        manager.add_level("beginner", difficulty=0.2, topics=["basics"])
        manager.add_level("intermediate", difficulty=0.5, topics=["advanced"])
        manager.add_level("expert", difficulty=0.9, topics=["complex"])

        assert manager.num_levels() == 3
        assert manager.get_level("beginner").difficulty == 0.2

    def test_level_progression(self):
        """Test progressing through curriculum levels."""
        manager = CurriculumManager()

        levels = [
            ("level1", 0.3),
            ("level2", 0.6),
            ("level3", 0.9),
        ]

        for name, difficulty in levels:
            manager.add_level(name, difficulty=difficulty)

        # Start at first level
        assert manager.current_level().name == "level1"

        # Progress to next level
        manager.advance()
        assert manager.current_level().name == "level2"

        # Progress again
        manager.advance()
        assert manager.current_level().name == "level3"

        # Can't advance beyond last level
        manager.advance()
        assert manager.current_level().name == "level3"

    def test_prerequisite_management(self):
        """Test handling level prerequisites."""
        manager = CurriculumManager()

        manager.add_level("basic", difficulty=0.2)
        manager.add_level("intermediate", difficulty=0.5, prerequisites=["basic"])
        manager.add_level("advanced", difficulty=0.8, prerequisites=["intermediate"])

        # Can't jump to advanced without completing prerequisites
        assert not manager.can_access_level("advanced")

        manager.mark_completed("basic")
        assert manager.can_access_level("intermediate")
        assert not manager.can_access_level("advanced")

        manager.mark_completed("intermediate")
        assert manager.can_access_level("advanced")

    def test_adaptive_difficulty_adjustment(self):
        """Test adaptive difficulty based on performance."""
        adaptive = AdaptiveCurriculum()

        # Add performance data
        adaptive.record_performance(level="current", score=0.9)
        adaptive.record_performance(level="current", score=0.95)
        adaptive.record_performance(level="current", score=0.92)

        # Should recommend advancing
        recommendation = adaptive.get_recommendation()
        assert recommendation == "advance"

        # Record poor performance
        adaptive.record_performance(level="current", score=0.4)
        adaptive.record_performance(level="current", score=0.3)

        # Should recommend reviewing
        recommendation = adaptive.get_recommendation()
        assert recommendation == "review"

    @pytest.mark.asyncio
    async def test_curriculum_based_generation(self):
        """Test generating data based on curriculum."""
        manager = CurriculumManager()

        manager.add_level("easy", difficulty=0.3, topics=["addition"])
        manager.add_level("medium", difficulty=0.6, topics=["multiplication"])
        manager.add_level("hard", difficulty=0.9, topics=["calculus"])

        # Mock generator
        async def mock_generate(level):
            return {
                "level": level.name,
                "difficulty": level.difficulty,
                "sample": f"Sample for {level.name}"
            }

        samples = []
        for level in manager.get_all_levels():
            sample = await mock_generate(level)
            samples.append(sample)

        assert len(samples) == 3
        assert samples[0]["difficulty"] == 0.3
        assert samples[-1]["difficulty"] == 0.9

    def test_progress_tracking(self):
        """Test tracking student progress through curriculum."""
        tracker = ProgressTracker(student_id="student_001")

        # Track progress
        tracker.record_attempt("level1", score=0.7, time_spent=120)
        tracker.record_attempt("level1", score=0.8, time_spent=100)
        tracker.record_attempt("level1", score=0.9, time_spent=90)

        stats = tracker.get_level_stats("level1")
        assert stats["attempts"] == 3
        assert abs(stats["average_score"] - 0.8) < 0.01  # Floating point tolerance
        assert stats["best_score"] == 0.9
        assert stats["total_time"] == 310

    def test_curriculum_branching(self):
        """Test branching paths in curriculum."""
        manager = CurriculumManager()

        # Create branching curriculum
        manager.add_level("foundation", difficulty=0.3)
        manager.add_level("path_a", difficulty=0.6, prerequisites=["foundation"])
        manager.add_level("path_b", difficulty=0.6, prerequisites=["foundation"])
        manager.add_level("advanced_a", difficulty=0.9, prerequisites=["path_a"])
        manager.add_level("advanced_b", difficulty=0.9, prerequisites=["path_b"])

        manager.mark_completed("foundation")

        # Both paths should be accessible
        assert manager.can_access_level("path_a")
        assert manager.can_access_level("path_b")

        # Choose path A
        manager.mark_completed("path_a")
        assert manager.can_access_level("advanced_a")
        assert not manager.can_access_level("advanced_b")

    def test_curriculum_export_import(self):
        """Test exporting and importing curriculum definitions."""
        manager = CurriculumManager()

        manager.add_level("level1", difficulty=0.3, topics=["topic1"])
        manager.add_level("level2", difficulty=0.6, topics=["topic2"])

        # Export curriculum
        exported = manager.export_json()

        # Create new manager and import
        new_manager = CurriculumManager()
        new_manager.import_json(exported)

        assert new_manager.num_levels() == 2
        assert new_manager.get_level("level1").difficulty == 0.3

    def test_mastery_criteria(self):
        """Test mastery criteria for level completion."""
        manager = CurriculumManager()

        level = CurriculumLevel(
            name="test_level",
            difficulty=0.5,
            mastery_threshold=0.85,
            min_attempts=3
        )

        manager.add_level_object(level)

        tracker = ProgressTracker("student_001")

        # Not enough attempts
        tracker.record_attempt("test_level", score=0.9, time_spent=100)
        assert not manager.is_mastered("test_level", tracker)

        # More attempts but below threshold
        tracker.record_attempt("test_level", score=0.7, time_spent=100)
        tracker.record_attempt("test_level", score=0.8, time_spent=100)
        assert not manager.is_mastered("test_level", tracker)

        # Achieve mastery
        tracker.record_attempt("test_level", score=0.9, time_spent=100)
        tracker.record_attempt("test_level", score=0.95, time_spent=100)
        assert manager.is_mastered("test_level", tracker)

    @pytest.mark.asyncio
    async def test_spaced_repetition_scheduling(self):
        """Test spaced repetition for curriculum review."""
        manager = CurriculumManager()
        scheduler = manager.get_spaced_repetition_scheduler()

        # Add completed levels
        scheduler.add_item("level1", initial_interval=1)  # 1 day
        scheduler.add_item("level2", initial_interval=3)  # 3 days

        # Simulate time passing
        await asyncio.sleep(0.1)  # Simulate time

        # Get items due for review
        due_items = scheduler.get_due_items()

        # Update based on performance
        scheduler.update_item("level1", performance=0.9)  # Good performance
        scheduler.update_item("level2", performance=0.5)  # Poor performance

        # Check updated intervals
        assert scheduler.get_interval("level1") > 1  # Interval increased
        assert scheduler.get_interval("level2") <= 3  # Interval same or decreased

    def test_curriculum_analytics(self):
        """Test analytics and insights from curriculum data."""
        analytics = CurriculumAnalytics()

        # Add student progress data
        for student_id in range(10):
            tracker = ProgressTracker(f"student_{student_id}")
            for level in ["level1", "level2", "level3"]:
                score = 0.6 + (student_id * 0.03)  # Varying scores
                tracker.record_attempt(level, score=min(score, 1.0), time_spent=100)
            analytics.add_tracker(tracker)

        # Get insights
        insights = analytics.generate_insights()

        assert "average_progress" in insights
        assert "struggling_levels" in insights
        assert "top_performers" in insights
        assert len(insights["top_performers"]) <= 3

    def test_personalized_curriculum_path(self):
        """Test generating personalized curriculum paths."""
        manager = CurriculumManager()

        # Add levels with different focuses
        manager.add_level("visual_basic", difficulty=0.3, tags=["visual"])
        manager.add_level("verbal_basic", difficulty=0.3, tags=["verbal"])
        manager.add_level("visual_advanced", difficulty=0.7, tags=["visual"])
        manager.add_level("verbal_advanced", difficulty=0.7, tags=["verbal"])

        # Student profile
        student_profile = {
            "learning_style": "visual",
            "pace": "moderate",
            "interests": ["geometry", "graphics"]
        }

        # Generate personalized path
        path = manager.generate_personalized_path(student_profile)

        # Should prefer visual levels
        assert "visual_basic" in path
        assert "visual_advanced" in path