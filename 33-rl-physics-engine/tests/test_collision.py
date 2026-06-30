"""Tests for collision detection system."""

import pytest
import numpy as np

from physicsrl import (
    Model, Body, Geom, Joint, Inertia, State,
    GeomType, JointType, Contact
)
from physicsrl.collision import BroadPhase, NarrowPhase, CollisionSystem


class TestBroadPhase:
    """Tests for AABB broad phase collision detection."""

    def test_broad_phase_creation(self, broad_phase):
        """Test BroadPhase can be created."""
        assert broad_phase.aabb_tree is None  # Initial state

    def test_aabb_overlap_true(self, broad_phase):
        """Test AABB overlap detection for overlapping boxes."""
        aabb1 = (np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))
        aabb2 = (np.array([1.0, 1.0, 1.0]), np.array([3.0, 3.0, 3.0]))

        assert broad_phase._aabb_overlap(aabb1, aabb2)

    def test_aabb_overlap_false(self, broad_phase):
        """Test AABB overlap detection for non-overlapping boxes."""
        aabb1 = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        aabb2 = (np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0]))

        assert not broad_phase._aabb_overlap(aabb1, aabb2)

    def test_aabb_overlap_touching(self, broad_phase):
        """Test AABB overlap for touching boxes."""
        aabb1 = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        aabb2 = (np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]))

        # Touching should be considered overlapping
        assert broad_phase._aabb_overlap(aabb1, aabb2)

    def test_aabb_overlap_one_axis(self, broad_phase):
        """Test AABB overlap when only one axis overlaps."""
        aabb1 = (np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))
        aabb2 = (np.array([1.0, 5.0, 5.0]), np.array([3.0, 6.0, 6.0]))

        # Only X overlaps, not a 3D overlap
        assert not broad_phase._aabb_overlap(aabb1, aabb2)

    def test_find_overlapping_pairs_no_overlap(self, two_sphere_model, broad_phase):
        """Test no pairs found when spheres don't overlap."""
        state = State.create(two_sphere_model)

        # Set sphere positions far apart
        state.xpos[0] = np.array([-5.0, 0.0, 0.0])
        state.xpos[1] = np.array([5.0, 0.0, 0.0])
        state.xquat[0] = np.array([1.0, 0.0, 0.0, 0.0])
        state.xquat[1] = np.array([1.0, 0.0, 0.0, 0.0])

        pairs = broad_phase.find_overlapping_pairs(two_sphere_model, state)
        assert len(pairs) == 0

    def test_find_overlapping_pairs_with_overlap(self, two_sphere_model, broad_phase):
        """Test pairs found when spheres overlap."""
        state = State.create(two_sphere_model)

        # Set sphere positions to overlap
        state.xpos[0] = np.array([0.0, 0.0, 0.0])
        state.xpos[1] = np.array([1.5, 0.0, 0.0])  # radius 1 each, gap = -0.5
        state.xquat[0] = np.array([1.0, 0.0, 0.0, 0.0])
        state.xquat[1] = np.array([1.0, 0.0, 0.0, 0.0])

        pairs = broad_phase.find_overlapping_pairs(two_sphere_model, state)
        assert len(pairs) == 1
        assert pairs[0] == (0, 1)


class TestNarrowPhaseSpheres:
    """Tests for narrow phase sphere collision detection."""

    def test_narrow_phase_creation(self, narrow_phase):
        """Test NarrowPhase can be created."""
        assert narrow_phase is not None

    def test_sphere_sphere_collision(self, narrow_phase):
        """Test sphere-sphere collision detection."""
        sphere1 = Geom(GeomType.SPHERE, np.array([1.0]))
        sphere2 = Geom(GeomType.SPHERE, np.array([1.0]))

        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([1.5, 0.0, 0.0])  # Overlapping
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere1, pos1, quat,
            sphere2, pos2, quat
        )

        assert contact is not None
        assert contact.penetration > 0
        np.testing.assert_allclose(contact.penetration, 0.5, atol=1e-6)

    def test_sphere_sphere_no_collision(self, narrow_phase):
        """Test sphere-sphere when not colliding."""
        sphere1 = Geom(GeomType.SPHERE, np.array([1.0]))
        sphere2 = Geom(GeomType.SPHERE, np.array([1.0]))

        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([3.0, 0.0, 0.0])  # Not touching
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere1, pos1, quat,
            sphere2, pos2, quat
        )

        assert contact is None

    def test_sphere_sphere_touching(self, narrow_phase):
        """Test sphere-sphere just touching."""
        sphere1 = Geom(GeomType.SPHERE, np.array([1.0]))
        sphere2 = Geom(GeomType.SPHERE, np.array([1.0]))

        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([2.0, 0.0, 0.0])  # Exactly touching
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere1, pos1, quat,
            sphere2, pos2, quat
        )

        # Exactly touching is not penetrating
        assert contact is None

    def test_sphere_sphere_normal_direction(self, narrow_phase):
        """Test sphere-sphere contact normal direction."""
        sphere1 = Geom(GeomType.SPHERE, np.array([1.0]))
        sphere2 = Geom(GeomType.SPHERE, np.array([1.0]))

        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([1.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere1, pos1, quat,
            sphere2, pos2, quat
        )

        # Normal should point from sphere1 to sphere2
        expected_normal = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(contact.normal, expected_normal, atol=1e-6)


class TestNarrowPhaseSphereBox:
    """Tests for narrow phase sphere-box collision detection."""

    def test_sphere_box_collision(self, narrow_phase):
        """Test sphere-box collision."""
        sphere = Geom(GeomType.SPHERE, np.array([1.0]))
        box = Geom(GeomType.BOX, np.array([1.0, 1.0, 1.0]))

        sphere_pos = np.array([1.5, 0.0, 0.0])
        box_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere, sphere_pos, quat,
            box, box_pos, quat
        )

        assert contact is not None
        assert contact.penetration > 0

    def test_sphere_box_no_collision(self, narrow_phase):
        """Test sphere-box when not colliding."""
        sphere = Geom(GeomType.SPHERE, np.array([0.5]))
        box = Geom(GeomType.BOX, np.array([1.0, 1.0, 1.0]))

        sphere_pos = np.array([3.0, 0.0, 0.0])
        box_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere, sphere_pos, quat,
            box, box_pos, quat
        )

        assert contact is None


class TestNarrowPhaseSpherePlane:
    """Tests for narrow phase sphere-plane collision detection."""

    def test_sphere_plane_collision(self, narrow_phase):
        """Test sphere-plane collision."""
        sphere = Geom(GeomType.SPHERE, np.array([1.0]))
        plane = Geom(GeomType.PLANE, np.array([10.0, 10.0, 0.1]))

        sphere_pos = np.array([0.0, 0.0, 0.5])  # Below plane surface
        plane_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere, sphere_pos, quat,
            plane, plane_pos, quat
        )

        assert contact is not None
        assert contact.penetration > 0

    def test_sphere_plane_no_collision(self, narrow_phase):
        """Test sphere-plane when not colliding."""
        sphere = Geom(GeomType.SPHERE, np.array([1.0]))
        plane = Geom(GeomType.PLANE, np.array([10.0, 10.0, 0.1]))

        sphere_pos = np.array([0.0, 0.0, 2.0])  # Above plane
        plane_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere, sphere_pos, quat,
            plane, plane_pos, quat
        )

        assert contact is None

    def test_sphere_plane_penetration_depth(self, narrow_phase):
        """Test sphere-plane penetration depth calculation."""
        sphere = Geom(GeomType.SPHERE, np.array([1.0]))
        plane = Geom(GeomType.PLANE, np.array([10.0, 10.0, 0.1]))

        # Sphere at z=0.3, radius=1.0 -> penetration = 1.0 - 0.3 = 0.7
        sphere_pos = np.array([0.0, 0.0, 0.3])
        plane_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere, sphere_pos, quat,
            plane, plane_pos, quat
        )

        np.testing.assert_allclose(contact.penetration, 0.7, atol=1e-6)


class TestNarrowPhaseCapsule:
    """Tests for narrow phase capsule collision detection."""

    def test_capsule_plane_collision(self, narrow_phase):
        """Test capsule-plane collision."""
        capsule = Geom(GeomType.CAPSULE, np.array([0.5, 1.0]))
        plane = Geom(GeomType.PLANE, np.array([10.0, 10.0, 0.1]))

        capsule_pos = np.array([0.0, 0.0, 0.3])
        plane_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            capsule, capsule_pos, quat,
            plane, plane_pos, quat
        )

        assert contact is not None
        assert contact.penetration > 0

    def test_capsule_capsule_collision(self, narrow_phase):
        """Test capsule-capsule collision."""
        capsule1 = Geom(GeomType.CAPSULE, np.array([0.5, 1.0]))
        capsule2 = Geom(GeomType.CAPSULE, np.array([0.5, 1.0]))

        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([0.8, 0.0, 0.0])  # Overlapping
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            capsule1, pos1, quat,
            capsule2, pos2, quat
        )

        assert contact is not None
        assert contact.penetration > 0


class TestNarrowPhaseBoxPlane:
    """Tests for narrow phase box-plane collision detection."""

    def test_box_plane_collision(self, narrow_phase):
        """Test box-plane collision."""
        box = Geom(GeomType.BOX, np.array([0.5, 0.5, 0.5]))
        plane = Geom(GeomType.PLANE, np.array([10.0, 10.0, 0.1]))

        box_pos = np.array([0.0, 0.0, 0.3])  # Box penetrating plane
        plane_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            box, box_pos, quat,
            plane, plane_pos, quat
        )

        assert contact is not None
        assert contact.penetration > 0

    def test_box_plane_no_collision(self, narrow_phase):
        """Test box-plane when not colliding."""
        box = Geom(GeomType.BOX, np.array([0.5, 0.5, 0.5]))
        plane = Geom(GeomType.PLANE, np.array([10.0, 10.0, 0.1]))

        box_pos = np.array([0.0, 0.0, 1.0])  # Above plane
        plane_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            box, box_pos, quat,
            plane, plane_pos, quat
        )

        assert contact is None


class TestNarrowPhaseClosestPoints:
    """Tests for closest points on line segments utility."""

    def test_closest_points_parallel_segments(self, narrow_phase):
        """Test closest points on parallel segments."""
        a1 = np.array([0.0, 0.0, 0.0])
        b1 = np.array([1.0, 0.0, 0.0])
        a2 = np.array([0.0, 1.0, 0.0])
        b2 = np.array([1.0, 1.0, 0.0])

        c1, c2 = narrow_phase._closest_points_segments(a1, b1, a2, b2)

        # Closest points should have same x, different y
        assert c1[1] == 0.0
        assert c2[1] == 1.0

    def test_closest_points_perpendicular_segments(self, narrow_phase):
        """Test closest points on perpendicular segments."""
        a1 = np.array([0.0, 0.0, 0.0])
        b1 = np.array([2.0, 0.0, 0.0])
        a2 = np.array([1.0, -1.0, 0.0])
        b2 = np.array([1.0, 1.0, 0.0])

        c1, c2 = narrow_phase._closest_points_segments(a1, b1, a2, b2)

        np.testing.assert_allclose(c1, np.array([1.0, 0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(c2, np.array([1.0, 0.0, 0.0]), atol=1e-6)


class TestCollisionSystem:
    """Tests for complete collision detection system."""

    def test_collision_system_creation(self, falling_sphere_model):
        """Test CollisionSystem can be created."""
        system = CollisionSystem(falling_sphere_model)
        assert system.model is falling_sphere_model
        assert system.broad_phase is not None
        assert system.narrow_phase is not None

    def test_detect_contacts_empty(self, falling_sphere_model):
        """Test no contacts when sphere is above ground."""
        system = CollisionSystem(falling_sphere_model)
        state = State.create(falling_sphere_model)

        # Position sphere high above ground
        state.xpos[1] = np.array([0.0, 0.0, 10.0])
        state.xquat[0] = np.array([1.0, 0.0, 0.0, 0.0])
        state.xquat[1] = np.array([1.0, 0.0, 0.0, 0.0])

        contacts = system.detect_contacts(state)
        assert len(contacts) == 0

    def test_detect_contacts_with_collision(self, falling_sphere_model):
        """Test contacts detected when sphere touches ground."""
        system = CollisionSystem(falling_sphere_model)
        state = State.create(falling_sphere_model)

        # Position sphere at ground level (radius 1.0)
        state.xpos[0] = np.array([0.0, 0.0, 0.0])  # Ground
        state.xpos[1] = np.array([0.0, 0.0, 0.5])  # Sphere penetrating
        state.xquat[0] = np.array([1.0, 0.0, 0.0, 0.0])
        state.xquat[1] = np.array([1.0, 0.0, 0.0, 0.0])

        contacts = system.detect_contacts(state)
        assert len(contacts) >= 1

    def test_contact_body_indices(self, falling_sphere_model):
        """Test contact has correct body indices."""
        system = CollisionSystem(falling_sphere_model)
        state = State.create(falling_sphere_model)

        state.xpos[0] = np.array([0.0, 0.0, 0.0])
        state.xpos[1] = np.array([0.0, 0.0, 0.5])
        state.xquat[0] = np.array([1.0, 0.0, 0.0, 0.0])
        state.xquat[1] = np.array([1.0, 0.0, 0.0, 0.0])

        contacts = system.detect_contacts(state)

        if contacts:
            contact = contacts[0]
            # One body should be 0 (ground), other should be 1 (sphere)
            assert {contact.body1_idx, contact.body2_idx} == {0, 1}


class TestContactProperties:
    """Tests for contact property computation."""

    def test_contact_friction(self, narrow_phase):
        """Test contact friction is computed correctly."""
        sphere1 = Geom(GeomType.SPHERE, np.array([1.0]), friction=0.5)
        sphere2 = Geom(GeomType.SPHERE, np.array([1.0]), friction=0.8)

        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([1.5, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere1, pos1, quat,
            sphere2, pos2, quat
        )

        # Friction should be min of the two
        assert contact.friction == 0.5

    def test_contact_restitution(self, narrow_phase):
        """Test contact restitution is computed correctly."""
        sphere1 = Geom(GeomType.SPHERE, np.array([1.0]), restitution=0.2)
        sphere2 = Geom(GeomType.SPHERE, np.array([1.0]), restitution=0.6)

        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([1.5, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            sphere1, pos1, quat,
            sphere2, pos2, quat
        )

        # Restitution should be max of the two
        assert contact.restitution == 0.6


class TestCollisionDispatch:
    """Tests for collision type dispatch."""

    def test_commutative_sphere_plane(self, narrow_phase):
        """Test sphere-plane collision is order-independent for penetration."""
        sphere = Geom(GeomType.SPHERE, np.array([1.0]))
        plane = Geom(GeomType.PLANE, np.array([10.0, 10.0, 0.1]))

        sphere_pos = np.array([0.0, 0.0, 0.5])
        plane_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Sphere first
        contact1 = narrow_phase.check_collision(
            sphere, sphere_pos, quat,
            plane, plane_pos, quat
        )

        # Plane first
        contact2 = narrow_phase.check_collision(
            plane, plane_pos, quat,
            sphere, sphere_pos, quat
        )

        # Penetration should be the same
        np.testing.assert_allclose(contact1.penetration, contact2.penetration)

    def test_commutative_sphere_box(self, narrow_phase):
        """Test sphere-box collision is order-independent for penetration."""
        sphere = Geom(GeomType.SPHERE, np.array([1.0]))
        box = Geom(GeomType.BOX, np.array([1.0, 1.0, 1.0]))

        sphere_pos = np.array([1.5, 0.0, 0.0])
        box_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact1 = narrow_phase.check_collision(
            sphere, sphere_pos, quat,
            box, box_pos, quat
        )

        contact2 = narrow_phase.check_collision(
            box, box_pos, quat,
            sphere, sphere_pos, quat
        )

        np.testing.assert_allclose(contact1.penetration, contact2.penetration)


class TestMultipleContacts:
    """Tests for scenarios with multiple contacts."""

    def test_box_on_plane_single_contact(self, narrow_phase):
        """Test box tilted on plane has single deepest contact."""
        box = Geom(GeomType.BOX, np.array([0.5, 0.5, 0.5]))
        plane = Geom(GeomType.PLANE, np.array([10.0, 10.0, 0.1]))

        box_pos = np.array([0.0, 0.0, 0.3])
        plane_pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            box, box_pos, quat,
            plane, plane_pos, quat
        )

        # Returns single deepest contact
        assert contact is not None


class TestGJKEPA:
    """Tests for GJK/EPA general convex collision (placeholder)."""

    def test_gjk_epa_returns_none(self, narrow_phase):
        """Test GJK/EPA currently returns None (placeholder)."""
        mesh1 = Geom(GeomType.MESH, np.array([1.0]))
        mesh2 = Geom(GeomType.MESH, np.array([1.0]))

        pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        contact = narrow_phase.check_collision(
            mesh1, pos, quat,
            mesh2, pos, quat
        )

        # Currently a placeholder that returns None
        assert contact is None
