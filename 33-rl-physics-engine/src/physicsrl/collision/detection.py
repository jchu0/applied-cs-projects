"""Collision detection with broad and narrow phases."""

import logging
import numpy as np
from typing import List, Tuple, Optional

from ..core.bodies import (
    Model, State, Geom, Contact, GeomType,
    quat_rotate, quat_mul
)

logger = logging.getLogger(__name__)


class BroadPhase:
    """Broad-phase collision detection using AABB tree."""

    def __init__(self):
        self.aabb_tree = None

    def find_overlapping_pairs(self, model: Model, state: State) -> List[Tuple[int, int]]:
        """
        Find all potentially overlapping geometry pairs.

        Returns:
            List of (geom1_idx, geom2_idx) pairs to check
        """
        pairs = []

        # Build AABB list
        geom_aabbs = []
        geom_to_body = []

        geom_idx = 0
        for body_idx, body in enumerate(model.bodies):
            for geom in body.geoms:
                aabb_min, aabb_max = geom.get_aabb(
                    state.xpos[body_idx], state.xquat[body_idx]
                )
                geom_aabbs.append((aabb_min, aabb_max))
                geom_to_body.append(body_idx)
                geom_idx += 1

        # Check all pairs (O(n^2) - can use BVH for efficiency)
        n_geoms = len(geom_aabbs)
        for i in range(n_geoms):
            for j in range(i + 1, n_geoms):
                # Skip same body
                if geom_to_body[i] == geom_to_body[j]:
                    continue

                # Check AABB overlap
                if self._aabb_overlap(geom_aabbs[i], geom_aabbs[j]):
                    pairs.append((i, j))

        return pairs

    @staticmethod
    def _aabb_overlap(aabb1: Tuple, aabb2: Tuple) -> bool:
        """Check if two AABBs overlap."""
        min1, max1 = aabb1
        min2, max2 = aabb2

        return (min1[0] <= max2[0] and max1[0] >= min2[0] and
                min1[1] <= max2[1] and max1[1] >= min2[1] and
                min1[2] <= max2[2] and max1[2] >= min2[2])


class NarrowPhase:
    """Narrow-phase collision detection for exact contact computation."""

    def __init__(self):
        # Track which unsupported geom-type pairs we've already warned about so
        # each pair type is reported once rather than every simulation step.
        self._warned_unsupported = set()

    def check_collision(self, geom1: Geom, pos1: np.ndarray, quat1: np.ndarray,
                        geom2: Geom, pos2: np.ndarray, quat2: np.ndarray) -> Optional[Contact]:
        """
        Check collision between two geometries.

        Returns:
            Contact if collision, None otherwise
        """
        # Dispatch based on geometry types
        type_pair = (geom1.geom_type, geom2.geom_type)

        if type_pair == (GeomType.SPHERE, GeomType.SPHERE):
            return self._sphere_sphere(geom1, pos1, geom2, pos2)

        elif type_pair == (GeomType.SPHERE, GeomType.PLANE):
            return self._sphere_plane(geom1, pos1, geom2, pos2, quat2)

        elif type_pair == (GeomType.PLANE, GeomType.SPHERE):
            contact = self._sphere_plane(geom2, pos2, geom1, pos1, quat1)
            if contact:
                contact.normal = -contact.normal
            return contact

        elif type_pair == (GeomType.SPHERE, GeomType.BOX):
            return self._sphere_box(geom1, pos1, geom2, pos2, quat2)

        elif type_pair == (GeomType.BOX, GeomType.SPHERE):
            contact = self._sphere_box(geom2, pos2, geom1, pos1, quat1)
            if contact:
                contact.normal = -contact.normal
            return contact

        elif type_pair == (GeomType.CAPSULE, GeomType.PLANE):
            return self._capsule_plane(geom1, pos1, quat1, geom2, pos2, quat2)

        elif type_pair == (GeomType.PLANE, GeomType.CAPSULE):
            contact = self._capsule_plane(geom2, pos2, quat2, geom1, pos1, quat1)
            if contact:
                contact.normal = -contact.normal
            return contact

        elif type_pair == (GeomType.CAPSULE, GeomType.CAPSULE):
            return self._capsule_capsule(geom1, pos1, quat1, geom2, pos2, quat2)

        elif type_pair == (GeomType.BOX, GeomType.PLANE):
            return self._box_plane(geom1, pos1, quat1, geom2, pos2, quat2)

        elif type_pair == (GeomType.PLANE, GeomType.BOX):
            contact = self._box_plane(geom2, pos2, quat2, geom1, pos1, quat1)
            if contact:
                contact.normal = -contact.normal
            return contact

        else:
            # Unsupported geom-type pair (e.g. box-box, capsule-box, any mesh
            # pair). Full GJK/EPA general-convex collision is not implemented,
            # so no contact is generated for these pairs. Rather than silently
            # returning None (a physics-correctness surprise), warn once per
            # unsupported pair type so the gap is visible in logs.
            self._warn_unsupported_pair(geom1.geom_type, geom2.geom_type)
            return self._gjk_epa(geom1, pos1, quat1, geom2, pos2, quat2)

    def _warn_unsupported_pair(self, type1: GeomType, type2: GeomType) -> None:
        """Log a one-time warning for an unsupported collision pair type.

        The unsupported-pair set is symmetric, so ``(box, capsule)`` and
        ``(capsule, box)`` are treated as the same pair and warned about once.
        """
        key = frozenset((type1, type2)) if type1 != type2 else (type1,)
        if key in self._warned_unsupported:
            return
        self._warned_unsupported.add(key)
        logger.warning(
            "Collision between geom types %s and %s is not supported "
            "(general convex GJK/EPA is not implemented); no contact will be "
            "generated for this pair. See docs/BLUEPRINT.md for the list of "
            "supported pairs.",
            type1.value, type2.value,
        )

    def _sphere_sphere(self, geom1: Geom, pos1: np.ndarray,
                       geom2: Geom, pos2: np.ndarray) -> Optional[Contact]:
        """Sphere-sphere collision."""
        r1, r2 = geom1.size[0], geom2.size[0]
        d = pos2 - pos1
        dist = np.linalg.norm(d)

        if dist < r1 + r2:
            normal = d / (dist + 1e-8)
            penetration = r1 + r2 - dist
            contact_pos = pos1 + normal * (r1 - penetration / 2)

            return Contact(
                pos=contact_pos,
                normal=normal,
                penetration=penetration,
                geom1_idx=0, geom2_idx=0,
                body1_idx=0, body2_idx=0,
                friction=min(geom1.friction, geom2.friction),
                restitution=max(geom1.restitution, geom2.restitution)
            )
        return None

    def _sphere_plane(self, sphere: Geom, sphere_pos: np.ndarray,
                      plane: Geom, plane_pos: np.ndarray,
                      plane_quat: np.ndarray) -> Optional[Contact]:
        """Sphere-plane collision."""
        # Plane normal in world frame
        normal = quat_rotate(plane_quat, np.array([0.0, 0.0, 1.0]))

        # Distance from sphere center to plane
        d = np.dot(sphere_pos - plane_pos, normal)
        r = sphere.size[0]

        if d < r:
            contact_pos = sphere_pos - normal * d
            penetration = r - d

            return Contact(
                pos=contact_pos,
                normal=-normal,  # Point into sphere
                penetration=penetration,
                geom1_idx=0, geom2_idx=0,
                body1_idx=0, body2_idx=0,
                friction=min(sphere.friction, plane.friction),
                restitution=max(sphere.restitution, plane.restitution)
            )
        return None

    def _sphere_box(self, sphere: Geom, sphere_pos: np.ndarray,
                    box: Geom, box_pos: np.ndarray,
                    box_quat: np.ndarray) -> Optional[Contact]:
        """Sphere-box collision."""
        r = sphere.size[0]
        half = box.size

        # Transform sphere center to box local frame
        d = sphere_pos - box_pos
        q_inv = np.array([box_quat[0], -box_quat[1], -box_quat[2], -box_quat[3]])
        local_pos = quat_rotate(q_inv, d)

        # Find closest point on box
        closest = np.clip(local_pos, -half, half)

        # Distance to closest point
        diff = local_pos - closest
        dist = np.linalg.norm(diff)

        if dist < r:
            if dist > 1e-8:
                normal_local = diff / dist
            else:
                # Sphere center inside box
                face_dist = np.minimum(half - local_pos, local_pos + half)
                min_axis = np.argmin(face_dist)
                normal_local = np.zeros(3)
                normal_local[min_axis] = np.sign(local_pos[min_axis])

            normal = quat_rotate(box_quat, normal_local)
            penetration = r - dist
            contact_pos = sphere_pos - normal * (r - penetration / 2)

            return Contact(
                pos=contact_pos,
                normal=normal,
                penetration=penetration,
                geom1_idx=0, geom2_idx=0,
                body1_idx=0, body2_idx=0,
                friction=min(sphere.friction, box.friction),
                restitution=max(sphere.restitution, box.restitution)
            )
        return None

    def _capsule_plane(self, capsule: Geom, cap_pos: np.ndarray, cap_quat: np.ndarray,
                       plane: Geom, plane_pos: np.ndarray,
                       plane_quat: np.ndarray) -> Optional[Contact]:
        """Capsule-plane collision."""
        r = capsule.size[0]
        half_len = capsule.size[1]

        # Capsule endpoints
        axis = quat_rotate(cap_quat, np.array([0.0, 0.0, 1.0]))
        p1 = cap_pos - axis * half_len
        p2 = cap_pos + axis * half_len

        # Plane normal
        normal = quat_rotate(plane_quat, np.array([0.0, 0.0, 1.0]))

        # Check both endpoints
        contacts = []
        for p in [p1, p2]:
            d = np.dot(p - plane_pos, normal)
            if d < r:
                contact_pos = p - normal * d
                contacts.append(Contact(
                    pos=contact_pos,
                    normal=-normal,
                    penetration=r - d,
                    geom1_idx=0, geom2_idx=0,
                    body1_idx=0, body2_idx=0,
                    friction=min(capsule.friction, plane.friction),
                    restitution=max(capsule.restitution, plane.restitution)
                ))

        # Return deepest contact
        if contacts:
            return max(contacts, key=lambda c: c.penetration)
        return None

    def _capsule_capsule(self, cap1: Geom, pos1: np.ndarray, quat1: np.ndarray,
                         cap2: Geom, pos2: np.ndarray, quat2: np.ndarray) -> Optional[Contact]:
        """Capsule-capsule collision."""
        r1, h1 = cap1.size[0], cap1.size[1]
        r2, h2 = cap2.size[0], cap2.size[1]

        axis1 = quat_rotate(quat1, np.array([0.0, 0.0, 1.0]))
        axis2 = quat_rotate(quat2, np.array([0.0, 0.0, 1.0]))

        a1 = pos1 - axis1 * h1
        b1 = pos1 + axis1 * h1
        a2 = pos2 - axis2 * h2
        b2 = pos2 + axis2 * h2

        # Closest points on line segments
        c1, c2 = self._closest_points_segments(a1, b1, a2, b2)

        dist = np.linalg.norm(c2 - c1)
        if dist < r1 + r2:
            normal = (c2 - c1) / (dist + 1e-8)
            penetration = r1 + r2 - dist
            contact_pos = c1 + normal * (r1 - penetration / 2)

            return Contact(
                pos=contact_pos,
                normal=normal,
                penetration=penetration,
                geom1_idx=0, geom2_idx=0,
                body1_idx=0, body2_idx=0,
                friction=min(cap1.friction, cap2.friction),
                restitution=max(cap1.restitution, cap2.restitution)
            )
        return None

    def _box_plane(self, box: Geom, box_pos: np.ndarray, box_quat: np.ndarray,
                   plane: Geom, plane_pos: np.ndarray, plane_quat: np.ndarray) -> Optional[Contact]:
        """Box-plane collision."""
        half = box.size

        # Get plane normal
        normal = quat_rotate(plane_quat, np.array([0.0, 0.0, 1.0]))

        # Box corners in local frame
        corners = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], dtype=np.float64) * half

        # Transform to world frame
        world_corners = np.array([
            box_pos + quat_rotate(box_quat, c) for c in corners
        ])

        # Find deepest penetrating corner
        min_dist = np.inf
        contact_corner = None

        for corner in world_corners:
            d = np.dot(corner - plane_pos, normal)
            if d < min_dist:
                min_dist = d
                contact_corner = corner

        if min_dist < 0:
            return Contact(
                pos=contact_corner,
                normal=-normal,
                penetration=-min_dist,
                geom1_idx=0, geom2_idx=0,
                body1_idx=0, body2_idx=0,
                friction=min(box.friction, plane.friction),
                restitution=max(box.restitution, plane.restitution)
            )
        return None

    @staticmethod
    def _closest_points_segments(a1, b1, a2, b2):
        """Find closest points between two line segments."""
        d1 = b1 - a1
        d2 = b2 - a2
        r = a1 - a2

        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d1, r)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)

        denom = a * e - b * b
        if abs(denom) < 1e-8:
            # Parallel segments
            s = 0
            t = f / e if e > 1e-8 else 0
        else:
            s = (b * f - c * e) / denom
            t = (a * f - b * c) / denom

        # Clamp to segment
        s = np.clip(s, 0, 1)
        t = np.clip(t, 0, 1)

        c1 = a1 + s * d1
        c2 = a2 + t * d2

        return c1, c2

    def _gjk_epa(self, geom1, pos1, quat1, geom2, pos2, quat2) -> Optional[Contact]:
        """GJK + EPA for general convex collision."""
        # Placeholder - implement GJK/EPA for general convex shapes
        return None


class CollisionSystem:
    """Complete collision detection system."""

    def __init__(self, model: Model):
        self.model = model
        self.broad_phase = BroadPhase()
        self.narrow_phase = NarrowPhase()

    def detect_contacts(self, state: State) -> List[Contact]:
        """Detect all contacts in current state."""
        contacts = []

        # Broad phase
        pairs = self.broad_phase.find_overlapping_pairs(self.model, state)

        # Build geometry map
        geom_map = []  # (body_idx, geom_idx within body)
        for body_idx, body in enumerate(self.model.bodies):
            for local_idx, geom in enumerate(body.geoms):
                geom_map.append((body_idx, local_idx))

        # Narrow phase
        for i, j in pairs:
            body1_idx, local1 = geom_map[i]
            body2_idx, local2 = geom_map[j]

            geom1 = self.model.bodies[body1_idx].geoms[local1]
            geom2 = self.model.bodies[body2_idx].geoms[local2]

            contact = self.narrow_phase.check_collision(
                geom1, state.xpos[body1_idx], state.xquat[body1_idx],
                geom2, state.xpos[body2_idx], state.xquat[body2_idx]
            )

            if contact:
                contact.geom1_idx = i
                contact.geom2_idx = j
                contact.body1_idx = body1_idx
                contact.body2_idx = body2_idx
                contacts.append(contact)

        return contacts
