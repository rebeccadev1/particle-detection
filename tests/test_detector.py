"""Tests for per-tile particle detection on structured synthetic surfaces."""

from __future__ import annotations

import numpy as np

from src.detection.detector import detect_particles
from src.preprocessing.corrections import apply_corrections
from tests.fixtures.synthetic import (
    detector_test_config,
    make_dark_field_flake,
    make_flake_on_step,
    make_pad_tile,
    make_structured_tile,
    make_two_region_tile,
)


def _nearest_distance(detected: list[tuple[float, float]], truth: tuple[float, float]) -> float:
    return min(
        float(np.hypot(y - truth[0], x - truth[1])) for y, x in detected
    )


def test_equivalent_diameter_matches_disk_area() -> None:
    from src.detection.detector import blob_equivalent_diameter

    image = np.zeros((81, 81), dtype=np.float32)
    yy, xx = np.ogrid[:81, :81]
    disk = (yy - 40) ** 2 + (xx - 40) ** 2 <= 10.0**2
    image[disk] = 1.0
    diameter = blob_equivalent_diameter(image, 40.0, 40.0, radius=10.0)
    expected = 2.0 * np.sqrt(float(np.count_nonzero(disk)) / np.pi)
    assert abs(diameter - expected) < 0.05
    assert abs(diameter - 20.0) < 1.0


def test_equivalent_diameter_uses_photo_not_floor() -> None:
    """A bright disk on a non-zero photo floor still measures ~disk width."""
    from src.detection.detector import blob_equivalent_diameter

    image = np.full((81, 81), 0.4, dtype=np.float32)
    yy, xx = np.ogrid[:81, :81]
    disk = (yy - 40) ** 2 + (xx - 40) ** 2 <= 10.0**2
    image[disk] = 1.0
    diameter = blob_equivalent_diameter(image, 40.0, 40.0, radius=10.0)
    expected = 2.0 * np.sqrt(float(np.count_nonzero(disk)) / np.pi)
    assert abs(diameter - expected) < 1.0
    assert abs(diameter - 20.0) < 1.5


def test_equivalent_diameter_separates_two_scales() -> None:
    small = make_structured_tile(particles=[(40.0, 40.0, 2.0)])
    large = make_structured_tile(particles=[(40.0, 40.0, 6.0)])
    config = detector_test_config(
        detection={
            "method": "tophat",
            "blob_min_sigma": 1.2,
            "blob_max_sigma": 8.0,
            "min_size_nm": 4.0,
            "max_size_nm": 40.0,
            "structure_neighbor_px": 0.0,
        }
    )
    found_small = detect_particles(apply_corrections(small, config), config)
    found_large = detect_particles(apply_corrections(large, config), config)
    assert found_small and found_large
    near_small = min(found_small, key=lambda c: abs(c.y_local - 40.0) + abs(c.x_local - 40.0))
    near_large = min(found_large, key=lambda c: abs(c.y_local - 40.0) + abs(c.x_local - 40.0))
    assert near_large.size > near_small.size * 1.3


def test_detector_recovers_known_blob_positions() -> None:
    particles = [(40.0, 36.0, 3.8), (90.0, 88.0, 4.2)]
    image = make_structured_tile(particles=particles)
    config = detector_test_config()
    corrected = apply_corrections(image, config)
    found = detect_particles(corrected, config)

    assert len(found) >= 2
    coords = [(c.y_local, c.x_local) for c in found]
    for y, x, _sigma in particles:
        assert _nearest_distance(coords, (y, x)) < 3.0

    for candidate in found:
        assert 6.0 <= candidate.size <= 40.0
        assert candidate.confidence > 0
        assert 0.0 <= candidate.circularity <= 1.0
        assert candidate.local_peak_snr >= 0.0
        assert candidate.edge_distance_px >= 0.0


def test_fft_suppression_finds_blob_not_lattice() -> None:
    particles = [(64.0, 48.0, 4.0)]
    image = make_structured_tile(shape=(128, 128), lattice_period=16, particles=particles)
    config = detector_test_config(detection={"method": "fft", "blob_threshold": 0.08})
    corrected = apply_corrections(image, config)
    found = detect_particles(corrected, config)
    coords = [(c.y_local, c.x_local) for c in found]
    assert coords, "expected at least one candidate after FFT suppression"
    assert _nearest_distance(coords, (64.0, 48.0)) < 4.0


def test_fft_mask_reuse_matches_per_tile_mask() -> None:
    from src.detection.detector import build_fft_notch_mask, suppress_periodic_fft

    image = make_structured_tile(shape=(128, 128), lattice_period=16, particles=[(64.0, 48.0, 4.0)])
    mask = build_fft_notch_mask(image, peak_threshold=0.3, notch_radius=2)
    a = suppress_periodic_fft(image, 0.3, 2, mask=mask)
    b = suppress_periodic_fft(image, 0.3, 2, mask=None)
    assert a.dtype == np.float32
    assert a.shape == image.shape
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_shared_fft_mask_not_built_for_per_tile() -> None:
    from pathlib import Path

    from src.pipeline.runner import _prepare_fft_mask

    config = detector_test_config(detection={"method": "fft", "fft_mask": "per_tile"})
    assert _prepare_fft_mask([Path("unused.tif")], config) is None


def test_apply_corrections_stays_float32() -> None:
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    config = detector_test_config(
        preprocessing={
            "denoise": True,
            "flatten_illumination": True,
            "contrast_stretch": True,
            "denoise_sigma": 0.8,
            "flatten_sigma": 8.0,
        }
    )
    out = apply_corrections(image, config)
    assert out.dtype == np.float32
    assert out.shape == image.shape
    assert float(out.min()) >= -1e-5
    assert float(out.max()) <= 1.0 + 1e-5


def test_two_region_tile_keeps_blob_rejects_step() -> None:
    particle = (64.0, 48.0, 4.0)
    step_x = 96
    image = make_two_region_tile(step_x=step_x, particle=particle)
    config = detector_test_config(
        detection={
            "method": "tophat",
            "blob_threshold": 0.03,
            "local_snr_sigma": 20.0,
            "edge_soften_sigma": 6.0,
            "edge_soften_strength": 2.0,
            "edge_exclude_px": 20.0,
            "min_circularity": 0.35,
            "structure_neighbor_px": 0.0,
        }
    )
    corrected = apply_corrections(image, config)
    found = detect_particles(corrected, config)
    coords = [(c.y_local, c.x_local) for c in found]
    assert coords, "expected the dark-field blob"
    assert _nearest_distance(coords, (particle[0], particle[1])) < 4.0
    for _y, x in coords:
        assert abs(x - step_x) > 20.0, "region-border step should not be a particle"


def test_clustered_border_beads_are_dropped() -> None:
    from src.detection.detector import ParticleCandidate, reject_clustered_candidates

    chain = [
        ParticleCandidate(y_local=10.0, x_local=float(20 + 12 * i), size=10.0, confidence=0.9)
        for i in range(8)
    ]
    isolated = ParticleCandidate(y_local=80.0, x_local=80.0, size=10.0, confidence=0.8)
    kept = reject_clustered_candidates(chain + [isolated], neighbor_px=28.0, min_neighbors=2)
    assert len(kept) == 1
    assert kept[0].y_local == 80.0


def test_cluster_keeps_large_flake_among_small_neighbors() -> None:
    from src.detection.detector import ParticleCandidate, reject_clustered_candidates

    beads = [
        ParticleCandidate(
            y_local=10.0,
            x_local=float(20 + 12 * i),
            size=10.0,
            confidence=0.9,
            local_peak_snr=4.0,
        )
        for i in range(6)
    ]
    flake = ParticleCandidate(
        y_local=10.0, x_local=44.0, size=32.0, confidence=1.0, local_peak_snr=12.0
    )
    kept = reject_clustered_candidates(beads + [flake], neighbor_px=28.0, min_neighbors=2)
    assert any(abs(c.x_local - 44.0) < 0.1 and c.size > 20 for c in kept)
    assert all(c.size > 20 for c in kept)


def test_cluster_keeps_modestly_larger_flake() -> None:
    """~20 µm flake in a 13 µm lattice is ~1.3× median, not 1.6×."""
    from src.detection.detector import ParticleCandidate, reject_clustered_candidates

    beads = [
        ParticleCandidate(y_local=10.0, x_local=float(20 + 12 * i), size=13.0, confidence=0.9)
        for i in range(6)
    ]
    flake = ParticleCandidate(y_local=10.0, x_local=44.0, size=17.3, confidence=1.0)
    kept = reject_clustered_candidates(beads + [flake], neighbor_px=28.0, min_neighbors=2)
    assert any(abs(c.x_local - 44.0) < 0.1 and c.size > 16 for c in kept)
    assert not any(abs(c.size - 13.0) < 0.1 for c in kept)


def test_structure_filters_can_be_deferred() -> None:
    from src.detection.detector import apply_structure_filters, detect_particles

    image = make_structured_tile(particles=[(40.0, 42.0, 3.0)])
    config = detector_test_config(
        detection={"structure_neighbor_px": 20.0, "structure_min_neighbors": 2}
    )
    corrected = apply_corrections(image, config)
    gated = detect_particles(corrected, config, structure_filters=False)
    filtered = detect_particles(corrected, config, structure_filters=True)
    assert len(gated) >= len(filtered)
    restored = apply_structure_filters(gated, config)
    assert len(restored) == len(filtered)


def test_axis_aligned_box_frame_is_dropped() -> None:
    from src.detection.detector import ParticleCandidate, reject_axis_aligned_chains

    frame = [
        ParticleCandidate(y_local=20.0, x_local=float(30 + 20 * i), size=10.0, confidence=0.9)
        for i in range(6)
    ]
    isolated = ParticleCandidate(y_local=90.0, x_local=90.0, size=10.0, confidence=0.8)
    kept = reject_axis_aligned_chains(frame + [isolated], bin_px=10.0, min_run=3, min_span=40.0)
    assert len(kept) == 1
    assert kept[0].x_local == 90.0


def test_axis_chain_keeps_large_outlier_in_a_row() -> None:
    from src.detection.detector import ParticleCandidate, reject_axis_aligned_chains

    frame = [
        ParticleCandidate(y_local=20.0, x_local=float(30 + 20 * i), size=10.0, confidence=0.9)
        for i in range(6)
    ]
    flake = ParticleCandidate(y_local=20.0, x_local=70.0, size=36.0, confidence=1.0)
    kept = reject_axis_aligned_chains(frame + [flake], bin_px=10.0, min_run=3, min_span=40.0)
    assert any(c.size >= 30 for c in kept)
    assert not any(c.size == 10.0 for c in kept)


def test_axis_chain_ignores_sparse_same_row() -> None:
    """Two distant flakes in the same 10 px band are not a box frame."""
    from src.detection.detector import ParticleCandidate, reject_axis_aligned_chains

    sparse = [
        ParticleCandidate(y_local=20.0, x_local=40.0, size=17.3, confidence=1.0),
        ParticleCandidate(y_local=20.0, x_local=80.0, size=11.7, confidence=0.9),
        ParticleCandidate(y_local=20.0, x_local=890.0, size=17.5, confidence=1.0),
    ]
    kept = reject_axis_aligned_chains(sparse, bin_px=10.0, min_run=3, min_span=48.0)
    assert len(kept) == 3


def test_fft_without_lattice_falls_back_to_tophat() -> None:
    from src.detection.detector import _background_residual

    yy, xx = np.mgrid[0:64, 0:64]
    image = np.full((64, 64), 0.2, dtype=np.float32)
    image += 0.8 * np.exp(-((yy - 32.0) ** 2 + (xx - 32.0) ** 2) / (2.0 * 2.5**2))
    config = detector_test_config(detection={"method": "fft"})
    _residual, used_tophat = _background_residual(image, config, fft_mask=None)
    assert used_tophat is True
    found = detect_particles(image, config)
    coords = [(c.y_local, c.x_local) for c in found]
    assert coords
    assert _nearest_distance(coords, (32.0, 32.0)) < 4.0


def test_coarse_edge_distance_is_to_thin_ridge() -> None:
    from src.detection.detector import coarse_edge_distance

    image = make_two_region_tile(step_x=96, particle=(64.0, 48.0, 4.0))
    dist = coarse_edge_distance(image, sigma=6.0, min_length=40.0)
    assert float(dist[64, 96]) < 4.0
    assert float(dist[64, 48]) > 20.0
    assert float(dist[64, 80]) > 8.0


def _large_debris_config(**detection: object) -> dict:
    """Production-like 15–100 µm window (pixel_size_nm = 1 → sizes in px)."""
    params = {
        "method": "tophat",
        "tophat_radius": 50,
        "blob_min_sigma": 5.5,
        "blob_max_sigma": 36.0,
        "min_size_nm": 15.0,
        "max_size_nm": 100.0,
        "blob_threshold": 0.08,
        "min_prominence": 0.30,
        "edge_soften_sigma": 12.0,
        "edge_soften_strength": 2.0,
        "edge_exclude_px": 48.0,
        "min_circularity": 0.30,
        "structure_neighbor_px": 48.0,
    }
    params.update(detection)
    return detector_test_config(detection=params)


def test_dark_residual_peak_is_not_a_particle() -> None:
    from src.detection.detector import blob_is_local_peak

    image = np.full((64, 64), 0.1, dtype=np.float32)
    image[20:28, 20:28] = 0.9
    assert blob_is_local_peak(image, 24.0, 24.0, 4.0, bright=True)
    assert not blob_is_local_peak(image, 50.0, 50.0, 4.0, bright=True)


def test_large_dark_field_flake_is_kept() -> None:
    center = (140.0, 160.0)
    image = make_dark_field_flake(center=center, axes=(36.0, 32.0))
    config = _large_debris_config(structure_neighbor_px=0.0)
    corrected = apply_corrections(image, config)
    found = detect_particles(corrected, config)
    coords = [(c.y_local, c.x_local) for c in found]
    assert coords, "expected the large dark-field flake"
    assert _nearest_distance(coords, center) < 12.0
    assert any(c.size >= 20.0 for c in found)


def test_interior_blob_on_pad_is_kept() -> None:
    particle = (110.0, 120.0, 6.0)
    pad = (40, 40, 180, 200)
    image = make_pad_tile(pad=pad, particle=particle)
    config = detector_test_config(
        detection={
            "method": "tophat",
            "tophat_radius": 17,
            "blob_min_sigma": 5.5,
            "blob_max_sigma": 12.0,
            "min_size_nm": 15.0,
            "max_size_nm": 32.0,
            "blob_threshold": 0.08,
            "min_prominence": 0.20,
            "edge_soften_sigma": 12.0,
            "edge_soften_strength": 2.0,
            "edge_exclude_px": 48.0,
            "min_circularity": 0.35,
            "structure_neighbor_px": 0.0,
        }
    )
    corrected = apply_corrections(image, config)
    found = detect_particles(corrected, config)
    coords = [(c.y_local, c.x_local) for c in found]
    assert coords, "expected the compact blob inside the pad"
    assert _nearest_distance(coords, (particle[0], particle[1])) < 6.0


def test_image_frame_is_not_a_particle() -> None:
    """A pad that runs off the top of the tile must not fire at y = 0."""
    image = make_pad_tile(shape=(160, 240), pad=(0, 60, 100, 180))
    config = detector_test_config(
        detection={
            "method": "tophat",
            "tophat_radius": 17,
            "blob_min_sigma": 5.5,
            "blob_max_sigma": 12.0,
            "min_size_nm": 15.0,
            "max_size_nm": 32.0,
            "blob_threshold": 0.08,
            "min_prominence": 0.30,
            "edge_soften_sigma": 12.0,
            "edge_soften_strength": 2.0,
            "edge_exclude_px": 48.0,
            "min_circularity": 0.30,
            "structure_neighbor_px": 48.0,
        }
    )
    corrected = apply_corrections(image, config)
    found = detect_particles(corrected, config)
    for candidate in found:
        assert candidate.y_local >= 48.0, "tile-frame edge should not be a particle"


def test_pad_corner_is_not_a_particle() -> None:
    """A bright pad corner looks like a blob after top-hat; drop it."""
    pad = (40, 40, 180, 200)
    image = make_pad_tile(pad=pad)
    corners = [
        (float(pad[0]), float(pad[1])),
        (float(pad[0]), float(pad[3] - 1)),
        (float(pad[2] - 1), float(pad[1])),
        (float(pad[2] - 1), float(pad[3] - 1)),
    ]
    for config in (
        detector_test_config(
            detection={
                "method": "tophat",
                "tophat_radius": 17,
                "blob_min_sigma": 5.5,
                "blob_max_sigma": 12.0,
                "min_size_nm": 15.0,
                "max_size_nm": 32.0,
                "blob_threshold": 0.08,
                "min_prominence": 0.30,
                "edge_soften_sigma": 12.0,
                "edge_soften_strength": 2.0,
                "edge_exclude_px": 48.0,
                "min_circularity": 0.30,
                "structure_neighbor_px": 48.0,
            }
        ),
        _large_debris_config(),
    ):
        corrected = apply_corrections(image, config)
        found = detect_particles(corrected, config)
        for candidate in found:
            for cy, cx in corners:
                dist = float(np.hypot(candidate.y_local - cy, candidate.x_local - cx))
                assert dist > 24.0, "pad corner should not be a particle"


def test_bright_blob_near_region_edge_is_kept() -> None:
    """A compact blob *outside* the exclude band is kept; the step itself is not."""
    particle = (64.0, 48.0, 2.2)
    image = make_two_region_tile(step_x=96, particle=particle)
    config = detector_test_config(
        detection={
            "method": "tophat",
            "blob_min_sigma": 1.2,
            "min_size_nm": 4.0,
            "blob_threshold": 0.05,
            "min_prominence": 0.20,
            "edge_soften_sigma": 6.0,
            "edge_soften_strength": 2.0,
            "edge_exclude_px": 24.0,
            "structure_neighbor_px": 0.0,
        }
    )
    corrected = apply_corrections(image, config)
    found = detect_particles(corrected, config)
    coords = [(c.y_local, c.x_local) for c in found]
    assert coords, "expected the compact bright blob away from the region border"
    assert _nearest_distance(coords, (particle[0], particle[1])) < 4.0
    for _y, x in coords:
        assert abs(x - 96) > 20.0, "region-border step should not be a particle"


def test_radial_means_peak_is_highest_in_inner_ring() -> None:
    from src.detection.detector import blob_radial_means

    image = np.zeros((81, 81), dtype=np.float32)
    yy, xx = np.ogrid[:81, :81]
    image += np.exp(-((yy - 40.0) ** 2 + (xx - 40.0) ** 2) / (2.0 * 4.0**2))
    inner, mid, outer = blob_radial_means(image, 40.0, 40.0, 8.0)
    assert inner > mid > outer


def _island_test_config(**detection: object) -> dict:
    """pixel_size_nm = 1 so nm thresholds are pixels; island logic is on."""
    params = {
        "method": "tophat",
        "tophat_radius": 50,
        "blob_min_sigma": 5.5,
        "blob_max_sigma": 36.0,
        "min_size_nm": 15.0,
        "max_size_nm": 100.0,
        "blob_threshold": 0.08,
        "min_prominence": 0.30,
        "edge_soften_sigma": 12.0,
        "edge_soften_strength": 2.0,
        "edge_exclude_px": 48.0,
        "min_circularity": 0.0,
        "structure_neighbor_px": 0.0,
        "island_min_nm": 20.0,
        "island_large_nm": 40.0,
        "island_max_aspect": 3.0,
    }
    params.update(detection)
    return detector_test_config(detection=params)


def test_large_flake_on_region_edge_is_kept() -> None:
    """≥40 px compact island on a single step survives edge softening."""
    center = (110.0, 140.0)
    image = make_flake_on_step(center=center, axes=(28.0, 26.0), step_x=140)
    cfg = _island_test_config()
    found = detect_particles(apply_corrections(image, cfg), cfg)
    coords = [(c.y_local, c.x_local) for c in found]
    assert coords, "expected the large flake on the region edge"
    assert _nearest_distance(coords, center) < 16.0
    assert any(c.size >= 20.0 for c in found)


def test_compact_flake_on_single_edge_is_kept() -> None:
    """20–40 px island on one edge is kept; not an L-junction."""
    center = (110.0, 140.0)
    image = make_flake_on_step(center=center, axes=(18.0, 16.0), step_x=140)
    cfg = _island_test_config()
    found = detect_particles(apply_corrections(image, cfg), cfg)
    coords = [(c.y_local, c.x_local) for c in found]
    assert coords, "expected the 20 µm-scale flake on a single edge"
    assert _nearest_distance(coords, center) < 16.0


def test_pad_corner_stays_dropped_with_island_protect() -> None:
    """L-junction residual islands must not be protected."""
    pad = (40, 40, 180, 200)
    image = make_pad_tile(pad=pad)
    corners = [
        (float(pad[0]), float(pad[1])),
        (float(pad[0]), float(pad[3] - 1)),
        (float(pad[2] - 1), float(pad[1])),
        (float(pad[2] - 1), float(pad[3] - 1)),
    ]
    config = _island_test_config(structure_neighbor_px=48.0, min_circularity=0.30)
    found = detect_particles(apply_corrections(image, config), config)
    for candidate in found:
        for cy, cx in corners:
            dist = float(np.hypot(candidate.y_local - cy, candidate.x_local - cx))
            assert dist > 24.0, "pad corner should not be a particle"


def test_recall_mode_relaxes_gates() -> None:
    from src.config import with_recall_profile

    config = detector_test_config(
        detection={
            "recall_mode": True,
            "edge_exclude_px": 48.0,
            "min_confidence": 0.95,
            "min_circularity": 0.30,
            "structure_min_neighbors": 2,
            "structure_line_bin_px": 10.0,
        }
    )
    relaxed = with_recall_profile(config)
    assert relaxed is not config
    det = relaxed["detection"]
    assert det["edge_exclude_px"] == 12.0
    assert det["min_confidence"] == 0.40
    assert det["min_circularity"] == 0.0
    assert det["structure_min_neighbors"] == 2
    assert det["structure_line_bin_px"] == 10.0
    assert config["detection"]["edge_exclude_px"] == 48.0


def test_trace_locations_marks_kept_and_empty() -> None:
    from src.detection.detector import detect_particles, trace_locations

    image = make_structured_tile(particles=[(40.0, 42.0, 3.0)])
    config = detector_test_config()
    corrected = apply_corrections(image, config)
    hits = detect_particles(corrected, config)
    assert hits
    kept = trace_locations(corrected, config, [(hits[0].y_local, hits[0].x_local)])
    assert kept[0]["reason"] == "already_kept"
    empty = trace_locations(corrected, config, [(5.0, 5.0)])
    assert empty[0]["reason"] != "already_kept"

