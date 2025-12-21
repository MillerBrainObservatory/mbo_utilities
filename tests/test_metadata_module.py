"""
Tests for the metadata module.

Tests the centralized metadata parameter handling including:
- get_param with aliases
- VoxelSize extraction
- ScanImage detection functions
"""

import pytest
from mbo_utilities.metadata import (
    MetadataParameter,
    VoxelSize,
    METADATA_PARAMS,
    ALIAS_MAP,
    get_canonical_name,
    get_param,
    get_voxel_size,
    normalize_resolution,
    normalize_metadata,
    detect_stack_type,
    is_lbm_stack,
    is_piezo_stack,
    get_lbm_ai_sources,
    get_num_color_channels,
    get_num_zplanes,
    get_frames_per_slice,
    get_log_average_factor,
    get_z_step_size,
    compute_num_timepoints,
    get_stack_info,
)


class TestMetadataParameter:
    """Test MetadataParameter dataclass."""

    def test_parameter_creation(self):
        """Create a basic MetadataParameter."""
        param = MetadataParameter(
            canonical="test",
            aliases=("t", "tst"),
            dtype=float,
            unit="Hz",
            default=1.0,
            description="Test param",
        )
        assert param.canonical == "test"
        assert "t" in param.aliases
        assert param.dtype == float
        assert param.default == 1.0

    def test_metadata_params_registry(self):
        """Check METADATA_PARAMS has expected entries."""
        assert "dx" in METADATA_PARAMS
        assert "fs" in METADATA_PARAMS
        assert "num_zplanes" in METADATA_PARAMS

        # check dx parameter
        dx = METADATA_PARAMS["dx"]
        assert dx.unit == "µm"
        assert "PhysicalSizeX" in dx.aliases

        # check num_zplanes has num_planes and nplanes as aliases
        num_zplanes = METADATA_PARAMS["num_zplanes"]
        assert "num_planes" in num_zplanes.aliases
        assert "nplanes" in num_zplanes.aliases


class TestAliasMap:
    """Test alias resolution."""

    def test_canonical_name_lookup(self):
        """get_canonical_name should resolve aliases."""
        assert get_canonical_name("dx") == "dx"
        assert get_canonical_name("PhysicalSizeX") == "dx"
        assert get_canonical_name("pixel_size_x") == "dx"
        assert get_canonical_name("frame_rate") == "fs"
        assert get_canonical_name("fps") == "fs"

    def test_unknown_name_returns_none(self):
        """Unknown names return None."""
        assert get_canonical_name("unknown_param") is None

    def test_case_insensitive(self):
        """Lookups should be case-insensitive."""
        assert get_canonical_name("DX") == "dx"
        assert get_canonical_name("Fs") == "fs"


class TestGetParam:
    """Test get_param function."""

    def test_canonical_key(self):
        """Get param by canonical key."""
        meta = {"dx": 0.5}
        assert get_param(meta, "dx") == 0.5

    def test_alias_key(self):
        """Get param by alias."""
        meta = {"pixel_size_x": 0.5}
        assert get_param(meta, "dx") == 0.5

    def test_another_alias(self):
        """Get param by another alias."""
        meta = {"PhysicalSizeX": 0.5}
        assert get_param(meta, "dx") == 0.5

    def test_frame_rate_aliases(self):
        """Test frame rate with various aliases."""
        assert get_param({"fs": 30.0}, "fs") == 30.0
        assert get_param({"frame_rate": 30.0}, "fs") == 30.0
        assert get_param({"fps": 30.0}, "fs") == 30.0

    def test_default_value(self):
        """Missing param returns default."""
        meta = {}
        assert get_param(meta, "dx", default=1.0) == 1.0

    def test_override_wins(self):
        """Override value takes precedence."""
        meta = {"dx": 0.5}
        assert get_param(meta, "dx", override=0.3) == 0.3

    def test_none_metadata(self):
        """Handle None metadata gracefully."""
        result = get_param(None, "dx", default=1.0)
        assert result == 1.0

    def test_shape_fallback_lx(self):
        """Lx can be inferred from shape."""
        meta = {}
        result = get_param(meta, "Lx", shape=(10, 128, 256))
        assert result == 256

    def test_shape_fallback_ly(self):
        """Ly can be inferred from shape."""
        meta = {}
        result = get_param(meta, "Ly", shape=(10, 128, 256))
        assert result == 128

    def test_pixel_resolution_tuple_dx(self):
        """dx can be extracted from pixel_resolution tuple."""
        meta = {"pixel_resolution": (0.5, 0.6)}
        assert get_param(meta, "dx") == 0.5

    def test_pixel_resolution_tuple_dy(self):
        """dy can be extracted from pixel_resolution tuple."""
        meta = {"pixel_resolution": (0.5, 0.6)}
        assert get_param(meta, "dy") == 0.6


class TestVoxelSize:
    """Test VoxelSize named tuple."""

    def test_creation(self):
        """Create VoxelSize."""
        vs = VoxelSize(0.5, 0.5, 5.0)
        assert vs.dx == 0.5
        assert vs.dy == 0.5
        assert vs.dz == 5.0

    def test_pixel_resolution_property(self):
        """pixel_resolution returns (dx, dy)."""
        vs = VoxelSize(0.5, 0.6, 5.0)
        assert vs.pixel_resolution == (0.5, 0.6)

    def test_voxel_size_property(self):
        """voxel_size returns (dx, dy, dz)."""
        vs = VoxelSize(0.5, 0.6, 5.0)
        assert vs.voxel_size == (0.5, 0.6, 5.0)

    def test_to_dict(self):
        """to_dict returns expected keys."""
        vs = VoxelSize(0.5, 0.5, 5.0)
        d = vs.to_dict()
        assert d["dx"] == 0.5
        assert d["dy"] == 0.5
        assert d["dz"] == 5.0
        assert d["pixel_resolution"] == (0.5, 0.5)

    def test_to_dict_includes_aliases(self):
        """to_dict includes standard aliases."""
        vs = VoxelSize(0.5, 0.5, 5.0)
        d = vs.to_dict(include_aliases=True)
        assert d["PhysicalSizeX"] == 0.5
        assert d["PhysicalSizeY"] == 0.5
        assert d["PhysicalSizeZ"] == 5.0
        assert d["z_step"] == 5.0


class TestGetVoxelSize:
    """Test get_voxel_size function."""

    def test_from_canonical_keys(self):
        """Extract from dx, dy, dz keys."""
        meta = {"dx": 0.5, "dy": 0.5, "dz": 5.0}
        vs = get_voxel_size(meta)
        assert vs.dx == 0.5
        assert vs.dz == 5.0

    def test_from_pixel_resolution(self):
        """Extract from pixel_resolution tuple."""
        meta = {"pixel_resolution": (0.5, 0.6)}
        vs = get_voxel_size(meta)
        assert vs.dx == 0.5
        assert vs.dy == 0.6

    def test_from_ome_keys(self):
        """Extract from OME format keys."""
        meta = {"PhysicalSizeX": 0.5, "PhysicalSizeY": 0.6, "PhysicalSizeZ": 5.0}
        vs = get_voxel_size(meta)
        assert vs.dx == 0.5
        assert vs.dy == 0.6
        assert vs.dz == 5.0

    def test_override_values(self):
        """User overrides take precedence."""
        meta = {"dx": 0.5}
        vs = get_voxel_size(meta, dx=0.3)
        assert vs.dx == 0.3

    def test_from_scanimage_nested(self):
        """Extract dz from ScanImage nested structure."""
        meta = {
            "si": {
                "hStackManager": {
                    "stackZStepSize": 5.0
                }
            }
        }
        vs = get_voxel_size(meta)
        assert vs.dz == 5.0

    def test_defaults_to_1(self):
        """Missing values default to 1.0 for non-LBM."""
        vs = get_voxel_size({})
        assert vs.dx == 1.0
        assert vs.dy == 1.0
        assert vs.dz == 1.0

    def test_lbm_no_default_dz(self):
        """LBM stacks should not get default dz - must be user-supplied."""
        # lbm_stack flag
        meta = {"lbm_stack": True}
        vs = get_voxel_size(meta)
        assert vs.dx == 1.0
        assert vs.dy == 1.0
        assert vs.dz is None

        # stack_type == "lbm"
        meta = {"stack_type": "lbm"}
        vs = get_voxel_size(meta)
        assert vs.dz is None

        # but user override still works
        vs = get_voxel_size(meta, dz=20.0)
        assert vs.dz == 20.0

    def test_lbm_ignores_scanimage_dz(self):
        """LBM stacks should not extract dz from ScanImage metadata."""
        # even if ScanImage has hStackManager.stackZStepSize, LBM should ignore it
        meta = {
            "lbm_stack": True,
            "si": {
                "hStackManager": {
                    "stackZStepSize": 5.0,
                    "actualStackZStepSize": 5.0
                }
            }
        }
        vs = get_voxel_size(meta)
        assert vs.dz is None  # should NOT be 5.0


class TestNormalizeResolution:
    """Test normalize_resolution function."""

    def test_adds_aliases(self):
        """normalize_resolution adds all standard aliases."""
        meta = {"dx": 0.5, "dy": 0.5, "dz": 5.0}
        normalize_resolution(meta)
        assert meta["PhysicalSizeX"] == 0.5
        assert meta["PhysicalSizeY"] == 0.5
        assert meta["PhysicalSizeZ"] == 5.0
        assert meta["z_step"] == 5.0


class TestScanImageDetection:
    """Test ScanImage-specific detection functions."""

    def test_detect_lbm_stack(self):
        """LBM stack detected by channelSave length > 2."""
        meta = {
            "si": {
                "hChannels": {
                    "channelSave": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                }
            }
        }
        assert detect_stack_type(meta) == "lbm"
        assert is_lbm_stack(meta) is True
        assert is_piezo_stack(meta) is False

    def test_detect_piezo_stack(self):
        """Piezo stack detected by hStackManager.enable."""
        meta = {
            "si": {
                "hStackManager": {"enable": True, "numSlices": 10},
                "hChannels": {"channelSave": 1}
            }
        }
        assert detect_stack_type(meta) == "piezo"
        assert is_piezo_stack(meta) is True
        assert is_lbm_stack(meta) is False

    def test_detect_single_plane(self):
        """Single plane when neither LBM nor piezo."""
        meta = {
            "si": {
                "hChannels": {"channelSave": 1},
                "hStackManager": {"enable": False}
            }
        }
        assert detect_stack_type(meta) == "single_plane"

    def test_empty_metadata(self):
        """Empty metadata defaults to single_plane."""
        assert detect_stack_type({}) == "single_plane"


class TestLbmColorChannels:
    """Test LBM color channel detection."""

    def test_single_color_channel(self):
        """Single AI source = 1 color channel."""
        meta = {
            "si": {
                "hChannels": {"channelSave": list(range(1, 15))},
                "hScan2D": {
                    "virtualChannelSettings__1": {"source": "AI0"},
                    "virtualChannelSettings__2": {"source": "AI0"},
                    "virtualChannelSettings__3": {"source": "AI0"},
                }
            }
        }
        sources = get_lbm_ai_sources(meta)
        assert "AI0" in sources
        assert len(sources) == 1
        assert get_num_color_channels(meta) == 1

    def test_dual_color_channel(self):
        """AI0 + AI1 = 2 color channels."""
        meta = {
            "si": {
                "hChannels": {"channelSave": list(range(1, 18))},
                "hScan2D": {
                    "virtualChannelSettings__1": {"source": "AI0"},
                    "virtualChannelSettings__2": {"source": "AI0"},
                    "virtualChannelSettings__15": {"source": "AI1"},
                    "virtualChannelSettings__16": {"source": "AI1"},
                }
            }
        }
        sources = get_lbm_ai_sources(meta)
        assert "AI0" in sources
        assert "AI1" in sources
        assert get_num_color_channels(meta) == 2


class TestPiezoStackParams:
    """Test piezo stack parameter extraction."""

    def test_get_num_zplanes_piezo(self):
        """numSlices from hStackManager."""
        meta = {
            "si": {
                "hStackManager": {"enable": True, "numSlices": 17},
                "hChannels": {"channelSave": 1}
            }
        }
        assert get_num_zplanes(meta) == 17

    def test_get_num_zplanes_lbm(self):
        """channelSave length for LBM."""
        meta = {
            "si": {
                "hChannels": {"channelSave": list(range(1, 15))}
            }
        }
        assert get_num_zplanes(meta) == 14

    def test_get_frames_per_slice(self):
        """framesPerSlice from hStackManager."""
        meta = {
            "si": {
                "hStackManager": {"framesPerSlice": 10}
            }
        }
        assert get_frames_per_slice(meta) == 10

    def test_get_log_average_factor(self):
        """logAverageFactor from hScan2D."""
        meta = {
            "si": {
                "hScan2D": {"logAverageFactor": 5}
            }
        }
        assert get_log_average_factor(meta) == 5

    def test_get_z_step_size(self):
        """stackZStepSize from hStackManager."""
        meta = {
            "si": {
                "hStackManager": {"stackZStepSize": 2.5}
            }
        }
        assert get_z_step_size(meta) == 2.5


class TestComputeNumTimepoints:
    """Test num_timepoints calculation."""

    def test_lbm_timepoints(self):
        """LBM: each frame is one timepoint."""
        meta = {
            "si": {
                "hChannels": {"channelSave": list(range(1, 15))}
            }
        }
        assert compute_num_timepoints(100, meta) == 100

    def test_piezo_no_averaging(self):
        """Piezo without averaging."""
        meta = {
            "si": {
                "hStackManager": {
                    "enable": True,
                    "numSlices": 10,
                    "framesPerSlice": 5
                },
                "hScan2D": {"logAverageFactor": 1},
                "hChannels": {"channelSave": 1}
            }
        }
        # 50 frames per volume (10 slices * 5 frames/slice)
        assert compute_num_timepoints(100, meta) == 2

    def test_piezo_with_averaging(self):
        """Piezo with frame averaging."""
        meta = {
            "si": {
                "hStackManager": {
                    "enable": True,
                    "numSlices": 10,
                    "framesPerSlice": 5
                },
                "hScan2D": {"logAverageFactor": 5},
                "hChannels": {"channelSave": 1}
            }
        }
        # with averaging: 1 saved frame per slice = 10 frames per volume
        assert compute_num_timepoints(100, meta) == 10

    def test_single_plane(self):
        """Single plane: each frame is one timepoint."""
        meta = {
            "si": {
                "hChannels": {"channelSave": 1},
                "hStackManager": {"enable": False}
            }
        }
        assert compute_num_timepoints(100, meta) == 100


class TestGetStackInfo:
    """Test comprehensive stack info extraction."""

    def test_lbm_stack_info(self):
        """Full info for LBM stack."""
        meta = {
            "si": {
                "hChannels": {"channelSave": list(range(1, 15))},
                "hScan2D": {
                    "virtualChannelSettings__1": {"source": "AI0"},
                    "logAverageFactor": 1
                },
                "hStackManager": {"framesPerSlice": 1}
            }
        }
        info = get_stack_info(meta)
        assert info["stack_type"] == "lbm"
        assert info["num_zplanes"] == 14
        assert info["num_color_channels"] == 1

    def test_piezo_stack_info(self):
        """Full info for piezo stack."""
        meta = {
            "si": {
                "hChannels": {"channelSave": 1},
                "hStackManager": {
                    "enable": True,
                    "numSlices": 17,
                    "framesPerSlice": 10,
                    "stackZStepSize": 2.5
                },
                "hScan2D": {"logAverageFactor": 1}
            }
        }
        info = get_stack_info(meta)
        assert info["stack_type"] == "piezo"
        assert info["num_zplanes"] == 17
        assert info["frames_per_slice"] == 10
        assert info["dz"] == 2.5


class TestRoiInfo:
    """Test ROI and FOV extraction."""

    def test_get_roi_info_basic(self):
        """Extract basic ROI dimensions."""
        from mbo_utilities.metadata import get_roi_info

        meta = {
            "si": {
                "hRoiManager": {
                    "linesPerFrame": 512,
                    "pixelsPerLine": 512,
                }
            }
        }
        info = get_roi_info(meta)
        assert info["roi"] == (512, 512)
        assert info["fov"] == (512, 512)
        assert info["num_mrois"] == 1

    def test_get_roi_info_missing(self):
        """Handle missing ROI info gracefully."""
        from mbo_utilities.metadata import get_roi_info

        info = get_roi_info({})
        assert info["num_mrois"] == 1
        assert "roi" not in info

    def test_get_roi_info_respects_existing_num_rois(self):
        """get_roi_info should use existing num_rois from metadata."""
        from mbo_utilities.metadata import get_roi_info

        # simulate metadata that already has num_rois from get_metadata_single
        meta = {
            "num_rois": 7,  # set by get_metadata_single from RoiGroups
            "si": {
                "hRoiManager": {
                    "linesPerFrame": 68,
                    "pixelsPerLine": 68,
                }
            }
        }
        info = get_roi_info(meta)
        assert info["num_mrois"] == 7
        assert info["fov"] == (7 * 68, 68)

    def test_get_roi_info_uses_roi_groups(self):
        """get_roi_info should count from roi_groups if num_rois not set."""
        from mbo_utilities.metadata import get_roi_info

        meta = {
            "roi_groups": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
            "si": {
                "hRoiManager": {
                    "linesPerFrame": 100,
                    "pixelsPerLine": 100,
                }
            }
        }
        info = get_roi_info(meta)
        assert info["num_mrois"] == 4
        assert info["fov"] == (400, 100)


class TestFrameRate:
    """Test frame rate extraction."""

    def test_get_frame_rate_direct(self):
        """Get frame rate from scanFrameRate."""
        from mbo_utilities.metadata import get_frame_rate

        meta = {
            "si": {
                "hRoiManager": {
                    "scanFrameRate": 30.5
                }
            }
        }
        assert get_frame_rate(meta) == 30.5

    def test_get_frame_rate_from_period(self):
        """Compute frame rate from scanFramePeriod."""
        from mbo_utilities.metadata import get_frame_rate

        meta = {
            "si": {
                "hRoiManager": {
                    "scanFramePeriod": 0.1  # 10 Hz
                }
            }
        }
        assert get_frame_rate(meta) == 10.0

    def test_get_frame_rate_missing(self):
        """Return None if frame rate not available."""
        from mbo_utilities.metadata import get_frame_rate

        assert get_frame_rate({}) is None


class TestColorChannelsUnified:
    """Test that color channel detection works for both LBM and non-LBM."""

    def test_non_lbm_uses_virtual_channel_settings(self):
        """Non-LBM should also use virtualChannelSettings when available."""
        from mbo_utilities.metadata import get_num_color_channels

        meta = {
            "si": {
                "hChannels": {"channelSave": 1},
                "hStackManager": {"enable": True},
                "hScan2D": {
                    "virtualChannelSettings__1": {"source": "AI0"},
                    "virtualChannelSettings__2": {"source": "AI1"},
                }
            }
        }
        # should detect 2 color channels from AI sources
        assert get_num_color_channels(meta) == 2


class TestCleanScanImageMetadata:
    """Test that clean_scanimage_metadata adds derived fields."""

    def test_adds_stack_detection_fields(self):
        """clean_scanimage_metadata should add lbm_stack, piezo_stack, etc."""
        from mbo_utilities.metadata import clean_scanimage_metadata

        # simulate raw ScanImage metadata with LBM config
        raw_meta = {
            "si": {
                "SI.hChannels.channelSave": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                "SI.hRoiManager.scanFrameRate": 7.5,
                "SI.hRoiManager.linesPerFrame": 512,
                "SI.hRoiManager.pixelsPerLine": 512,
            }
        }
        result = clean_scanimage_metadata(raw_meta)

        assert result["stack_type"] == "lbm"
        assert result["lbm_stack"] is True
        assert result["piezo_stack"] is False
        assert result["num_zplanes"] == 14
        assert result["fs"] == 7.5
        assert result["roi"] == (512, 512)

    def test_adds_piezo_stack_fields(self):
        """clean_scanimage_metadata should detect piezo stack."""
        from mbo_utilities.metadata import clean_scanimage_metadata

        raw_meta = {
            "si": {
                "SI.hChannels.channelSave": 1,
                "SI.hStackManager.enable": True,
                "SI.hStackManager.numSlices": 17,
                "SI.hStackManager.stackZStepSize": 2.5,
            }
        }
        result = clean_scanimage_metadata(raw_meta)

        assert result["stack_type"] == "piezo"
        assert result["lbm_stack"] is False
        assert result["piezo_stack"] is True
        assert result["num_zplanes"] == 17
        assert result["dz"] == 2.5


class TestExtractRoiSlices:
    """Test extract_roi_slices function."""

    def test_extracts_roi_slices(self):
        """extract_roi_slices should compute correct slice boundaries."""
        from mbo_utilities.metadata import extract_roi_slices

        meta = {
            "page_height": 500,
            "page_width": 100,
            "num_fly_to_lines": 10,
            "roi_groups": [
                {"scanfields": {"pixelResolutionXY": [100, 200]}},
                {"scanfields": {"pixelResolutionXY": [100, 280]}},
            ],
        }
        rois = extract_roi_slices(meta)

        assert len(rois) == 2
        assert rois[0]["y_start"] == 0
        assert rois[0]["width"] == 100
        assert rois[1]["y_start"] == rois[0]["y_end"] + 10  # fly-to lines
        assert rois[1]["y_end"] == 500  # last roi ends at page_height

    def test_empty_roi_groups(self):
        """extract_roi_slices should return empty list for no roi_groups."""
        from mbo_utilities.metadata import extract_roi_slices

        assert extract_roi_slices({}) == []
        assert extract_roi_slices({"roi_groups": []}) == []

    def test_missing_page_dimensions(self):
        """extract_roi_slices should return empty list if page dims missing."""
        from mbo_utilities.metadata import extract_roi_slices

        meta = {
            "roi_groups": [{"scanfields": {"pixelResolutionXY": [100, 200]}}],
        }
        assert extract_roi_slices(meta) == []

    def test_single_roi(self):
        """extract_roi_slices should handle single ROI."""
        from mbo_utilities.metadata import extract_roi_slices

        meta = {
            "page_height": 512,
            "page_width": 512,
            "num_fly_to_lines": 0,
            "roi_groups": [{"scanfields": {"pixelResolutionXY": [512, 512]}}],
        }
        rois = extract_roi_slices(meta)

        assert len(rois) == 1
        assert rois[0]["y_start"] == 0
        assert rois[0]["y_end"] == 512
        assert rois[0]["height"] == 512
        assert rois[0]["slice"] == slice(0, 512)
