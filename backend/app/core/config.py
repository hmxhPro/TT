"""
app/core/config.py
------------------
Central application configuration via Pydantic Settings.
Reads from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Literal


def _gdino_config_default() -> Path:
    """Use the config bundled with the installed groundingdino package."""
    try:
        import groundingdino
        pkg_config = Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
        if pkg_config.exists():
            return pkg_config
    except ImportError:
        pass
    return Path("./models/groundingdino/config/GroundingDINO_SwinT_OGC.py")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ── Storage ─────────────────────────────────────────────────────────────
    UPLOAD_DIR: Path = Path("./uploads")
    RESULTS_DIR: Path = Path("./results")

    # ── Detection Model ──────────────────────────────────────────────────────
    DETECTION_MODEL: Literal["florence2", "grounding_dino", "yolo_world"] = "grounding_dino"
    DEVICE: str = "cuda:0"

    # Florence-2
    FLORENCE2_MODEL_ID: str = "microsoft/Florence-2-large"

    # Grounding DINO
    GDINO_CONFIG_PATH: Path = Field(default_factory=_gdino_config_default)
    GDINO_CHECKPOINT_PATH: Path = Path(
        "./models/groundingdino/weights/groundingdino_swint_ogc.pth"
    )

    # YOLO-World
    YOLO_WORLD_MODEL: str = "yolo11l-world.pt"  # yolo11l-world.pt, yolo11m-world.pt, yolo11s-world.pt
    SAHI_SLICE_HEIGHT: int = 640
    SAHI_SLICE_WIDTH: int = 640
    SAHI_OVERLAP_HEIGHT_RATIO: float = 0.2
    SAHI_OVERLAP_WIDTH_RATIO: float = 0.2

    # ── Detection Thresholds ─────────────────────────────────────────────────
    # Three-tier confidence strategy
    DINO_DISCARD_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    DINO_CANDIDATE_THRESHOLD: float = Field(default=0.30, ge=0.0, le=1.0)
    DINO_DIRECT_CONFIRM_THRESHOLD: float = Field(default=0.60, ge=0.0, le=1.0)

    # Legacy thresholds (for backward compatibility)
    BOX_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    TEXT_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)

    # ── ByteTrack ────────────────────────────────────────────────────────────
    # Aligned with BOX_THRESHOLD: ByteTrack would otherwise filter detector
    # outputs a second time at this threshold and drop low-score small-object
    # boxes (0.25–0.5) before any track can be created.
    TRACK_THRESH: float = Field(default=0.25, ge=0.0, le=1.0)
    TRACK_BUFFER: int = 30
    MATCH_THRESH: float = Field(default=0.8, ge=0.0, le=1.0)

    # ── Processing ───────────────────────────────────────────────────────────
    # Run full detection every N frames; track in between for speed
    DETECTION_INTERVAL: int = Field(default=5, ge=1)
    MAX_CONCURRENT_TASKS: int = 2

    # ── Adaptive Keyframe Detection ──────────────────────────────────────────
    ENABLE_ADAPTIVE_KEYFRAME: bool = True
    FRAME_DIFF_THRESHOLD: float = Field(default=0.05, ge=0.0, le=1.0)
    PHASH_THRESHOLD: int = Field(default=8, ge=0)
    HIST_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0)
    MIN_DETECTION_INTERVAL: int = Field(default=10, ge=1)
    MAX_DETECTION_INTERVAL: int = Field(default=30, ge=1)

    # ── Frame Saving Strategy ────────────────────────────────────────────────
    # Options: "all" | "keyframes_only" | "detections_only"
    # - "all": Save every frame (original behavior, large ZIP files)
    # - "keyframes_only": Only save frames where detection was run (recommended, 70-90% reduction)
    # - "detections_only": Only save frames with actual detections (maximum reduction)
    SAVE_FRAMES_MODE: Literal["all", "keyframes_only", "detections_only"] = "keyframes_only"

    # JPEG quality for saved frames (1-100, lower = smaller files)
    JPEG_QUALITY: int = Field(default=85, ge=1, le=100)

    # ── Frame Quality Filtering ──────────────────────────────────────────────
    ENABLE_FRAME_QUALITY_CHECK: bool = True
    BLACK_FRAME_THRESHOLD: float = Field(default=15.0, ge=0.0, le=255.0)
    DARK_FRAME_THRESHOLD: float = Field(default=30.0, ge=0.0, le=255.0)
    BRIGHT_FRAME_THRESHOLD: float = Field(default=240.0, ge=0.0, le=255.0)
    MIN_FRAME_STD_DEV: float = Field(default=10.0, ge=0.0)

    # ── Visualization ────────────────────────────────────────────────────────
    CORNER_STYLE_BOX: bool = True
    CORNER_LENGTH: int = Field(default=30, ge=10)
    BOX_THICKNESS: int = Field(default=4, ge=1)

    # ── Color Filtering ──────────────────────────────────────────────────────
    ENABLE_COLOR_FILTER: bool = True
    COLOR_FILTER_LOG_LEVEL: str = "DEBUG"  # DEBUG | INFO | WARNING

    # ── VLM (MiniCPM-V) ─────────────────────────────────────────────────────
    VLM_ENABLED: bool = True
    VLM_API_BASE: str = "http://localhost:8010/v1"
    VLM_MODEL_NAME: str = "MiniCPM-V-4_5"
    VLM_MAX_CONCURRENT: int = 1
    VLM_INTERVAL_TENTATIVE: float = 3.0
    VLM_INTERVAL_CONFIRMED: float = 8.0
    VLM_SCORE_THRESHOLD: float = 0.45
    CROP_PADDING_NORMAL: float = 0.15
    CROP_PADDING_SMALL: float = 0.30
    SMALL_OBJECT_AREA_RATIO: float = 0.01

    # ── Fusion ───────────────────────────────────────────────────────────────
    VLM_WEIGHT: float = 0.6
    DINO_WEIGHT: float = 0.4
    VLM_CONFIRM_THRESHOLD: float = 0.65
    FINAL_CONFIRM_THRESHOLD: float = 0.65
    CONFIRM_THRESHOLD: float = 0.65
    REJECT_THRESHOLD: float = 0.35

    # Track confirmation requirements
    MIN_HIT_COUNT_HIGH_CONF: int = 2
    MIN_HIT_COUNT_MID_CONF: int = 3

    # ── Optional Redis ───────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── PostgreSQL (task history persistence) ───────────────────────────────
    # Override in .env for non-default credentials. Falls back gracefully if
    # the DB is unreachable — detection still works, just without history.
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sod"
    DATABASE_ECHO: bool = False  # set True to log every SQL statement

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
settings.ensure_dirs()
