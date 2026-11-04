from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
#logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# NEM settings
REGIONS = ["SA1", "VIC1", "NSW1"]
# Regions whose AEMO forecast price is kept as a feature only (not modelled)
PRICE_ONLY_REGIONS = ["QLD1", "TAS1"]
START = "2021/10/01 00:00:00"
END = "2026/10/01 00:00:00"
NEMOSIS_CACHE = RAW_DATA_DIR / "nemosis_cache"
NEMOSIS_CACHE.mkdir(parents=True, exist_ok=True)

# Administered pricing (June 2022 energy crisis: price cap, then market suspension).
# Whole days D, inclusive. Kept for training, excluded from scoring and battery value.
ADMIN_PRICE_DAYS = ("2022-06-13", "2022-06-24")
