import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH = DATA_DIR / "KDDTest+.txt"
RESULTS_DIR = ROOT / "results"

SEED = 42
SPLIT = 0.30
CAP_GRID = [0.001, 0.005, 0.01, 0.02]
DEFAULT_CAP = 0.01

MAX_RULES = 5
MAX_CONDS = 3
MIN_TP = 20
MIN_FAMILY_N = 30
NUM_QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]

DRIFT_TOL = 0.0025  # --decide: max B2 val-fit drift 0.0000pp over 4 caps -> ceil 0.25pp
RECOMMENDED_CAP = 0.02  # --decide: highest val macro-recall 0.0977 among 4 SAFE caps

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite")

# Positional NSL-KDD names, UNVERIFIED until the schema report is checked.
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
LABEL_COL = "label"
DIFFICULTY_COL = "difficulty"
EXPECTED_FIELDS = 43

FEATURE_COLUMNS = COLUMNS[:41]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]
