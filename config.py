from pathlib import Path

EXPERIMENT_NAME = "MLP"

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
CUDA_DIR = ROOT_DIR / "cuda"
COMMON_DIR = CUDA_DIR / "common"
REFERENCE_PATH = CUDA_DIR / "reference" / "impl.cu"
IMPL_PATH = CUDA_DIR / "optimized" / "impl.cu"
BUILD_DIR = ROOT_DIR / "build"
LOGS_DIR = ROOT_DIR / "logs"
LOG_FILE = LOGS_DIR / "log.jsonl"
REF_OUTPUT_FILE = BUILD_DIR / "baseline.bin"
SYSTEM_PROMPT_PATH = ROOT_DIR / "prompts" / "system_prompt.md"

# Tool paths
CMAKE = "cmake"
NCU = "ncu"
GIT = "git"

# CMake / build
REFERENCE_TARGET = "reference_bin"
OPTIMIZED_TARGET = "optimized_bin"

# Binary paths (cmake outputs to build/bin/)
REFERENCE_BIN = BUILD_DIR / "bin" / "reference_bin"
OPTIMIZED_BIN = BUILD_DIR / "bin" / "optimized_bin"

# Benchmark
BENCHMARK_WARMUP = 50
BENCHMARK_TIMED = 100
BENCHMARK_METRIC = "median"  # "median" or "min"
# number of seeds for validation/benchmark, set to 1 if all your hyperparameters are constexpr
NUM_SEEDS = 5

# Improvement must be at least this fraction (e.g. 0.01 = 1% faster to accept)
IMPROVEMENT_THRESHOLD = 0.01

# LLM
OPENAI_BASE_URL = "http://127.0.0.1:8001/v1"
OPENAI_MODEL = "Qwen3.8-27B-Q4_K_M.gguf"
MAX_ITERATIONS = 125
MAX_RETRIES = 5  # per iteration: apply/build/validate failures before giving up and restarting a new iteration
MAX_NO_IMPROVEMENT = 25  # early stop after this many consecutive iterations without improvement
