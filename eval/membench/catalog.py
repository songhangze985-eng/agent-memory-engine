"""Official MemData files from import-myself/Membench.

The git tree publishes category files under MemData/. The paper's noise-extended
`data2test` sample (Sub-dataset 1 / 2) is not in that tree.
"""

from __future__ import annotations

RAW_BASE = "https://raw.githubusercontent.com/import-myself/Membench/main/MemData"
REPO_URL = "https://github.com/import-myself/Membench"
PAPER_URL = "https://arxiv.org/abs/2506.21605"
DRIVE_URL = "https://drive.google.com/file/d/112Zraj4pTPH4Idph6i1uMOLA_LPFdGr0/view?usp=sharing"

# README badge says MIT. The repository has no LICENSE file.
LICENSE_NOTE = (
    "README 徽章写着 MIT（https://opensource.org/licenses/MIT），"
    "仓库里没有 LICENSE 文件，GitHub 也未登记许可证。"
)

# subtype file, memory level
PARTICIPATION_FACTUAL = [
    "FirstAgent/simple.json",
    "FirstAgent/aggregative.json",
    "FirstAgent/comparative.json",
    "FirstAgent/conditional.json",
    "FirstAgent/knowledge_update.json",
    "FirstAgent/post_processing.json",
    "FirstAgent/noisy.json",
    "FirstAgent/lowlevel_rec.json",
    "FirstAgent/RecMultiSession.json",
]
PARTICIPATION_REFLECTIVE = [
    "FirstAgent/highlevel.json",
    "FirstAgent/highlevel_rec.json",
]
OBSERVATION_FACTUAL = [
    "ThirdAgent/simple.json",
    "ThirdAgent/aggregative.json",
    "ThirdAgent/comparative.json",
    "ThirdAgent/conditional.json",
    "ThirdAgent/knowledge_update.json",
    "ThirdAgent/post_processing.json",
    "ThirdAgent/noisy.json",
]
OBSERVATION_REFLECTIVE = [
    "ThirdAgent/highlevel.json",
]

CATEGORIES = {
    "participation_factual": PARTICIPATION_FACTUAL,
    "participation_reflective": PARTICIPATION_REFLECTIVE,
    "observation_factual": OBSERVATION_FACTUAL,
    "observation_reflective": OBSERVATION_REFLECTIVE,
}

# Table 5, Qwen2.5-7B-Instruct, Sub-dataset 1. Cited only; not this run.
PAPER_TABLE5_QWEN_SUBDATASET1 = {
    "source": "Tan et al., MemBench, arXiv:2506.21605, Table 5, model Qwen2.5-7B-Instruct, Sub-dataset 1.",
    "FullMemory": {
        "factual_participation_accuracy": 0.647,
        "factual_observation_accuracy": 0.786,
        "reflective_participation_accuracy": 0.733,
        "reflective_observation_accuracy": 0.883,
    },
    "RetrievalMemory": {
        "factual_participation_accuracy": 0.692,
        "factual_observation_accuracy": 0.883,
        "reflective_participation_accuracy": 0.692,
        "reflective_observation_accuracy": 0.883,
    },
    "RecentMemory": {
        "factual_participation_accuracy": 0.639,
        "factual_observation_accuracy": 0.800,
        "reflective_participation_accuracy": 0.700,
        "reflective_observation_accuracy": 0.867,
    },
}
