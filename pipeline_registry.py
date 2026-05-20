"""
Pipeline Registry — Pipeline Optima™
Complete list of all managed pipelines, grouped by category.
Used for the two-level dropdown (Category → Pipeline) in the UI and
for dispatching to per-pipeline DRA ML models.
"""

from typing import Dict, List

# ── Master registry: code → metadata ─────────────────────────────────────────
PIPELINE_REGISTRY: Dict[str, Dict[str, str]] = {

    # ── Product Pipelines (A1) ────────────────────────────────────────────────
    "KAPL":   {"name": "Koyali Ahmedabad Pipeline",              "category": "Product Pipelines"},
    "KJPPL":  {"name": "Koyali Jaipur Panipat Pipeline",         "category": "Product Pipelines"},
    "KDPL":   {"name": "Koyali Dahej Pipeline",                  "category": "Product Pipelines"},
    "KRPL":   {"name": "Koyali Ratlam Pipeline",                 "category": "Product Pipelines"},
    "BKPL":   {"name": "Barauni Kanpur Pipeline",                "category": "Product Pipelines"},
    "HMRPL":  {"name": "Haldia Mourigram Rajbandh Pipeline",     "category": "Product Pipelines"},
    "HBPL":   {"name": "Haldia Barauni Pipeline",                "category": "Product Pipelines"},
    "GSPL":   {"name": "Guwahati Siliguri Pipeline",             "category": "Product Pipelines"},
    "PBPL":   {"name": "Panipat Bhatinda Pipeline",              "category": "Product Pipelines"},
    "PRPL":   {"name": "Panipat Rewari Pipeline",                "category": "Product Pipelines"},
    "PAJPL":  {"name": "Panipat Ambala Jalandhar Pipeline",      "category": "Product Pipelines"},
    "PDPL":   {"name": "Panipat Delhi Pipeline",                 "category": "Product Pipelines"},
    "MDPL":   {"name": "Mathura Delhi Pipeline",                 "category": "Product Pipelines"},
    "MAGPL":  {"name": "Mathura Agra Gawria Pipeline",           "category": "Product Pipelines"},
    "MBPL":   {"name": "Mathura Bharatpur Pipeline",             "category": "Product Pipelines"},
    "CTMPL":  {"name": "Chennai Trichy Madurai Pipeline",        "category": "Product Pipelines"},
    "CBPL":   {"name": "Chennai Bangalore Pipeline",             "category": "Product Pipelines"},
    "PRRPL":  {"name": "Paradip Raipur Ranchi Pipeline",         "category": "Product Pipelines"},
    "PHPL":   {"name": "Paradip Hyderabad Pipeline",             "category": "Product Pipelines"},
    "PSHPL":  {"name": "Paradip Somnathpur Haldia Pipeline",     "category": "Product Pipelines"},
    "KASPL":  {"name": "Koyali Ahmednagar Solapur Pipeline",     "category": "Product Pipelines"},
    "HBPL18": {"name": "18\" HBPL - Haldia Barauni Product Pipeline",
                                                                  "category": "Product Pipelines"},

    # ── ATF Pipelines (A2) ────────────────────────────────────────────────────
    "PBPL_ATF":        {"name": "Panipat Bijwasan ATF line",              "category": "ATF Pipelines"},
    "CHENNAI_ATF":     {"name": "Chennai Meenambakkam ATF Pipeline",      "category": "ATF Pipelines"},
    "BENGALURU_ATF":   {"name": "Devangonthi Devanhalli Pipeline",        "category": "ATF Pipelines"},
    "KOLKATA_ATF":     {"name": "Kolkata ATF Pipeline",                   "category": "ATF Pipelines"},
    "LUCKNOW_ATF":     {"name": "Lucknow ATF Pipeline",                   "category": "ATF Pipelines"},
    "BHUBANESWAR_ATF": {"name": "Bhubaneswar ATF Pipeline",               "category": "ATF Pipelines"},

    # ── LPG Pipelines (A3) ────────────────────────────────────────────────────
    "PJPL":   {"name": "Panipat Jalandhar Pipeline",             "category": "LPG Pipelines"},
    "PHBMPL": {"name": "Paradip Haldia Barauni Motihari Pipeline","category": "LPG Pipelines"},

    # ── Crude Oil Pipelines (B) ───────────────────────────────────────────────
    "SMPL":  {"name": "Salaya Mathura Pipeline",                 "category": "Crude Oil Pipelines"},
    "MPPL":  {"name": "Mundra Panipat Pipeline",                 "category": "Crude Oil Pipelines"},
    "PHBPL": {"name": "Paradip Haldia Barauni Pipeline",         "category": "Crude Oil Pipelines"},

    # ── Gas Pipelines (C) ─────────────────────────────────────────────────────
    "DPPL":  {"name": "Dadri Panipat R LNG Pipeline",            "category": "Gas Pipelines"},
    "ETBPL": {"name": "Ennore-Tuticorin-Bengaluru Pipeline",     "category": "Gas Pipelines"},
    "DKPL":  {"name": "Dahej Koyali R LNG Pipeline",             "category": "Gas Pipelines"},
}

# ── Category → ordered list of pipeline codes ─────────────────────────────────
PIPELINE_CATEGORIES: Dict[str, List[str]] = {
    "Product Pipelines": [
        "KAPL", "KJPPL", "KDPL", "KRPL", "BKPL", "HMRPL", "HBPL", "GSPL",
        "PBPL", "PRPL", "PAJPL", "PDPL", "MDPL", "MAGPL", "MBPL",
        "CTMPL", "CBPL", "PRRPL", "PHPL", "PSHPL", "KASPL", "HBPL18",
    ],
    "ATF Pipelines": [
        "PBPL_ATF", "CHENNAI_ATF", "BENGALURU_ATF",
        "KOLKATA_ATF", "LUCKNOW_ATF", "BHUBANESWAR_ATF",
    ],
    "LPG Pipelines": [
        "PJPL", "PHBMPL",
    ],
    "Crude Oil Pipelines": [
        "SMPL", "MPPL", "PHBPL",
    ],
    "Gas Pipelines": [
        "DPPL", "ETBPL", "DKPL",
    ],
}

# Default category and pipeline shown on first load
DEFAULT_CATEGORY = "Crude Oil Pipelines"
DEFAULT_PIPELINE_CODE = "PHBPL"  # current CSV data is for this pipeline


def get_display_options(category: str) -> List[str]:
    """Return display strings for a category dropdown: ['CODE - Full Name', ...]"""
    codes = PIPELINE_CATEGORIES.get(category, [])
    return [f"{c} - {PIPELINE_REGISTRY[c]['name']}" for c in codes if c in PIPELINE_REGISTRY]


def code_from_display(display: str) -> str:
    """Extract pipeline code from a display string like 'BKPL - Barauni Kanpur Pipeline'."""
    return display.split(" - ")[0].strip()


def get_pipeline_info(code: str) -> Dict[str, str]:
    """Return metadata dict for a pipeline code, or empty dict if not found."""
    return PIPELINE_REGISTRY.get(code, {})
