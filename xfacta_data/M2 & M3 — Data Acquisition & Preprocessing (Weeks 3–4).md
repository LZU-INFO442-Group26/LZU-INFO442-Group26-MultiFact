# XFacta Dataset — Data Acquisition, Preprocessing & Quality Report

**Course:** Milestone 2 & 3 — Data Acquisition & Preprocessing  
**Date:** May 2026

---

## 1. Data Acquisition

### 1.1 Source

The dataset used is **XFacta** (arXiv: 2508.09999), a contemporary multimodal misinformation detection corpus composed of real-world social media posts (text + images) collected from X (formerly Twitter).

**Source URL:** [Google Drive — XFacta Dataset](https://drive.google.com/drive/folders/1Sj5Rr6TpbPNzWhUjQt60fRc6xSQD2DWK)  
**Paper:** https://arxiv.org/abs/2508.09999

### 1.2 Download Instructions

The dataset can be downloaded using the provided script `xfacta_data/download_data.sh`:

<details>
<summary>Click to view the script contents</summary>

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR"

if command -v gdown &> /dev/null; then
    gdown --folder \
        "https://drive.google.com/drive/folders/1Sj5Rr6TpbPNzWhUjQt60fRc6xSQD2DWK" \
        -O "$DATA_DIR" \
        --remaining-ok
else
    echo "gdown not found. Install with: pip install gdown"
    echo "Manual download from:"
    echo "  https://drive.google.com/drive/folders/1Sj5Rr6TpbPNzWhUjQt60fRc6xSQD2DWK"
fi
```
</details>

**Option 1 — Automated (recommended):**
```bash
pip install gdown
chmod +x xfacta_data/download_data.sh
./xfacta_data/download_data.sh
```

**Option 2 — Manual:** Visit the [Google Drive folder](https://drive.google.com/drive/folders/1Sj5Rr6TpbPNzWhUjQt60fRc6xSQD2DWK), download the files, and place them in `xfacta_data/` matching the structure in §1.3.

> Both options produce identical results. Manual download is sufficient for grading purposes — the script is provided for full reproducibility.

### 1.3 Original Repository Structure

```
xfacta_data/
├── dev.json               # Development split (simple format)
├── test.json              # Test split (simple format)
├── real_sample/           # Verified real news (12 batch files)
│   ├── batch1.json
│   ├── ...
│   └── batch12.json
├── fake_sample/           # Debunked misinformation (12 batch files)
│   ├── batch1.json
│   ├── ...
│   └── batch12.json
└── {sample}_sample/
    └── media/             # Image files (NOT included in this download)
        └── batch*/        # See Section 1.5
```

**Note on `real_sample` vs `true_sample`:** The original XFacta repository uses `true_sample/` as the directory name. Our downloaded version uses `real_sample/`. The content is semantically identical.

### 1.4 Data Formats

**Simple format (dev.json / test.json):**
```json
{
  "text": "Tweet content",
  "images": ["path/to/image.jpeg"],
  "label": true
}
```

**Detailed format (batch JSON files):**
```json
{
  "tweet_id": "1",
  "ooc_tweet": {
    "text": "Tweet content",
    "images": ["./media/batch1/1/images/img0.jpeg"],
    "metadata": {
      "author_id": "CNN",
      "post_url": "https://x.com/CNN/status/...",
      "date_posted": "2024-08-26 23:25:27",
      "topic": "Politics-United-States-Presidential-Election",
      "label": true
    }
  },
  "flagging_tweet": [   // only present in fake_sample
    {
      "text": "Fact-checker explanation",
      "metadata": {
        "author_id": "Shayan Sardarizadeh",
        "post_url": "...",
        "date_posted": "..."
      }
    }
  ]
}
```

### 1.5 Media Files / Multimodal Scope

**Image files are NOT included in the downloaded JSON data.** The `images` column contains path references only (e.g., `fake_sample/media/batch5/84/images/img0.jpeg`). These paths point to the expected locations of JPEG image files that reside on the original cloud source.

**Current preprocessing scope:** This report covers **text-only preprocessing**. The image paths have been normalized and documented so that a future multimodal pipeline can load images from those locations once they are sourced. To perform multimodal analysis, the user must separately download the `media/` directories from the same Google Drive source and place them as follows:

```
xfacta_data/
├── real_sample/
│   └── media/           ← downloaded media directory
│       ├── batch1/
│       ├── ...
│       └── batch12/
├── fake_sample/
│   └── media/           ← downloaded media directory
│       ├── batch1/
│       ├── ...
│       └── batch12/
```

The normalized image paths (e.g., `fake_sample/media/batch5/84/images/img0.jpeg`) will resolve directly once `media/` is in place.

### 1.6 Dataset Composition

| Dataset | Files | Raw Records | After Cleaning |
|---------|-------|-------------|----------------|
| dev.json | 1 | 240 | 239 |
| test.json | 1 | 2,160 | 2,148 |
| real_sample (12 batches) | 12 | 1,200 | 1,200 |
| fake_sample (12 batches) | 12 | 1,200 | 1,188 |
| **Total** | **26** | **4,800** | **4,775** |

---

## 2. Preprocessing Log

### 2.1 Transformation Steps

| Step | Description | Justification |
|------|-------------|---------------|
| **2.1.1** Flatten nested JSON structures | Extracted `ooc_tweet.metadata.*` fields to top-level columns | CSV format does not support nested objects; flattening makes the data tabular and analysis-ready |
| **2.1.2** Merge split datasets | `dev.json` + `test.json` → single CSV with `split` column; 24 batch files → single CSV with `sample_type` and `batch_file` columns | Unified format enables cross-dataset analysis and consistent downstream processing |
| **2.1.3** Convert image arrays to strings | Joined `images[]` arrays with `; ` separator | CSV does not support arrays; `;` is chosen as it rarely appears in file paths |
| **2.1.4** Concatenate multiple flagging tweets | Joined `flagging_tweet[].text` and `[].metadata.author_id` with ` \|\|\| ` separator | `\|\|\|` is chosen over `;` to avoid potential collisions with tweet text content; preserves all annotations in a single row |
| **2.1.5** Normalize image paths | Converted absolute server paths (`/projects/vig/hzy/XFacta/...`) and relative paths (`./media/...`) to a consistent relative format: `{sample_type}/media/batch{id}/...` | Ensures path consistency across the unified dataset; removes dependency on original server structure |
| **2.1.6** Remove empty-text records | Dropped 13 records from dev/test and 12 from batches where text was empty | Empty text provides no signal for NLP analysis (25 records = 0.5% of total); image files are also absent from the download, so these records have no usable modality. Removal is standard practice and does not affect class balance (removed records are all `False`-labeled from `fake_sample`). |
| **2.1.7** Preserve unlabeled records | 395 records in batch11/12 have no `label` field — kept with empty label | These represent raw/unannotated data that may still be useful for semi-supervised learning or distribution analysis |
| **2.1.8** Map numeric error categories | Converted numeric codes (`"1"`, `"2"`, `"3"`, `"1; 3"`, `"2; 3"`) to readable labels | See Section 2.3 for mapping details |
| **2.1.9** Standardize date format | Validated and normalized `date_posted` to ISO format (`YYYY-MM-DD HH:MM:SS`); preserved raw value in separate column | Ensures temporal sortability for time-series analysis |
| **2.1.10** Fill structural missing values | Empty `topic` → `"Unknown"` (200 records); single empty `author` → `"Unknown"` | Prevents downstream errors from missing categorical metadata |
| **2.1.11** Overlap analysis | Checked text-based overlap between dev/test and batch datasets | Ensures data independence is understood before merged analysis (see Section 2.4) |

### 2.2 Records Removed (Empty Text)

| Source | Raw | After Drop | Removed | Reason |
|--------|-----|------------|---------|--------|
| dev.json | 240 | 239 | 1 | Empty text (label: False, had images only) |
| test.json | 2,160 | 2,148 | 12 | Empty text — from fake_sample records without text |
| real_sample batches | 1,200 | 1,200 | 0 | All records have text ✅ |
| fake_sample batches | 1,200 | 1,188 | 12 | Empty text — posts that were pure image/meme |
| **Total** | **4,800** | **4,775** | **25** | — |

### 2.3 Error Category Mapping

The `error_category` field uses two different annotation schemes across batches:

- **Batches 1-10 (fake_sample):** Human-readable labels (`Deepfakes`, `De-contextualization`, `Misattributed Images`, `Incorrectly Captioned Images`, `Miscaptioned Images`, `Named Entity Manipulations`)
- **Batches 11-12 (fake_sample):** Numeric codes (`"1"`, `"2"`, `"3"`, `"1; 3"`, `"2; 3"`) from a different annotation round

Based on cross-batch distribution analysis, numeric codes are mapped as follows:

| Numeric Code | Mapped Label | Supporting Evidence |
|-------------|--------------|-------------------|
| `"1"` | Misattributed Images | 19.5% of batches 11-12 vs 22.2% in batches 1-10 |
| `"2"` | De-contextualization | 33.5% vs 29.9% |
| `"3"` | Incorrectly Captioned Images | 39.0% vs 37.0% |
| `"1; 3"` | Misattributed Images; Incorrectly Captioned Images | Combined code |
| `"2; 3"` | De-contextualization; Incorrectly Captioned Images | Combined code |

> **Note:** This mapping is our best estimate based on ratio comparison. The original codes (`error_category_raw` column) are preserved for verification. No Deepfakes were identified in batches 11-12.

The mapped output is written to the `error_category` column; the original code is retained in `error_category_raw`.

### 2.4 Dataset Overlap (dev/test vs batches)

The dev.json and test.json datasets significantly overlap with the batch data:

| Dataset | Overlap with Batch Data | Total Records | Overlap % |
|---------|------------------------|---------------|-----------|
| dev.json | 219 | 239 | **91.6%** |
| test.json | 1,975 | 2,148 | **91.9%** |
| **Combined** | **2,194** | **2,387** | **91.9%** |

This indicates that dev/test were sampled from the same underlying batch sources. **Consequence for analysis:**

- Do **not** merge `dev_test_clean.csv` and `batches_clean.csv` without deduplication — 91.9% of records would be double-counted
- Use `batches_clean.csv` when richer metadata (author, topic, error_category) is needed
- Use `dev_test_clean.csv` when using the official train/test partition from the original paper

> **Note on methodology:** Overlap is determined by exact string matching on the normalized `text` field, since dev/test records do not carry a `tweet_id`. In theory, identical text could appear in different tweets, so the actual overlap rate may be marginally lower. However, tweet text is highly specific and the 91.9% figure is a reliable estimate.

---

## 3. Output Files

### 3.1 dev_test_clean.csv

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `split` | string | Dataset partition: `"dev"` or `"test"` | Derived from filename |
| `text` | string | Tweet content | `text` |
| `images` | string | Semicolon-separated normalized image paths | `images[]` (flattened) |
| `label` | string | Veracity label: `"True"` or `"False"` | `label` |

**Location:** `xfacta_csv/processed/dev_test_clean.csv`  
**Records:** 2,387 (2,387 data + 1 header)

### 3.2 batches_clean.csv

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `sample_type` | string | `"real_sample"` or `"fake_sample"` | Directory name |
| `batch_file` | string | Source file name, e.g. `"batch1"` | Derived from filename |
| `tweet_id` | string | Unique tweet identifier | `tweet_id` |
| `text` | string | Tweet content | `ooc_tweet.text` |
| `images` | string | Semicolon-separated normalized image paths | `ooc_tweet.images[]` |
| `label` | string | Veracity label or empty for unlabeled | `metadata.label` |
| `author` | string | Tweet author handle (filled as `"Unknown"` if missing) | `metadata.author_id` |
| `post_url` | string | Original post URL | `metadata.post_url` |
| `date_posted` | string | Normalized datetime (`YYYY-MM-DD HH:MM:SS`) | `metadata.date_posted` |
| `date_raw` | string | Original unparsed date string (for verification) | `metadata.date_posted` |
| `topic` | string | Topic classification (filled as `"Unknown"` if missing) | `metadata.topic` |
| `error_category_raw` | string | Original error category (preserves numeric codes) | `metadata.error_category` |
| `error_category` | string | Mapped error category (numeric→readable, see §2.3) | Mapped from `error_category_raw` |
| `flagging_tweet_text` | string | Fact-checker explanations (`\|\|\|`-separated for multiple) | `flagging_tweet[].text` |
| `flagging_tweet_authors` | string | Fact-checker author names (`\|\|\|`-separated) | `flagging_tweet[].metadata.author_id` |

**Location:** `xfacta_csv/processed/batches_clean.csv`  
**Records:** 2,388 (2,388 data + 1 header)

### 3.3 Data Flow

```
xfacta_data/
├── download_data.sh ─────────► Google Drive (automated download)
├── dev.json ──────────────────┐
├── test.json ─────────────────┤──► dev_test_clean.csv
│                              │
├── real_sample/               │
│   └── batch01~12.json ───────┤
├── fake_sample/               ├──► batches_clean.csv
│   └── batch01~12.json ───────┘
│
└── preprocess.py (applies all transformations)
```

---

## 4. Data Quality Summary

### 4.1 Label Balance

#### dev_test_clean.csv

| Label | Count | Percentage |
|-------|-------|------------|
| True (real) | 1,200 | **50.27%** |
| False (fake) | 1,187 | **49.73%** |
| **Total** | **2,387** | **100%** |

**Verdict:** ✅ Near-perfectly balanced. No resampling needed.

#### batches_clean.csv

| Label | Count | Percentage | Notes |
|-------|-------|------------|-------|
| True (real) | 1,000 | 41.88% | All from real_sample (batches 1-10, ~95 of batch12) |
| False (fake) | 993 | 41.58% | All from fake_sample (batches 1-10) |
| Unlabeled | 395 | **16.54%** | Batches 11-12 (both real & fake) |
| **Total** | **2,388** | **100%** | — |

**Verdict:** ✅ Balanced among labeled records (1,000 True : 993 False). Unlabeled records are structurally separate and can be excluded for supervised tasks.

### 4.2 Null / Missing Value Analysis

After preprocessing, null values are handled as follows:

| Dataset | Field | Missing | Rate | Treatment |
|---------|-------|---------|------|-----------|
| dev_test_clean | `text` | 0 | 0% | Removed during cleaning |
| dev_test_clean | `images` | 0 | 0% | — |
| dev_test_clean | `label` | 0 | 0% | — |
| batches_clean | `text` | 0 | 0% | Removed during cleaning |
| batches_clean | `images` | 0 | 0% | — |
| batches_clean | `label` | 395 | 16.54% | Kept as-is (genuinely unlabeled) |
| batches_clean | `author` | 0 | 0% | 1 record filled as `"Unknown"` |
| batches_clean | `topic` | 0 | 0% | 200 records filled as `"Unknown"` |
| batches_clean | `error_category` | 1,171 (`fake_sample` only) | 0% of fake | Field structurally absent from real_sample |
| batches_clean | `flagging_tweet_text` | 1,200 | 0% of real | Field structurally absent from real_sample |

### 4.3 Outlier Treatment

| Issue | Details | Action |
|-------|---------|--------|
| **Empty text** | 25 records (0.5% of total) with empty string as `text`, all `False`-labeled from `fake_sample`; image files also absent from download | **Removed** — no usable modality (neither text nor local images available), negligible impact on dataset size and balance |
| **Numeric error categories** | 200 records (batches 11-12) use codes `"1"`, `"2"`, `"3"` | **Mapped** to readable labels (§2.3); original preserved in `error_category_raw` |
| **Structural missing fields** | `error_category` and `flagging_*` absent from real_sample | **Documented** — not actually missing, structurally absent |
| **Image files** | Not included in download | **Documented** (§1.5) — paths point to expected locations |

### 4.4 Topic Distribution (batches_clean.csv)

| Topic | Count |
|-------|-------|
| Society | 409 |
| Politics-Warfare-Israel-Palestine | 335 |
| Politics-United-States-Presidential-Election | 234 |
| Politics-other | 226 |
| Entertainment | 224 |
| *(filled as "Unknown" — batches 11-12)* | 200 |
| Politics-Warfare-Middle-East-Conflict | 184 |
| Politics-Warfare-Lebanon-Israel Conflict | 135 |
| Politics-Warfare-War-in-Ukraine | 112 |
| Science | 97 |
| Others (≤ 75 each) | 232 |

### 4.5 Error Category Distribution (batches_clean.csv, mapped)

| Error Category | Count | Source |
|----------------|-------|--------|
| Incorrectly Captioned Images | 441 | Batches 1-10 (370) + mapped from code "3" (71) |
| De-contextualization | 373 | Batches 1-10 (299) + mapped from code "2" (67 + 7 from "2;3") |
| Misattributed Images | 260 | Batches 1-10 (222) + mapped from code "1" (38) |
| Deepfakes | 109 | Batches 1-10 only |
| Miscaptioned Images | 14 | Batches 1-10 only (variant label) |
| De-contextualization; Incorrectly Captioned Images | 9 | Mapped from code "2; 3" |
| Misattributed Images; Incorrectly Captioned Images | 7 | Mapped from code "1; 3" |
| Named Entity Manipulations | 4 | Batches 1-10 only |

### 4.6 Temporal Range

The `date_posted` column has been validated and normalized to `YYYY-MM-DD HH:MM:SS` format (sortable as-is). All dates parsed successfully. The dataset spans approximately **February 2024 to April 2025**, with the primary concentration around the 2024 US Presidential Election cycle.

### 4.7 Complete Schema

```
dev_test_clean.csv:
  split:   string  — "dev" | "test"
  text:    string  — tweet body (stripped, non-empty)
  images:  string  — normalized paths, ";"-separated
  label:   string  — "True" | "False"

batches_clean.csv:
  sample_type:          string  — "real_sample" | "fake_sample"
  batch_file:           string  — e.g. "batch1"
  tweet_id:             string  — unique ID
  text:                 string  — tweet body (stripped, non-empty)
  images:               string  — normalized paths, ";"-separated
  label:                string  — "True" | "False" | "" (unlabeled)
  author:               string  — tweet author ("Unknown" if missing)
  post_url:             string  — original X (Twitter) URL
  date_posted:          string  — ISO datetime (YYYY-MM-DD HH:MM:SS)
  date_raw:             string  — original unparsed date string
  topic:                string  — classification topic ("Unknown" if missing)
  error_category_raw:   string  — original error category value
  error_category:       string  — mapped error category (numeric→readable, §2.3)
  flagging_tweet_text:  string  — fact-checker explanation(s), "|||"-separated
  flagging_tweet_authors: string — fact-checker name(s), "|||"-separated
```

---

## 5. Reproducibility

### 5.1 Full Pipeline

```bash
# Step 1: Download raw data
chmod +x xfacta_data/download_data.sh
./xfacta_data/download_data.sh

# Step 2: Run preprocessing
python3 xfacta_data/preprocess.py
```

### 5.2 Prerequisites

- Python 3.8+
- `gdown` (only for automated download, `pip install gdown`)
- Standard library modules only for preprocessing (`json`, `csv`, `re`, `datetime`, `pathlib`)

### 5.3 Output

```
xfacta_csv/processed/
├── dev_test_clean.csv    (2,387 records)
└── batches_clean.csv     (2,388 records)
```

---

## 6. Known Limitations & Future Work

| Issue | Status | Recommendation |
|-------|--------|---------------|
| Image files not included | ⚠️ Documented | Download `media/` directories from Google Drive separately |
| Numeric error category mapping | ⚠️ Best-effort | Verify mapping with original annotators or paper authors |
| Miscaptioned Images vs Incorrectly Captioned Images | ⚠️ Overlapping labels | These appear to be synonymous; consider merging |
| Overlap between dev/test and batches | ⚠️ Documented | Avoid merging without deduplication (91.9% overlap) |
| Unlabeled data (batches 11-12) | ⚠️ Preserved | Exclude for supervised tasks; may be useful for self-training |

---

*End of report.*
