"""
XFacta Data Preprocessing Script
Converts raw JSON data to cleaned, analysis-ready CSVs.

Usage: python3 preprocess.py

Requires: raw JSON files in xfacta_data/ (from cloud source)
Output:  cleaned CSVs in xfacta_csv/processed/
"""

import json
import csv
import os
import re
from datetime import datetime
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RAW_DATA_DIR = BASE_DIR
PROCESSED_DIR = PROJECT_DIR / "xfacta_csv" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Error Category Mapping ─────────────────────────────────────────────────
# Batches 1-10 use readable category names; batches 11-12 use numeric codes
# from a different annotation round. Based on cross-batch distribution analysis,
# the most likely mapping is:
ERROR_CATEGORY_MAP = {
    "1": "Misattributed Images",
    "2": "De-contextualization",
    "3": "Incorrectly Captioned Images",
    "1; 3": "Misattributed Images; Incorrectly Captioned Images",
    "2; 3": "De-contextualization; Incorrectly Captioned Images",
}

# ─── Helper Functions ───────────────────────────────────────────────────────

def flatten_images(images_list):
    """Convert list of image paths to a semicolon-separated string."""
    if not images_list:
        return ""
    return "; ".join(images_list)


def normalize_image_path(path, sample_type=None):
    """
    Normalize image paths to a consistent relative format.

    Before: /projects/vig/hzy/XFacta/fake_sample/media/batch5/84/images/img0.jpeg
    Before: ./media/batch1/1/images/img0.jpeg
    After:  fake_sample/media/batch5/84/images/img0.jpeg
    """
    if not path:
        return ""
    path = re.sub(r'^/projects/vig/hzy/XFacta/', '', path)
    path = re.sub(r'^\./', '', path)
    if path.startswith("media/") and sample_type:
        path = f"{sample_type}/{path}"
    return path


def normalize_image_string(images_str, sample_type=None):
    """Normalize all image paths in a semicolon-separated string."""
    if not images_str:
        return ""
    paths = [p.strip() for p in images_str.split(";")]
    normalized = [normalize_image_path(p, sample_type) for p in paths]
    return "; ".join(normalized)


def map_error_category(raw_cat):
    """
    Map numeric error categories to human-readable names.
    If already readable, return as-is.
    """
    if not raw_cat:
        return ""
    # Convert list to semicolon string if needed
    if isinstance(raw_cat, list):
        raw_cat = "; ".join(c for c in raw_cat if c) if raw_cat else ""
    # Try to map (normalize spacing: "1;3" -> "1; 3")
    normalized = raw_cat.replace(";", "; ").replace("  ", " ")
    if normalized in ERROR_CATEGORY_MAP:
        return ERROR_CATEGORY_MAP[normalized]
    return raw_cat


def parse_date(date_str):
    """
    Parse date string and return ISO-formatted datetime string.
    Returns empty string if unparseable.
    """
    if not date_str:
        return ""
    # Try common formats
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return date_str  # return as-is if can't parse


def join_with_delimiter(items, delimiter=" ||| "):
    """Join a list of strings with a safe delimiter."""
    if not items:
        return ""
    return delimiter.join(items)


# ─── 1. Process dev.json & test.json ───────────────────────────────────────

def process_dev_test():
    print("[1/3] Processing dev.json and test.json ...")
    rows = []

    for split_name in ["dev", "test"]:
        filepath = RAW_DATA_DIR / f"{split_name}.json"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dropped = 0
        for record in data:
            text = (record.get("text") or "").strip()
            if not text:
                dropped += 1
                continue

            rows.append({
                "split": split_name,
                "text": text,
                "images": flatten_images(record.get("images", [])),
                "label": str(record.get("label", "")),
            })

        print(f"   {split_name}.json: {len(data)} raw → {len(data) - dropped} kept ({dropped} dropped for empty text)")

    # Normalize image paths
    for row in rows:
        sample_type = None
        if row["images"]:
            first_img = row["images"].split(";")[0].strip()
            if "fake_sample" in first_img:
                sample_type = "fake_sample"
            elif "real_sample" in first_img:
                sample_type = "real_sample"
        row["images"] = normalize_image_string(row["images"], sample_type)

    outpath = PROCESSED_DIR / "dev_test_clean.csv"
    with open(outpath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "text", "images", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"   → Saved {len(rows)} rows to {outpath}")
    return rows


# ─── 2. Process batch JSON files ───────────────────────────────────────────

def process_batches():
    print("\n[2/3] Processing batch JSON files (real_sample & fake_sample) ...")
    rows = []
    total_dropped = 0

    for sample_type in ["real_sample", "fake_sample"]:
        sample_dir = RAW_DATA_DIR / sample_type
        batch_files = sorted(sample_dir.glob("batch*.json"))

        for bf in batch_files:
            batch_name = bf.stem  # e.g. "batch1"
            with open(bf, "r", encoding="utf-8") as f:
                data = json.load(f)

            for record in data:
                tweet = record.get("ooc_tweet", {})
                text = (tweet.get("text") or "").strip()

                if not text:
                    total_dropped += 1
                    continue

                meta = tweet.get("metadata", {})

                # Flatten flagging tweets (use ||| delimiter to avoid collision with tweet text)
                flagging_list = record.get("flagging_tweet", [])
                flag_texts = []
                flag_authors = []
                for ft in flagging_list:
                    ft_text = (ft.get("text") or "").strip()
                    if ft_text:
                        flag_texts.append(ft_text)
                    ft_meta = ft.get("metadata", {})
                    ft_author = (ft_meta.get("author_id") or "").strip()
                    if ft_author:
                        flag_authors.append(ft_author)

                # Raw error_category
                error_cat_raw = meta.get("error_category")
                if isinstance(error_cat_raw, list):
                    error_cat_raw = "; ".join(c for c in error_cat_raw if c) if error_cat_raw else ""

                # Mapped error_category (numeric → readable)
                error_cat_mapped = map_error_category(error_cat_raw)

                # Parse date
                date_raw = (meta.get("date_posted") or "").strip()
                date_iso = parse_date(date_raw)

                # Topic: fill empty with "Unknown"
                topic = (meta.get("topic") or "").strip()
                if not topic:
                    topic = "Unknown"

                # Author: fill empty with "Unknown"
                author = (meta.get("author_id") or "").strip()
                if not author:
                    author = "Unknown"

                rows.append({
                    "sample_type": sample_type,
                    "batch_file": batch_name,
                    "tweet_id": record.get("tweet_id", ""),
                    "text": text,
                    "images": flatten_images(tweet.get("images", [])),
                    "label": str(meta.get("label", "")),
                    "author": author,
                    "post_url": (meta.get("post_url") or "").strip(),
                    "date_posted": date_iso,
                    "date_raw": date_raw,  # original unparsed date for verification
                    "topic": topic,
                    "error_category_raw": error_cat_raw or "",
                    "error_category": error_cat_mapped or "",
                    "flagging_tweet_text": join_with_delimiter(flag_texts, " ||| ") if flag_texts else "",
                    "flagging_tweet_authors": join_with_delimiter(flag_authors, " ||| ") if flag_authors else "",
                })

    # Normalize image paths
    for row in rows:
        row["images"] = normalize_image_string(row["images"], row["sample_type"])

    total_raw = len(rows) + total_dropped
    print(f"   Total: {total_raw} raw → {len(rows)} kept ({total_dropped} dropped for empty text)")
    print(f"     real_sample: {sum(1 for r in rows if r['sample_type'] == 'real_sample')}")
    print(f"     fake_sample: {sum(1 for r in rows if r['sample_type'] == 'fake_sample')}")

    outpath = PROCESSED_DIR / "batches_clean.csv"
    fieldnames = [
        "sample_type", "batch_file", "tweet_id", "text", "images", "label",
        "author", "post_url", "date_posted", "date_raw", "topic",
        "error_category_raw", "error_category",
        "flagging_tweet_text", "flagging_tweet_authors"
    ]
    with open(outpath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"   → Saved {len(rows)} rows to {outpath}")
    return rows


# ─── 3. Data Quality Stats ─────────────────────────────────────────────────

def compute_quality_stats(dev_test_rows, batch_rows):
    print("\n[3/3] Computing data quality statistics ...")
    stats = {}

    # ── dev_test stats ──
    total_dt = len(dev_test_rows)
    true_dt = sum(1 for r in dev_test_rows if r["label"] == "True")
    false_dt = sum(1 for r in dev_test_rows if r["label"] == "False")
    empty_images_dt = sum(1 for r in dev_test_rows if not r["images"])

    stats["dev_test"] = {
        "total_records": total_dt,
        "true_label": true_dt,
        "false_label": false_dt,
        "true_pct": round(true_dt / total_dt * 100, 2) if total_dt else 0,
        "empty_images": empty_images_dt,
    }

    # ── batch stats ──
    total_b = len(batch_rows)
    true_b = sum(1 for r in batch_rows if r["label"] == "True")
    false_b = sum(1 for r in batch_rows if r["label"] == "False")
    unlabeled_b = sum(1 for r in batch_rows if r["label"] == "")
    empty_images_b = sum(1 for r in batch_rows if not r["images"])
    empty_author = sum(1 for r in batch_rows if r["author"] == "Unknown")
    empty_topic = sum(1 for r in batch_rows if r["topic"] == "Unknown")
    has_flagging = sum(1 for r in batch_rows if r["flagging_tweet_text"])
    has_error_cat = sum(1 for r in batch_rows if r["error_category"])

    # Topic distribution
    topics = {}
    for r in batch_rows:
        t = r["topic"]
        topics[t] = topics.get(t, 0) + 1
    top_topics = sorted(topics.items(), key=lambda x: -x[1])[:10]

    # Error category distribution (mapped)
    error_cats = {}
    for r in batch_rows:
        if r["error_category"]:
            ec = r["error_category"]
            error_cats[ec] = error_cats.get(ec, 0) + 1
    top_errors = sorted(error_cats.items(), key=lambda x: -x[1])

    # Numeric vs readable split
    numeric_count = sum(1 for r in batch_rows if r["error_category_raw"] in ERROR_CATEGORY_MAP)
    readable_count = sum(1 for r in batch_rows if r["error_category_raw"] and r["error_category_raw"] not in ERROR_CATEGORY_MAP and r["error_category"])

    stats["batches"] = {
        "total_records": total_b,
        "true_label": true_b,
        "false_label": false_b,
        "unlabeled": unlabeled_b,
        "true_pct": round(true_b / total_b * 100, 2) if total_b else 0,
        "empty_images": empty_images_b,
        "empty_author_filled": empty_author,
        "empty_topic_filled": empty_topic,
        "has_flagging_tweet": has_flagging,
        "has_error_category": has_error_cat,
        "error_category_numeric_mapped": numeric_count,
        "error_category_readable": readable_count,
        "top_topics": top_topics,
        "top_error_categories": top_errors,
    }

    return stats


# ─── 4. Overlap Check ──────────────────────────────────────────────────────

def check_overlap(dev_test_rows, batch_rows):
    """Check how many dev/test records also appear in the batch data."""
    print("\n[Overlap] Checking for duplicate records between dev/test and batches ...")

    # Build a set of unique texts from batch data
    batch_texts = set()
    for r in batch_rows:
        batch_texts.add(r["text"])

    dev_overlap = sum(1 for r in dev_test_rows if r["split"] == "dev" and r["text"] in batch_texts)
    test_overlap = sum(1 for r in dev_test_rows if r["split"] == "test" and r["text"] in batch_texts)
    total_dev = sum(1 for r in dev_test_rows if r["split"] == "dev")
    total_test = sum(1 for r in dev_test_rows if r["split"] == "test")

    print(f"   dev: {dev_overlap}/{total_dev} records overlap with batches ({round(dev_overlap/total_dev*100, 1)}%)")
    print(f"   test: {test_overlap}/{total_test} records overlap with batches ({round(test_overlap/total_test*100, 1)}%)")
    print(f"   Total: {dev_overlap + test_overlap}/{len(dev_test_rows)} ({(dev_overlap + test_overlap)/len(dev_test_rows)*100:.1f}%)")

    return {
        "dev_overlap": dev_overlap,
        "test_overlap": test_overlap,
        "dev_total": total_dev,
        "test_total": total_test,
        "total_overlap": dev_overlap + test_overlap,
        "total_dev_test": len(dev_test_rows),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dev_test_rows = process_dev_test()
    batch_rows = process_batches()
    quality_stats = compute_quality_stats(dev_test_rows, batch_rows)
    overlap_stats = check_overlap(dev_test_rows, batch_rows)

    print("\n" + "=" * 60)
    print("Data Quality Summary")
    print("=" * 60)

    print("\n── dev_test_clean.csv ──")
    for k, v in quality_stats["dev_test"].items():
        print(f"  {k}: {v}")

    print("\n── batches_clean.csv ──")
    for k, v in quality_stats["batches"].items():
        if k in ("top_topics", "top_error_categories"):
            print(f"  {k}:")
            for item, count in v:
                print(f"    - {item}: {count}")
        else:
            print(f"  {k}: {v}")

    print("\n── Overlap ──")
    for k, v in overlap_stats.items():
        print(f"  {k}: {v}")

    print("\n✅ Preprocessing complete!")
    print(f"   Output: {PROCESSED_DIR}")
