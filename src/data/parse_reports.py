"""
parse_reports.py — Parse OpenI IU Chest X-ray XML reports.

XML structure:
  <eCitation>
    <uId id="1"/>
    <Abstract>
      <AbstractText Label="FINDINGS">...</AbstractText>
      <AbstractText Label="IMPRESSION">...</AbstractText>
      ...
    </Abstract>
    <parentImage id="CXR1_IM-0001-1001"/>
    ...
  </eCitation>

Each XML file → one report record with:
  - study_id: XML filename stem (e.g. "1" for "1.xml")
  - findings:  text of FINDINGS section (empty string if missing)
  - impression: text of IMPRESSION section (empty string if missing)
  - image_ids:  list of parentImage id values (first = primary/frontal)
"""

import os
import glob
import logging
from xml.etree import ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_single_xml(xml_path: str) -> dict | None:
    """Parse a single OpenI XML report. Returns None on parse error."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning(f"Failed to parse {xml_path}: {e}")
        return None

    study_id = Path(xml_path).stem  # e.g. "1" for "1.xml"

    # Extract section text
    sections = {}
    for elem in root.iter("AbstractText"):
        label = (elem.get("Label") or "").upper()
        text = (elem.text or "").strip()
        sections[label] = text

    findings = sections.get("FINDINGS", "")
    impression = sections.get("IMPRESSION", "")

    # Extract image IDs — parentImage id attribute
    image_ids = [img.get("id") for img in root.iter("parentImage") if img.get("id")]

    return {
        "study_id": study_id,
        "findings": findings,
        "impression": impression,
        "image_ids": image_ids,
    }


def parse_all_reports(reports_dir: str) -> list[dict]:
    """
    Parse all XML files in reports_dir.
    Returns a list of record dicts.
    """
    xml_files = sorted(glob.glob(os.path.join(reports_dir, "**", "*.xml"), recursive=True))
    logger.info(f"Found {len(xml_files)} XML files in {reports_dir}")

    records = []
    missing_findings = 0
    missing_impression = 0
    missing_images = 0

    for xml_path in xml_files:
        record = parse_single_xml(xml_path)
        if record is None:
            continue

        if not record["findings"]:
            missing_findings += 1
        if not record["impression"]:
            missing_impression += 1
        if not record["image_ids"]:
            missing_images += 1
            logger.warning(f"No parentImage found in {xml_path} — skipping")
            continue  # skip reports with no image association

        records.append(record)

    logger.info(f"Parsed {len(records)} valid records")
    logger.info(f"  Missing FINDINGS:   {missing_findings}")
    logger.info(f"  Missing IMPRESSION: {missing_impression}")
    logger.info(f"  Skipped (no image): {missing_images}")

    return records


if __name__ == "__main__":
    REPORTS_DIR = "data/raw/reports"
    records = parse_all_reports(REPORTS_DIR)
    print(f"\nFirst record: {records[0]}")
