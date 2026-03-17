"""
prompts.py — Prompt templates for RadReason report generation.
"""

SYSTEM_PROMPT = """You are a radiology decision-support assistant.
Use only the predictions and retrieved evidence provided.
Do not invent findings not present in the evidence.
Return ONLY a valid JSON object with keys: findings, impression, rationale, caution, escalate, confidence.
No markdown, no code fences, no text outside the JSON."""

USER_PROMPT_TEMPLATE = """Predicted findings (calibrated probabilities, threshold = 0.4):
{findings_list}

Retrieved evidence (top {top_k} chunks):
{evidence_chunks}

Generate: findings, impression, rationale, caution, escalate (bool), confidence (high/moderate/low)."""

def format_findings(predictions, threshold=0.4):
    """Format a list of (label, prob) into a string."""
    active = [f"{label} ({prob:.2f})" for label, prob in predictions.items() if prob >= threshold]
    if not active:
        return "No specific findings predicted above threshold."
    return ", ".join(active)

def format_evidence(hits):
    """Format retrieval hits into a numbered list."""
    if not hits:
        return "No relevant evidence found."
    lines = []
    for i, hit in enumerate(hits):
        lines.append(f"[{i+1}] (Source: {hit['study_id']}) {hit['chunk_text']}")
    return "\n".join(lines)
