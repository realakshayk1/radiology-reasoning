import os
import logging
from src.generation.generate_report import ReportGenerator
from src.api.schemas import RadiologyReport
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():
    generator = ReportGenerator(model_name="gpt-5.4-mini-2026-03-17")
    
    # Mock data
    preds_dict = {"cardiomegaly": 0.85, "effusion": 0.12}
    hits = [
        {"study_id": "MOCK_1", "chunk_text": "Mild cardiomegaly is noted.", "section": "findings"},
        {"study_id": "MOCK_2", "chunk_text": "No pleural effusion.", "section": "findings"}
    ]
    
    print(f"Testing generation with model: {generator.model_name}")
    try:
        report = generator.generate(preds_dict, hits)
        print("Generator returned success!")
        print(f"Report: {report.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Caught top-level error: {e}")
        
    # Manual check of raw response
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    from src.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_findings, format_evidence
    
    findings_str = format_findings(preds_dict)
    evidence_str = format_evidence(hits)
    user_content = USER_PROMPT_TEMPLATE.format(findings_list=findings_str, top_k=2, evidence_chunks=evidence_str)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    
    print("\nManual raw call for debugging...")
    resp = client.chat.completions.create(
        model=generator.model_name,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0
    )
    content = resp.choices[0].message.content
    print(f"Raw Content: {content}")
    
    try:
        # Robust extraction logic from generate_report.py
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        clean_json = content[start_idx:end_idx+1]
        print(f"Cleaned JSON: {clean_json}")
        RadiologyReport.model_validate_json(clean_json)
        print("Pydantic validation passed manually!")
    except Exception as ve:
        print(f"Pydantic verification failed: {ve}")

if __name__ == "__main__":
    main()
