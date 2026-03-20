import os
import pandas as pd
from src.retrieval.retrieve import RadiologyRetriever
from src.generation.generate_report import ReportGenerator
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():
    studies = pd.read_csv("data/processed/studies.csv")
    val_splits = pd.read_csv("data/splits/val.csv")
    row = studies[studies["study_id"].isin(val_splits["study_id"])].iloc[0]
    
    retriever = RadiologyRetriever()
    generator = ReportGenerator()
    
    LABEL_COLS = ["cardiomegaly", "effusion", "edema", "pneumothorax", "consolidation"]
    preds_dict = {col: float(row[col]) for col in LABEL_COLS}
    hits = retriever.search(row["report_text"], top_k=5)
    
    try:
        print("Attempting generation...")
        report = generator.generate(preds_dict, hits)
        print("Success!")
        print(report.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error: {e}")
        # Let's see the raw response if possible
        # Need to hack into generator to get it or just re-implement here briefly
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        from src.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_findings, format_evidence
        findings_str = format_findings(preds_dict)
        evidence_str = format_evidence(hits)
        user_content = USER_PROMPT_TEMPLATE.format(findings_list=findings_str, top_k=5, evidence_chunks=evidence_str)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
        
        resp = client.chat.completions.create(
            model=generator.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = resp.choices[0].message.content
        print(f"Raw Content: {content}")
        
        from src.api.schemas import RadiologyReport
        try:
            # Try to see where it fails
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            clean_json = content[start_idx:end_idx+1]
            RadiologyReport.model_validate_json(clean_json)
        except Exception as ve:
            print(f"Validation Error Detail: {ve}")

if __name__ == "__main__":
    main()
