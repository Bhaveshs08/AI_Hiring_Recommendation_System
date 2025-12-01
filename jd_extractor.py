import os
import json
import argparse
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_jd(text: str):
    prompt = f"""
You are an expert Job Description extractor.

Given the raw JD text below, extract the following fields:

- job_title
- experience_required (number or range)
- primary_skills (array)
- responsibilities (summary paragraph)
- location
- description (full cleaned summary)
- source_text (original text back for reference)

Return ONLY valid JSON.

JD TEXT:
{text}
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    try:
        out = resp.output_text
        return json.loads(out)
    except Exception:
        # fallback safe parse
        cleaned = out[out.find("{") : out.rfind("}") + 1]
        return json.loads(cleaned)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JD text file")
    args = parser.parse_args()

    src_file = Path(args.input)
    if not src_file.exists():
        print("ERROR: JD input file NOT found:", src_file)
        return

    raw_text = src_file.read_text(encoding="utf-8", errors="replace")

    jd = extract_jd(raw_text)

    out_path = Path("data/jds") / (src_file.stem.replace(" ", "_") + ".json")
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jd, f, indent=2)

    print("JD extraction completed:", out_path)

if __name__ == "__main__":
    main()
