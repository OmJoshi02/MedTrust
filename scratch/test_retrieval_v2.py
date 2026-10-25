import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.similarity_v2 import get_similarity_engine_v2

engine = get_similarity_engine_v2()
query_img = Image.open('dataset/images/00000001_000.png').convert('RGB')

for k in [5, 10, 20]:
    res = engine.find_similar_cases(
        image_input=query_img,
        top_k=k,
        predicted_findings=['Cardiomegaly'],
        primary_finding='Cardiomegaly',
        exclude_filenames=['00000001_000.png'],
    )
    print(f"=== K = {k} ===")
    print("Status:", res["status"])
    print("Candidates searched:", res["candidates_searched"])
    print("Matches returned:", len(res["matches"]))
    print(f"Top Match: {res['top_match']['image_id']} ({res['top_match']['similarity_score']}%)")
    print(f"Primary Concordance: {res['primary_concordance_rate']}% ({res['primary_concordant_count']}/{k})")
    print(f"Any Finding Overlap: {res['any_finding_overlap_rate']}% ({res['any_finding_overlap_count']}/{k})")
    ev = res["predicted_findings_evidence"][0]
    print("Evidence:", ev["explanation"])
    rep = engine.get_reference_report(res["top_match"]["image_id"])
    print(f"Top Match Report Available? {rep is not None} (ID: {rep.get('reference_id') if rep else None})")
