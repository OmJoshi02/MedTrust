import sqlite3
import json
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.engine import MedTrustBackendEngine

def backfill():
    eng = MedTrustBackendEngine()
    db_path = PROJECT_ROOT / "backend" / "medtrust.db"
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute("SELECT id, filename, full_result_json FROM cases").fetchall()

    for cid, fname, res_json in rows:
        try:
            data = json.loads(res_json)
            pred = data.get("prediction", {})
            p_finding = pred.get("primary_finding", {}).get("disease")
            det_findings = [
                f["disease"]
                for f in pred.get("findings", [])
                if f.get("decision") == "POSITIVE" or f.get("calibrated_predicted_positive")
            ]

            sample_path = PROJECT_ROOT / "test_images" / fname
            if sample_path.exists():
                img = Image.open(sample_path)
                sim_res = eng.run_similarity(
                    img,
                    top_k=5,
                    query_findings=det_findings,
                    primary_finding=p_finding,
                )
                data["similarity"] = sim_res
                cur.execute("UPDATE cases SET full_result_json = ? WHERE id = ?", (json.dumps(data), cid))
                top_name = sim_res["top_match"]["filename"] if sim_res.get("top_match") else "None"
                top_score = sim_res["top_match"]["similarity_score"] if sim_res.get("top_match") else 0.0
                conc_rate = sim_res.get("concordance_rate", 0.0)
                print(f"Updated Case #{cid} ({fname}): Top match {top_name} ({top_score}%), Concordance: {conc_rate}%.")
        except Exception as e:
            print(f"Error updating Case #{cid}: {e}")

    con.commit()
    con.close()
    print("Backfill complete.")

if __name__ == "__main__":
    backfill()
