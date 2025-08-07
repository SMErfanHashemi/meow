# pip install sentence-transformers
import json
import re
from sentence_transformers import SentenceTransformer

# -------- Load Sentence Transformer Model --------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------- Triple Extraction Function --------
def extract_triples_from_caption(caption: str):
    caption = caption.lower()
    triples = []

    if re.search(r'\bwithout (?:a )?helmet\b', caption):
        triples.append(("person", "lacks", "helmet"))
    elif re.search(r'\bwith (?:a )?helmet\b', caption):
        triples.append(("person", "wears", "helmet"))

    if re.search(r'\bwithout (?:a )?vest\b', caption):
        triples.append(("person", "lacks", "vest"))
    elif re.search(r'\bwith (?:a )?vest\b', caption):
        triples.append(("person", "wears", "vest"))

    return triples

# -------- Main Function to Enrich Detections JSON --------
def add_triples_to_detections(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    enriched = {}
    for img_path, dets in detections.items():
        enriched[img_path] = []
        for det in dets:
            cap = det.get("cap", "")
            triples = extract_triples_from_caption(cap)

            # build string list for encoding
            triple_strings = [f"{s} {p} {o}" for (s, p, o) in triples]
            embeddings = model.encode(triple_strings) if triple_strings else []

            # attach triples + embeddings
            det_with_triples = det.copy()
            det_with_triples["triples"] = [
                {
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "embedding": emb.tolist(),
                    "image": img_path
                }
                for (s, p, o), emb in zip(triples, embeddings)
            ]

            enriched[img_path].append(det_with_triples)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"Written enriched detections with sentence-transformer embeddings to {output_json_path}")

# -------- Run Enrichment --------
if __name__ == "__main__":
    add_triples_to_detections(
        input_json_path="step 3 kg/result_postprocessed.json",
        output_json_path="step 3 kg/captions.json"
    )
