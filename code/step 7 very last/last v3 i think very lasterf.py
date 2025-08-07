import json
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# ---------- Config ----------
BASE_DIR      = Path("C:/Users/Technokade/Desktop/ConDenseCap-main/code/step 3 kg")
RULES_FILE    = BASE_DIR / "rules.json"
CAPTIONS_FILE = BASE_DIR / "captions.json"
IMAGE_DIR     = Path("C:/Users/Technokade/Desktop/ConDenseCap-main")
OUTPUT_DIR    = BASE_DIR.parent / "outputs"
SIM_THRESHOLD = 0.6
GAT_OUT_DIM   = 8
GAT_HEADS     = 2

# ---------- Load JSON ----------
def load_json(fp: Path):
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)

rules_data    = load_json(RULES_FILE)
captions_data = load_json(CAPTIONS_FILE)

# ---------- Flatten Triples ----------
def extract_triples(data):
    triples = []
    if isinstance(data, dict):
        for img_path, regions in data.items():
            for region in regions:
                cap = region.get("cap", "")
                subj_id = cap.split(" ")[0] if cap.startswith("person") else "person"
                for t in region.get("triples", []):
                    entry = {
                        "subject": subj_id,
                        "predicate": t["predicate"],
                        "object": t["object"],
                        "image": img_path,
                    }
                    if "embedding" in t:
                        entry["embedding"] = t["embedding"]
                    triples.append(entry)
    elif isinstance(data, list):
        triples = [r.copy() for r in data]
    else:
        raise ValueError("Unsupported data format for triples")
    return triples

rule_triples    = extract_triples(rules_data)
caption_triples = extract_triples(captions_data)

# ---------- Generate Embeddings ----------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

rule_texts = [f"{r['subject']} {r['predicate']} {r['object']}" for r in rule_triples]
rule_embs  = embed_model.encode(rule_texts, convert_to_numpy=True)

caption_texts = []
caption_embs  = []
valid_caption_triples = []
for cap in caption_triples:
    if "embedding" in cap:
        caption_texts.append(f"{cap['subject']} {cap['predicate']} {cap['object']}")
        caption_embs.append(cap["embedding"])
        valid_caption_triples.append(cap)
caption_embs = np.array(caption_embs)

# ---------- Build NetworkX Graph ----------
G = nx.MultiDiGraph()

def add_node_if_needed(G, node_id, embedding=None, origin=None, image=None):
    if node_id not in G:
        G.add_node(node_id, embedding=embedding, origin=[origin] if origin else [], image=image)
    else:
        if origin and origin not in G.nodes[node_id]["origin"]:
            G.nodes[node_id]["origin"].append(origin)
        if image and not G.nodes[node_id]["image"]:
            G.nodes[node_id]["image"] = image

def add_triples_to_graph(triples, origin):
    for t in triples:
        s, p, o = t["subject"], t["predicate"], t["object"]
        emb = t.get("embedding")
        img = t.get("image") if origin == "caption" else None
        add_node_if_needed(G, s, embedding=emb, origin=origin, image=img)
        add_node_if_needed(G, o, embedding=None, origin=origin)
        G.add_edge(s, o, relation=p, source=origin)

add_triples_to_graph(rule_triples, "rule")
add_triples_to_graph(caption_triples, "caption")

if "person" in G.nodes and "rule" not in G.nodes["person"]["origin"]:
    G.remove_node("person")

# ---------- GAT Processing ----------
for _, data in G.nodes(data=True):
    if data.get("embedding") is not None:
        dim = len(data["embedding"])
        break
else:
    raise ValueError("No embeddings found in graph")

id_map = {n: i for i, n in enumerate(G.nodes)}
x = torch.zeros((len(id_map), dim), dtype=torch.float32)
for n, idx in id_map.items():
    emb = G.nodes[n].get("embedding")
    if emb is not None:
        x[idx] = torch.tensor(emb, dtype=torch.float32)

edges = torch.tensor([[id_map[u], id_map[v]] for u, v in G.edges()], dtype=torch.long).t()
data = Data(x=x, edge_index=edges)

class GATModel(torch.nn.Module):
    def __init__(self, in_ch, out_ch, heads):
        super().__init__()
        self.gat = GATConv(in_ch, out_ch, heads=heads, concat=True)
    def forward(self, x, edge_index):
        return F.elu(self.gat(x, edge_index))

model = GATModel(in_ch=dim, out_ch=GAT_OUT_DIM, heads=GAT_HEADS)
out_embeds = model(data.x, data.edge_index)

# ---------- Compliance Checking ----------
sim_mat = cosine_similarity(caption_embs, rule_embs)
violations = []
report_data = []

for i, cap in enumerate(valid_caption_triples):
    subj, pred, obj = cap["subject"], cap["predicate"], cap["object"]

    if pred in ("lacks", "violates"):
        for rule in rule_triples:
            if rule["predicate"] == "requires" and rule["object"] == obj:
                violations.append((cap, f"{rule['subject']} requires {rule['object']}", 1.0))
                report_data.append({
                    "image": cap.get("image"),
                    "caption": f"{subj} {pred} {obj}",
                    "matched_rule": f"{rule['subject']} requires {rule['object']}",
                    "similarity": 1.0,
                    "type": "logic_reasoning",
                    "explanation": f"Logic inferred violation: {subj} involved in {obj}, which requires {rule['object']}, but {subj} lacks it.",
                    "doc_link": "https://example.com/safety-docs"
                })
                break
        continue

    score = sim_mat[i].max()
    best_rule = rule_texts[sim_mat[i].argmax()]

    if score < SIM_THRESHOLD:
        violations.append((cap, best_rule, score))
        report_data.append({
            "image": cap.get("image"),
            "caption": f"{subj} {pred} {obj}",
            "matched_rule": best_rule,
            "similarity": score,
            "type": "semantic_match",
            "explanation": f"Low similarity ({score:.2f}) between caption and matched rule.",
            "doc_link": "https://example.com/safety-docs"
        })

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Safety Violation Dashboard", layout="wide")
st.title("🔍 Hazard Violation Dashboard")
st.markdown("### GAT-enhanced Compliance Detection")
st.markdown("---")

if not report_data:
    st.success("✅ No violations detected.")
else:
    cols = st.columns(3)
    for i, v in enumerate(report_data):
        with cols[i % 3]:
            st.subheader(f"Violation {i+1}")
            if v['image']:
                img_path = IMAGE_DIR / v['image']
                if img_path.exists():
                    st.image(str(img_path), caption=v['image'], use_column_width=True)
            st.markdown(f"**Caption**: {v['caption']}")
            st.markdown(f"**Matched Rule**: {v['matched_rule']}")
            st.markdown(f"**Score**: {v['similarity']:.2f}")
            st.markdown(f"**Type**: {v['type']}")
            st.markdown(f"**Explanation**: {v['explanation']}")
            st.markdown(f"[View Safety Rule]({v['doc_link']})")
            st.markdown("---")

# ---------- Summary ----------
st.markdown("---")
st.success(f"Total Nodes in Graph: {len(G.nodes)}")
st.success(f"Total Edges in Graph: {len(G.edges)}")