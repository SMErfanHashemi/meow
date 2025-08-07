import json
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your rule.json file
with open('step 3 kg\RulesWithoutEmbed.json', 'r', encoding='utf-8') as f:
    rules = json.load(f)

# Prepare sentences for embedding
sentences = [f"{r['subject']} {r['predicate']} {r['object']}" for r in rules]

# Get embeddings
embeddings = model.encode(sentences)

# Add embeddings to each rule entry
for rule, emb in zip(rules, embeddings):
    rule['embedding'] = emb.tolist()

# Save enriched rules
with open('step 3 kg\\rules.json', 'w', encoding='utf-8') as f:
    json.dump(rules, f, ensure_ascii=False, indent=2)

print("Embeddings added and saved to rule.json")
