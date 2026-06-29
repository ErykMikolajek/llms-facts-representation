# llms-facts-representation
Master thesis focused on extracting, identifying and visualizing facts from LLM structure


TODO: implement multiple files-fractions dataset files

## Semantic Domain Triage

After training a Top-K SAE, run semantic triage to cluster SAE features by their logit-lens promoted tokens, select orthogonal semantic domains, and build balanced TinyStories validation sets.

Install dependencies:

```bash
pip install -r requirements.txt
```

Example run:

```bash
python3 semantic_domain_triage.py \
  --data-path data/tinystories_dataset \
  --checkpoint data/tinystories_dataset/models/checkpoints/topk_sae_layer_4_best.pt \
  --n-domains 5 \
  --samples-per-domain 100
```

Outputs are written to `data/tinystories_dataset/analysis/domain_triage/`, including `domains.json`, `feature_domain_assignments.csv`, `domain_report.md`, and balanced validation files under `domain_validation/`.