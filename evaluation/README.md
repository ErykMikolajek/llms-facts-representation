# Walidacja i benchmarki

Walidacja PPL na zamrożonym holdoucie, konstrukcja niezależnego benchmarku i
metryki porównujące model bazowy z MoE. Zużyte zbiory testowe nie mogą wracać
do strojenia routera, masek ani promptów.

```bash
python3 -m evaluation.moe_validation --help
python3 -m evaluation.prepare_independent_moe_benchmark --help
```

`domain_expert_benchmark.py` zawiera osobny protokół porównania każdego
kompaktowego eksperta z bazą: wybór domeny, sparowane NLL/accuracy, szacunek
MAC i porównanie pomiarów prefill. Notebook Kaggle korzysta z tych funkcji
bez routera i bez fallbacku.
