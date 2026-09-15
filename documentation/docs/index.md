# Dokumentacja kodu eksperymentalnego

Dokumentacja opisuje kanoniczne pliki Python realizujące eksperyment dotyczący
lokalizacji reprezentacji faktów i konceptów z użyciem rzadkich autoenkoderów
(SAE), domen semantycznych, pruningowania MLP oraz wariantu
mixture-of-experts (MoE). Każdy moduł ma osobną stronę: najpierw przedstawia
ona rolę modułu i jego miejsce w eksperymencie, a następnie precyzyjnie opisuje
funkcje, dane, algorytmy, checkpointy, warunki poprawności i ograniczenia.

## Dwa tory eksperymentalne

Repozytorium zawiera dwa powiązane, ale różne tory:

1. **Pythia-160M / lokalny Top-K SAE** — sekwencjonowanie korpusu, strumieniowe
   pozyskiwanie aktywacji, trening własnego Top-K SAE i analiza cech.
2. **Gemma 3 270M / Gemma Scope 2** — analiza gotowego JumpReLU SAE ładowanego
   przez SAELens; etap treningu SAE jest pomijany.

Oba tory mogą zasilać dalszy pipeline:

`cechy SAE → domeny semantyczne → neurony MLP → eksperci → router → MoE → PPL`.

!!! info "Stan eksperymentu Gemma"
    Dla Gemmy 3 270M wykonano pełny PoC: stabilny triaż sześciu domen,
    niezależne dane development/holdout, mapping, walidację selektywności,
    pruning 25%, trening routera, hard-routed MoE i zamrożony test PPL.
    Pierwszy niezależny benchmark 600 zadań został wykonany i przeanalizowany.
    Świeży follow-up v3 naprawia metrykę Pythona, upraszcza matematykę, dodaje
    warianty kolejności MC oraz pomiar prefill i jest gotowy do Kaggle.
    [Zobacz kompletny przebieg i wyniki](gemma_moe_experiment.md).

!!! warning "Interpretacja metodologiczna"
    MoE zachował większość jakości bazowej, ale nie poprawił perplexity.
    Automatyczne nazwy domen pozostają proxy, korelacja nie dowodzi
    przyczynowości, a eksperci są przyciętymi kopiami jednej warstwy MLP, nie
    niezależnie trenowanymi ekspertami klasycznego sparse MoE.

## Jak czytać dokumentację

- [Przepływ eksperymentu](pipeline.md) pokazuje zależności i formaty artefaktów.
- [Zakres i status plików](scope.md) wyjaśnia, dlaczego dokumentowane są
  wersje z pakietów etapowych, a nie ich kopie z `praca_tex/source_code/`.
- Strony modułów zawierają sekcję **Przebieg krok po kroku**, a następnie
  kompletny **Katalog funkcji i klas**.
- Ostrzeżenia wskazują miejsca, w których nazwa historyczna, przybliżenie albo
  sposób doboru danych mogą prowadzić do błędnej interpretacji wyników.

## Moduły objęte dokumentacją

| Obszar | Moduły |
| --- | --- |
| Sterowanie i dane | [`sae_pipeline/main.py`](main.md), [`sae_pipeline/dataset_sequencing.py`](dataset_sequencing.md) |
| Aktywacje i SAE | [`sae_pipeline/activations_collecting.py`](activations_collecting.md), [`sae_pipeline/autoencoder_training.py`](autoencoder_training.md), [`sae_pipeline/features_analysis.py`](features_analysis.md) |
| Gotowy SAE Gemmy | [`sae_pipeline/gemma_scope_analysis.py`](gemma_scope_analysis.md) |
| Domeny i MLP | [`domain_triage/semantic_domain_triage.py`](semantic_domain_triage.md), [`domain_mapping/topographic_mlp_sae_mapping.py`](topographic_mlp_sae_mapping.md), [`domain_mapping/domain_mlp_activation_mapping.py`](domain_mlp_activation_mapping.md), [`domain_mapping/domain_mlp_pruning.py`](domain_mlp_pruning.md) |
| Router i ewaluacja | [`moe/router_training.py`](router_training.md), [`moe/moe_assembly.py`](moe_assembly.md), [`evaluation/moe_validation.py`](moe_validation.md) |
| Stabilność i dane | [`domain_triage/semantic_domain_stability.py`, `domain_triage/semantic_domain_cross_view.py`](triage_stability.md), [`domain_triage/prepare_gemma_domain_experiment.py`, `domain_mapping/prepare_domain_datasets.py`](domain_datasets.md), [`domain_mapping/validate_domain_sae_selectivity.py`](domain_selectivity.md) |
| Reprodukcja | [`moe/freeze_gemma_moe_protocol.py`, `moe/prepare_gemma_moe_bundle.py`](reproducibility.md) |
| Benchmark | [`evaluation/prepare_independent_moe_benchmark.py`, `evaluation/moe_benchmark.py`](independent_benchmark.md), [notebook MoE](kaggle_moe_benchmark.md), [osobne modele dziedzinowe](domain_expert_benchmark.md) |
| Infrastruktura | [`common/utils.py`](utils.md), [`kaggle/cli/kaggle_pipeline.py`](kaggle_pipeline.md) |

## Konwencje tensorów

| Symbol | Znaczenie |
| --- | --- |
| `B` | liczba sekwencji w batchu modelu |
| `S` | długość sekwencji |
| `D = d_model` | szerokość residual stream / wymiar wejścia SAE |
| `F = d_sae` | liczba cech SAE |
| `H = n_inner` | liczba fizycznych neuronów pośrednich MLP |
| `V` | rozmiar słownika tokenizera |
| `N` | liczba ważnych tokenów w analizowanym fragmencie danych |

Najważniejsze rozróżnienie to `D ≠ H`: aktywacja residual stream ma wymiar
`D`, natomiast wektor fizycznych neuronów MLP ma wymiar `H`. To rozróżnienie
decyduje, czy wynik można wykorzystać do pruningowania wag MLP.
