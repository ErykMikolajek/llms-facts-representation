# Niezależny benchmark MoE

Benchmark porównuje zachowanie Gemmy 3 270M i gotowego MoE na zadaniach
kompetencyjnych. Dane nie były używane w discovery, mapowaniu SAE–MLP,
pruningu, treningu routera ani kalibracji confidence. Nie oznacza to gwarancji
braku tych publicznych zbiorów w pretrainingu Gemmy.

## Wersje i status

- `independent_benchmark_v1` był smoke testem i jest zużyty;
- `independent_benchmark_v2_sealed` został uruchomiony na Kaggle; jego wyniki
  są historyczne i opisane w `results/gemma_moe_validation`;
- `independent_benchmark_v3_sealed` jest nowym, zamrożonym follow-upem. Nie
  wolno na nim stroić masek, routera, progu ani promptów.

V3 ma seed `20260916`, 600 przykładów i SHA-256
`297973c3460ff6a7205a6811c316c737c1946f0a9eb7cbc3b0aecf06c69b5fe5`.
Generator odmawia nadpisania niepustego katalogu, odrzuca wszystkie zużyte
`benchmark_id` z v1/v2 i deduplikuje BIG-bench także po treści stwierdzenia.

## Skład v3

| Domena | Zbiór/konfiguracja | Zadanie | Liczba |
| --- | --- | --- | ---: |
| prawo | MMLU `professional_law` | wybór A–D | 100 |
| biomedycyna | MMLU: anatomia, wiedza kliniczna, genetyka, medycyna zawodowa | wybór A–D | 100 |
| sport | BIG-bench `sports_understanding`, klasy 50/50 | wybór A/B | 100 |
| polityka | MMLU: `high_school_government_and_politics` 34, `security_studies` 33, `us_foreign_policy` 33 | wybór A–D | 100 |
| matematyka/LaTeX | lokalna, zbalansowana arytmetyka elementarna CC0 | wybór A–D | 100 |
| Python | MBPP `full/test` | generacja kodu | 100 |

Matematyka obejmuje po 25 dodawań, odejmowań, mnożeń i dokładnych dzieleń na
małych liczbach. Wyrażenia używają `\(...\)`, `\times` i `\div`. Zastępuje to
GSM8K, na którym oba warianty modelu 270M miały efekt podłogi.

Politykę rozszerzono o trzy zgodne podzbiory, ponieważ po wykluczeniu zużytych
rekordów w `high_school_government_and_politics` zostało tylko 46 nowych pytań.

## Multiple choice i powtórzenia

Każdy rekord MC ma strukturalne pola `mc_stem`, `mc_options`,
`mc_instruction` i `mc_stem_label`. `multiple_choice_variants()` tworzy
deterministyczne, unikalne permutacje opcji: porządek oryginalny oraz dwa
dodatkowe porządki; zadanie binarne ma maksymalnie dwa porządki. Predykcja jest
mapowana z powrotem na semantyczny indeks pierwotnej odpowiedzi.

Nie wykonuje się kilku identycznych, deterministycznych runów — ich wynik byłby
identyczny i nie mierzyłby wariancji. Raportowane są:

- accuracy dla pierwotnej kolejności;
- średnia accuracy po wariantach kolejności;
- zgodność semantycznej predykcji między wariantami;
- sparowana różnica base–MoE z bootstrapowym 95% CI;
- dokładny dwustronny test McNemara/binomialny na niezgodnych parach.

## NLL i generacja Pythona

`score_completion_pairs()` mierzy teacher-forced NLL wyłącznie tokenów
odpowiedzi, korzysta z `logits_to_keep` i dynamicznego budżetu logitów.
`compare_evaluations()` bootstrapuje różnicę `MoE - base` na poziomie
przykładów; ujemna wartość sprzyja MoE.

`extract_python_code()` obsługuje format MBPP, w którym prompt już otwiera
fence: jeśli generacja zaczyna się od `def`, `class`, dekoratora lub importu,
bierze kod przed pierwszym zamykającym fence. Następnie preferuje niepusty blok
oznaczony `python`, a dopiero potem dowolny niepusty blok. Funkcja
`is_syntactically_valid_python()` wymaga parsowalnego i **niepustego** AST.
Notebook nie wykonuje wygenerowanego kodu; funkcjonalne pass@1 wymaga
izolowanego runnera bez sieci i sekretów.

## Pomiar kosztu implementacyjnego

`benchmark_prefill_runtime()` mierzy ten sam zestaw promptów dla base i MoE,
dla batch size 1/4/8. Stosuje warm-up, wiele powtórzeń, synchronizację CUDA,
medianę i kwartyle czasu, tokens/s oraz incremental peak allocated VRAM.
Jest to pomiar prefill zależny od sprzętu i implementacji, a nie FLOP, energii
lub autoregresyjnego decode.

## Najważniejsze funkcje

| Funkcja | Odpowiedzialność |
| --- | --- |
| `render_multiple_choice_prompt()` | renderowanie permutacji z metadanych MC |
| `multiple_choice_variants()` | stabilne, unikalne runy kolejności odpowiedzi |
| `score_completion_pairs()` | pamięciooszczędne NLL completion |
| `evaluate_model()` | metryki per przykład i domena |
| `compare_evaluations()` | sparowane delty, bootstrap CI i exact paired test |
| `benchmark_prefill_runtime()` | latency, throughput i peak VRAM prefilla |
| `extract_python_code()` | odporna ekstrakcja kodu z fence Markdown |
| `is_syntactically_valid_python()` | niepusty, parsowalny AST |

## Reprodukcja

```bash
.venv/bin/python -m evaluation.prepare_independent_moe_benchmark \
  --experiment-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --samples-per-domain 100 --seed 20260916 --math-benchmark elementary
```

Po pierwszym pełnym uruchomieniu v3 staje się zużytym zbiorem testowym. Każde
dalsze strojenie wymaga osobnego developmentu i kolejnego niezależnego testu.
