# Stabilność i zgodność triażu

Strona opisuje moduły `domain_triage/semantic_domain_stability.py` oraz
`domain_triage/semantic_domain_cross_view.py`, które poprzedziły ręczny wybór sześciu domen.

## `domain_triage/semantic_domain_stability.py`

Skrypt ładuje pełną analizę cech, buduje filtrowaną macierz cecha–token przez
wspólne funkcje z `domain_triage/semantic_domain_triage.py`, a następnie powtarza
klasteryzację dla siatki kontrolnej:

- baseline;
- `svd_components` 30 i 80;
- dwa dodatkowe seedy;
- `min_cluster_size` 20 i 80.

Dla każdego klastra baseline szuka najlepszego odpowiednika w każdym runie i
zapisuje Jaccard zbiorów `feature_id`, przecięcie oraz etykietę dopasowania.
Ranking kandydatów łączy kohezję, stabilność, rozmiar, liczbę różnych tokenów
i kary za klastry zbyt szerokie. Skrypt nie nadaje ostatecznych nazw: zapisuje
shortlistę i konteksty do inspekcji.

Opcjonalny drugi widok budowany jest z TF–IDF n-gramów słów w kontekstach
aktywacji cech. Parametry `context_min_df`, `context_max_df`, limit słownika i
liczba terminów na cechę zapobiegają nieograniczonemu wzrostowi macierzy.

Artefakty obejmują `stability_results.json`, `stability_report.md`,
`shortlist.json`, macierz `.npz`, `valid_feature_ids.npy`, etykiety każdego
runu oraz `candidate_contexts/`.

## `domain_triage/semantic_domain_cross_view.py`

Moduł ocenia, czy klaster odkryty w widoku podstawowym pozostaje zwarty w
drugiej macierzy cecha–termin. Kohezja jest średnim podobieństwem cech do
znormalizowanego centroidu. Istotność wyznacza test permutacyjny z losowymi
zbiorami cech o tym samym rozmiarze; wartości `p` są korygowane procedurą
Benjamini–Hochberga.

Wyniki `cross_view_results.json` i `cross_view_report.md` są dowodem
pomocniczym. Zgodność dwóch widoków nie gwarantuje monosemantyczności SAE ani
przyczynowej roli klastra. Oba widoki pochodzą z tego samego modelu i korpusu,
więc nie są niezależną replikacją.

## Katalog funkcji

| Moduł | Funkcja | Odpowiedzialność |
| --- | --- | --- |
| stability | `load_context_feature_analysis()` | macierz TF–IDF kontekstów przypisana do cech |
| stability | `label_sets()` / `best_jaccard()` | zbiory cech i najlepsze dopasowanie klastra |
| stability | `run_clustering_grid()` | uruchomienie wspólnej siatki wariantów |
| stability | `stability_for_baseline_clusters()` | agregacja stabilności baseline |
| stability | `write_report()` | raport Markdown z shortlistą |
| cross-view | `centroid_cohesion()` | kohezja klastra w macierzy wtórnej |
| cross-view | `benjamini_hochberg()` | kontrola FDR dla wielu klastrów |
| cross-view | `write_markdown()` | raport testów zgodności widoków |

## Ograniczenia

- HDBSCAN może zmieniać etykiety liczbowe; znaczenie ma zbiór cech, nie numer.
- Najlepszy Jaccard nie wymusza dopasowania jeden-do-jednego i może ukrywać
  split lub merge klastra.
- Siatka jest analizą wrażliwości, a nie pełnym przeszukaniem hiperparametrów.
- Test permutacyjny ocenia ponadlosową kohezję dla danej reprezentacji, nie
  poprawność ręcznie nadanej nazwy domeny.

