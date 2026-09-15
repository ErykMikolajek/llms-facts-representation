# `sae_pipeline/gemma_scope_analysis.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/sae_pipeline/gemma_scope_analysis.py) · [wspólne sekwencjonowanie](dataset_sequencing.md) · [następny etap: domeny](semantic_domain_triage.md)

## Rola modułu

Moduł realizuje osobny tor analizy gotowego SAE Gemma Scope 2 dla
`google/gemma-3-270m`. Nie trenuje autoenkodera. Ładuje model językowy i
JumpReLU SAE przez SAELens, przechwytuje `resid_post` warstwy 9 i buduje
ograniczone pamięciowo karty cech.

Domyślna konfiguracja:

| Pole | Wartość |
| --- | --- |
| model/tokenizer | `google/gemma-3-270m` |
| release SAELens | `gemma-scope-2-270m-pt-res` |
| SAE ID | `layer_9_width_16k_l0_medium` |
| warstwa/site | `9`, `resid_post` |
| długość sekwencji | `1024` |

Alias release'u kończący się `-resid_post` jest zamieniany na aktualny klucz
`-res`. Kod obsługuje SAELens ≤5 zwracający tuple oraz v6 zwracający SAE.

## Kontrakt model–SAE

Po załadowaniu `_validate_sae_model_contract()` sprawdza:

- `cfg.d_in == model.config.hidden_size`;
- jeśli metadata zawiera `hook_name`, musi on być równy
  `blocks.<layer>.hook_resid_post`;
- docelowa warstwa musi rzeczywiście istnieć.

Metadane `d_in`, `d_sae`, nazwa modelu, hooki i ustawienia normalizacji są
zapisywane w summary. Summary zawiera też rozmiar słownika tokenizera oraz
stabilny SHA-256 jego mapowania `token → token_id`; triage może dzięki temu
wykryć zmianę semantyki ID mimo tej samej liczby tokenów. To ważne, ponieważ
SAE z niewłaściwego site'u albo analiza z innym tokenizatorem mogą mieć zgodne
wymiary, ale inne znaczenie reprezentacyjne.

## Przygotowanie danych

`_prepare_sequences_if_needed()` najpierw szuka jawnych `tokens_path` i
`attention_mask_path` albo standardowych plików w `<data_path>/sequenced`.

- Jeżeli oba istnieją, używa ich bez kopiowania.
- Jeśli użytkownik podał którąkolwiek ścieżkę lub `require_prepared=True`, brak
  pliku jest błędem i tokenizacja nie zostanie uruchomiona.
- W innym przypadku wymagane jest `input_path`, a dane są przygotowywane przez
  `prepare_sequences_from_jsonl()` ze wspólnego modułu.

## Pozyskiwanie `resid_post`

`iter_resid_post_batches()` zachowuje kształt `[B,S,D]`, ponieważ analiza
kontekstów potrzebuje współrzędnych sekwencji. Dla każdego podanego zakresu:

1. czyta tokeny i maskę z memmap;
2. rejestruje hook na wyjściu konkretnego bloku;
3. uruchamia sam backbone z `use_cache=False`;
4. przenosi tylko jeden batch hidden states na CPU;
5. zwraca `(start, end, hidden, tokens, mask)`;
6. usuwa hook w `finally`.

W przeciwieństwie do trenera Top-K SAE hidden states nie są tutaj od razu
spłaszczane na stałe; robi to akumulator z zachowaniem mapy pozycji.

## `FeatureAccumulator` — przebieg batcha

1. Maska daje współrzędne wszystkich ważnych tokenów i długości sekwencji.
2. `hidden[mask]` tworzy `[N_valid,D]`, dzielone następnie na mikrobatch'e
   `sae_batch_size`.
3. Dla każdego mikrobatcha wykonywane jest `sae.encode()` w inference mode.
4. `torch.nonzero(encoded > activation_threshold)` zwraca wyłącznie aktywne
   pary `(token, feature)`, co omija pętlę po całym `F` dla nieaktywnych cech.
5. Dla każdej pary aktualizowane są wektory `counts`, `sums`, `maxima`, bounded
   licznik tokenów oraz heap najmocniejszych przykładów.
6. Płaski indeks jest mapowany z powrotem do wiersza i pozycji, więc kontekst
   nigdy nie przechodzi przez granicę sekwencji.

## Ograniczony licznik tokenów

Każda zaobserwowana cecha przechowuje najwyżej `max_token_entries` tokenów.
Gdy pojawia się nowy token przy pełnym liczniku, usuwany jest element o
najmniejszym liczniku, a nowy otrzymuje `min_count + 1`. Jest to wariant
algorytmu Space-Saving: pamięć jest ograniczona, ale liczby rzadkich elementów
są przybliżone i mogą być zawyżone. `top_k_tokens` wybiera potem najczęstsze z
tej ograniczonej struktury.

## Sampling, checkpoint i resume

Zakresy `head`, `tail` i `uniform` mają tę samą semantykę co w analizie Pythii;
uniform tworzy rozłączne okna zawierające dokładnie żądaną liczbę sekwencji.

Checkpoint `gemma_scope_analysis_checkpoint_v1` przechowuje wektory statystyk,
liczniki tokenów, heap'y, total tokens oraz kursor zakresu i sekwencji. Zapis
jest atomowy. Przy wznowieniu ścieżki mountów mogą się zmienić, jeśli nazwy
plików tokenów i maski pozostają te same; pozostała konfiguracja akumulatora
musi być identyczna.

### Finalizacja przerwanego checkpointu

`--finalize-checkpoint-only` wywołuje `finalize_analysis_checkpoint()` i
materializuje raporty bez wznawiania forwardów oraz bez dostępu do pierwotnych
memmap. Funkcja sprawdza format checkpointu i zgodność kształtów `counts`,
`sums`, `maxima`, odtwarza przybliżone liczniki tokenów i heapy przykładów, a
następnie buduje karty wszystkich cech.

Jeżeli logit lens nie jest wyłączony, finalizacja nadal musi załadować zgodny
model i SAE, sprawdzić `d_sae` i policzyć promowane/tłumione tokeny. Bez logit
lens potrzebny jest tylko tokenizer do dekodowania trigger tokens. Summary
jawnie zapisuje `analysis_completed`, `finalized_from_checkpoint` oraz
`finalized_from_partial_checkpoint`; dla takiej rekonstrukcji pełna liczba
sekwencji i liczba przeanalizowanych sekwencji są `None`, a zachowany jest
ostatni kursor i liczba batchy. Częściowego raportu nie wolno przedstawiać jako
pełnej analizy.

## Logit lens

Tylko zaobserwowane cechy są rzutowane blokami:

```text
logits_f = SAE.W_dec[f] @ model_output_embeddings.T
```

Dla każdej zachowywane jest do `logit_top_k` największych i najmniejszych
wartości z ID i tekstem tokenu. Domyślnie `128` elementów zapewnia wystarczająco
bogate wejście dla późniejszego triage.

!!! warning "RMSNorm Gemmy"
    Jest to `approximate_resid_decoder_to_unembedding`. Kierunek dekodera nie
    przechodzi przez końcową, zależną od wejścia RMSNorm, więc wartości nie są
    dokładnymi efektami interwencji na finalnych logitach.

## Główna funkcja `analyze()`

1. Waliduje batch sizes i interwały.
2. Wybiera urządzenie, ładuje tokenizer i przygotowuje/odnajduje memmapy.
3. `_model_dtype()` wybiera float32 na CPU/MPS; na CUDA bf16, jeśli karta go
   wspiera, w przeciwnym razie float16. Można wymusić dtype argumentem.
4. Ładuje model i SAE oraz sprawdza ich kontrakt.
5. Waliduje dokładny kształt tokenów `[N,seq_length]` i maski.
6. Inicjalizuje lub odtwarza akumulator.
7. Przechodzi od kursora po zakresach i batchach, zapisując checkpoint co
   `checkpoint_every_batches` i na końcu zakresów.
8. Opcjonalnie wykonuje logit lens.
9. Buduje kartę każdej z `F` cech, summary i trzy raporty.
10. Zapisuje końcowy checkpoint z `completed=True`.

## Artefakty

| Plik | Zawartość |
| --- | --- |
| `analysis/analysis_checkpoint.pt` | resumowalny stan analizy |
| `analysis/feature_cards.jsonl` | tylko cechy zaobserwowane, jedna karta na linię |
| `analysis/feature_analysis.json` | summary i wszystkie cechy, również niezaobserwowane |
| `analysis/feature_analysis.txt` | czytelny raport tylko cech zaobserwowanych |

Liczba pojedyncza `feature_analysis.*` odróżnia te artefakty od plików Pythii
`features_analysis.*`. `domain_triage/semantic_domain_triage.py` obsługuje oba formaty.

## Katalog funkcji i metod

| Element | Odpowiedzialność |
| --- | --- |
| `_json_safe()` | serializuje skalary NumPy, tensory i zagnieżdżone struktury |
| `_atomic_torch_save()` / `_load_torch_checkpoint()` | bezpieczny zapis i kompatybilne ładowanie |
| `_checkpoint_config_matches()` | kontrola resume tolerująca relokację mountu |
| `_load_sae()` | importuje SAELens, normalizuje alias release'u i wersje API |
| `_sae_metadata()` | odczytuje istotne pola `cfg` i metadata |
| `_validate_sae_model_contract()` | sprawdza wymiar, site i istnienie warstwy |
| `_model_dtype()` | dobiera dtype do urządzenia lub jawnego żądania |
| `_load_model()` | ładuje model z kompatybilnością nazw `torch_dtype`/`dtype` |
| `_analysis_ranges()` | tworzy zakresy próbkowania |
| `_prepare_sequences_if_needed()` | wybiera gotowe sekwencje albo uruchamia JSONL sequencing |
| `iter_resid_post_batches()` | hookuje blok i zachowuje układ sekwencja–token |
| `FeatureAccumulator.__init__()` | alokuje bounded statystyki szerokości SAE i konfigurację mikrobatchy |
| `FeatureAccumulator._decode_token()` | cache dekodowania pojedynczych ID |
| `_update_bounded_counter()` | przybliżony Space-Saving dla trigger tokens |
| `_context()` / `_update_example()` | kontekst i bounded top examples |
| `process_batch()` | mikrobatching SAE i aktualizacja rzadkich statystyk |
| `_feature_card()` / `cards()` | materializuje rekord jednej/wszystkich cech |
| `active_feature_ids()` | indeksy `counts > 0` |
| `state_dict()` / `load_state_dict()` | pełny resumowalny stan bounded accumulatora |
| `_compute_logit_lens()` | blokowy decoder-to-unembedding dla aktywnych cech |
| `_write_text_report()` | raport tekstowy |
| `finalize_analysis_checkpoint()` | tworzy oznaczone raporty z pełnego lub częściowego checkpointu bez streamingu aktywacji |
| `analyze()` | kompletny pipeline Gemma Scope |
| `_parser()` / `main()` | CLI i mapowanie argumentów do `analyze()` |

## Ograniczenia i warunki interpretacji

- Karty cech opisują wyłącznie analizowany sample; gotowy SAE nie dostarcza
  lokalnego `usage_counts` z treningu, więc brak obserwacji nie oznacza martwej cechy.
- Liczniki trigger tokens są przybliżone po przekroczeniu pojemności.
- Checkpoint nie utrwala dtype ani ustawień końcowego logit lens. Można wznowić
  te same akumulatory i wygenerować końcowe tokeny logit lens z innym
  `logit_top_k`; należy zapisać finalne summary razem z wynikami.
- Finalizacja częściowego checkpointu zachowuje wszystkie zebrane statystyki,
  ale brakujące fragmenty korpusu mogą zmienić częstości, top tokeny i
  reprezentatywne konteksty.
- Forward wykonuje wszystkie bloki backbone'u, choć analizowany jest jeden hook.

## Wykonany przebieg Gemma 3 270M

Finalny checkpoint użyty w eksperymencie jest ukończony. Obejmuje 99 999 989
tokenów w 103 513 sekwencjach i 16 343 zaobserwowane cechy z 16 384. Logit
lens został sfinalizowany z listami wystarczającymi do triażu. Wcześniejszy
raport z 31 068 456 tokenów był smoke testem częściowego checkpointu i nie jest
źródłem sześciu ostatecznych domen. Dalszy przebieg opisuje
[strona eksperymentu](gemma_moe_experiment.md).
