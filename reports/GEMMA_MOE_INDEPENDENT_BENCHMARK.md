# Niezależny benchmark Gemma 3 270M: model bazowy kontra MoE

Data przygotowania i aktualizacji: 14.09.2026. Aktualny follow-up: `independent_benchmark_v3_sealed`.

> **Status wersji:** dalsza część dokumentu do sekcji 8 opisuje historyczny,
> zużyty benchmark v2 i pozwala odtworzyć pierwszą walidację. Aktualny protokół
> v3 opisano poniżej; nie wolno mieszać jego assetów z wynikami v2.

## 0. Aktualizacja v3 po pierwszej walidacji

V3 naprawia błąd ekstrakcji kodu Python, usuwa efekt podłogi GSM8K i mierzy
wrażliwość MC na pozycję odpowiedzi. Ma 600 nowych rekordów: po 100 na domenę,
z czego 500 to multiple choice, a 100 to MBPP. Żaden `benchmark_id` nie pokrywa
się z v1/v2. SHA-256 pliku to
`297973c3460ff6a7205a6811c316c737c1946f0a9eb7cbc3b0aecf06c69b5fe5`.

- matematyka: po 25 prostych dodawań, odejmowań, mnożeń i dokładnych dzieleń,
  z zapisem LaTeX; accuracy zamiast GSM8K exact match;
- polityka: 34 pytania `high_school_government_and_politics`, 33
  `security_studies` i 33 `us_foreign_policy`, ponieważ po wcześniejszych
  runach pierwszy podzbiór miał tylko 46 niezużytych rekordów;
- MC: trzy unikalne, deterministyczne kolejności opcji (dwie dla sportu),
  wariantowa accuracy, stabilność semantycznej predykcji, bootstrap CI i
  dokładny test sparowany;
- Python: kod przed pierwszym zamykającym fence ma pierwszeństwo, puste AST nie
  jest poprawne; testów MBPP nadal nie wykonuje się w notebooku;
- wydajność: osobny prefill dla batch 1/4/8, warm-up, synchronizacja CUDA,
  mediana/kwartyle, tokens/s i peak allocated VRAM. To nie jest benchmark
  decode, FLOP ani energii.

Aktualne artefakty to `kaggle/notebooks/kaggle_gemma_moe_benchmark.ipynb` oraz
`kaggle/assets/gemma_moe_benchmark_v3.zip`. Generator używa seedu `20260916`
i automatycznie wyklucza zużyte identyfikatory oraz powtórzone treści sportowe.

## 1. Cel i status metodologiczny

Benchmark służy do sparowanego porównania niezmodyfikowanej Gemmy 3 270M z domenowym MoE warstwy 9. Żaden przykład z wariantu finalnego nie był używany podczas odkrywania domen, mapowania SAE–MLP, pruningu, treningu routera, kalibracji progu ani wcześniejszej ewaluacji PPL.

„Niezależność” ma tutaj precyzyjne, ograniczone znaczenie: repozytoria źródłowe są inne niż SCOTUS, PubMedQA, DBpedia14, MATH, lokalny CPython i WikiText, a hash żadnego wyrenderowanego przykładu nie pokrywa się z dotychczasowymi zbiorami projektu. Nie można natomiast zagwarantować, że popularne publiczne benchmarki nie występowały w danych pretrainingowych Gemmy.

Pierwszy wariant `independent_benchmark_v1` został użyty do technicznego smoke testu po jednym przykładzie na domenę. Choć nie wykonano na jego podstawie żadnego strojenia, został konserwatywnie wycofany. Finalny wariant utworzono z innym seedem (`20260915`) już po sprawdzeniu kodu ewaluatora i nie uruchamiano na nim modelu lokalnie.

## 2. Skład benchmarku

| Domena | Źródło | Konfiguracja | Liczba | Typ zadania | Główna metryka zadaniowa |
|---|---|---|---:|---|---|
| prawo i orzecznictwo | MMLU | `professional_law` | 100 | wybór A–D | accuracy z warunkowego log-likelihood |
| biomedycyna | MMLU | `anatomy`, `clinical_knowledge`, `medical_genetics`, `professional_medicine` | 100, po 25 | wybór A–D | accuracy z warunkowego log-likelihood |
| sport | BIG-bench | `sports_understanding` | 100, klasy zbalansowane | plausible/implausible | accuracy z warunkowego log-likelihood |
| polityka i wiadomości | MMLU | `high_school_government_and_politics` | 100 | wybór A–D | accuracy z warunkowego log-likelihood |
| matematyka / zapis LaTeX | GSM8K | `main/test` | 100 | generacja liczby | numeric exact match |
| Python | MBPP | `full/test` | 100 | generacja kodu | poprawność składni; NLL kodu referencyjnego |

Łącznie benchmark zawiera 600 przykładów: 400 pytań wyboru, 100 zadań arytmetycznych i 100 zadań programistycznych.

Źródła i rewizje:

- `cais/mmlu`: `c30699e8356da336a370243923dbaf21066bb9fe`, MIT;
- `EleutherAI/bigbench`: `f975b5fd41084e0f90085b98292430e8d56609dc`, Apache-2.0;
- `openai/gsm8k`: `740312add88f781978c0658806c59bc2815b9866`, MIT;
- `google-research-datasets/mbpp`: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, CC-BY-4.0.

Użycie tego samego formatu MMLU dla prawa, biomedycyny i polityki częściowo redukuje konfuzję „domena kontra format zbioru” między tymi trzema domenami. Nie usuwa jej dla sportu, matematyki i Pythona, które z natury mają inne formaty zadań.

## 3. Sposób próbkowania i uszczelnienie

Przykłady wybrano deterministycznie z oficjalnych części testowych, po uprzednim przypięciu pełnych hashy rewizji. Sport został zbalansowany do 50 zdań prawdopodobnych i 50 nieprawdopodobnych. Biomedycyna zawiera równą liczbę przykładów z czterech poddziedzin.

Skrypt sprawdza:

1. unikalność `benchmark_id`;
2. unikalność znormalizowanych treści;
3. brak repozytoriów wspólnych z wcześniejszym pipeline;
4. brak bezpośredniego przecięcia hashy treści;
5. dokładnie 100 przykładów na domenę;
6. hash SHA-256 finalnego `benchmark.jsonl`.

Jeżeli katalog benchmarku już istnieje i nie jest pusty, skrypt odmawia jego nadpisania. Nowa iteracja wymaga nowego katalogu i nowego seedu, co zapobiega przypadkowemu „odświeżaniu” benchmarku po zobaczeniu wyników.

## 4. Metryki

### 4.1. Teacher-forced reference-completion NLL

Dla każdego przykładu obliczany jest ujemny log-likelihood wyłącznie tokenów odpowiedzi, warunkowany promptem. Jest to jedyna wspólna metryka dla wszystkich sześciu domen. Notebook zapisuje:

- sumę NLL;
- średni NLL na token odpowiedzi;
- tokenowo ważony NLL i perplexity;
- różnicę sparowaną `MoE − base` dla każdego przykładu;
- średnią różnicę i bootstrapowe 95% CI na poziomie przykładów.

Wartość ujemna różnicy przemawia za MoE. Bezwzględnych NLL nie należy porównywać między domenami, ponieważ odpowiedzi mają różną postać i długość — od jednej litery po całe funkcje Python.

Ewaluator korzysta z argumentu Gemmy `logits_to_keep` i materializuje logity tylko dla pozycji potrzebnych do oceny odpowiedzi. Dodatkowy budżet dynamicznie ogranicza iloczyn liczby przykładów i długości odpowiedzi w batchu. Jest to istotne przy dużym słowniku Gemmy i zapobiega tworzeniu wielogigabajtowego tensora `[batch, długość całego promptu, słownik]` na T4. Test porównawczy potwierdził numeryczną zgodność z pełnym obliczeniem logitów.

### 4.2. Multiple choice

Dla prawa, biomedycyny, sportu i polityki model ocenia każdą dopuszczalną odpowiedź. Wybierana jest odpowiedź o najmniejszym średnim NLL. Ta procedura nie zależy od swobodnego generowania i jest stabilniejsza dla małego, bazowego modelu niż parsowanie wygenerowanej litery.

Raportowane są accuracy modelu bazowego, accuracy MoE, ich sparowana różnica oraz bootstrapowe 95% CI.

### 4.3. GSM8K

Oprócz NLL finalnej odpowiedzi notebook wykonuje deterministyczną generację i porównuje ostatnią rozpoznaną liczbę z odpowiedzią referencyjną po normalizacji separatorów. Jest to surowsza i bardziej zadaniowa metryka niż perplexity.

### 4.4. MBPP

Notebook mierzy NLL kompletnego kodu referencyjnego i sprawdza, czy wygenerowany tekst daje się sparsować przez `ast.parse`. Poprawność składni nie jest równoznaczna z poprawnością funkcjonalną.

Testy MBPP są zachowane w benchmarku, ale notebook ich nie wykonuje. Uruchamianie arbitralnego kodu wygenerowanego przez model w procesie notebooka byłoby ryzykowne. Rzetelne pass@1 należy później obliczyć w odizolowanym kontenerze, np. za pomocą EvalPlus, bez sekretów i dostępu do sieci.

## 5. Routing i koszt warstwy

Telemetria routingu jest liczona na promptach, po jednym przejściu każdego przykładu. Nie wykorzystuje wielokrotnie powtórzonych promptów powstających podczas oceny wszystkich wariantów odpowiedzi multiple-choice.

Dla każdej domeny notebook raportuje:

- odsetek tokenów wysłanych do dowolnego eksperta;
- odsetek wysłany do eksperta zgodnego z etykietą domeny;
- odsetek fallbacku do oryginalnego MLP;
- efektywną szerokość MLP warstwy 9: `fallback + 0,75 × routed`.

Nie jest to pomiar przyspieszenia całego modelu. Notebook uruchamia bazę i MoE sekwencyjnie, bez rozbudowanego warm-upu i wielu powtórzeń, dlatego czasu ściennego nie należy traktować jako wiarygodnego benchmarku wydajności.

## 6. Notebook Kaggle

Notebook: `kaggle/notebooks/kaggle_gemma_moe_benchmark.ipynb`.

Paczka danych i kodu: `kaggle_gemma_moe_benchmark_assets_v2.zip`.

Paczka zawiera:

- finalny `benchmark.jsonl` i jego manifest;
- sześć masek boolowskich ekspertów;
- `pruning_summary.json`;
- router i zamrożony próg confidence;
- kod składania MoE i ewaluatora;
- `asset_manifest.json` z rozmiarami i hashami wszystkich zależności runtime.

Notebook dodatkowo weryfikuje hashe czterech kluczowych plików bazowego checkpointu Gemmy. Dzięki temu model bazowy i MoE na pewno korzystają z tych samych wag.

Typ numeryczny jest dobierany konserwatywnie: `bfloat16` wyłącznie na GPU z natywnym wsparciem BF16 (np. L4/A100), a `float32` na T4/P100 i CPU. Wymuszenie `float16` na T4 może prowadzić do nieciągłych logitów Gemmy. Ewaluator zatrzymuje się teraz przy pierwszym takim wyniku z identyfikatorem spłaszczonej pary, zamiast dopuścić `NaN` do raportu. JSON jest serializowany przed zapisem i atomowo zastępuje plik docelowy, więc błąd nie pozostawia częściowego wyniku.

Procedura uruchomienia:

1. utworzyć prywatny Kaggle Dataset z rozpakowanego `kaggle_gemma_moe_benchmark_assets_v2.zip`;
2. podpiąć ten Dataset do notebooka;
3. podpiąć Kaggle Model `google/gemma-3/transformers/gemma-3-270m/2` i zaakceptować warunki Gemmy;
4. wybrać GPU T4, L4 albo A100;
5. uruchomić wszystkie komórki bez zmiany promptów, seedu, progu i masek;
6. pobrać `/kaggle/working/gemma_moe_independent_benchmark_results.zip`.

Notebook zapisuje pełne wyniki per przykład, podsumowania domen, wykres z 95% CI oraz osobne wyniki generacyjne.

## 7. Reprodukcja przygotowania

Finalny benchmark:

```bash
.venv/bin/python -m evaluation.prepare_independent_moe_benchmark \
  --experiment-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --output-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/independent_benchmark_v2_sealed \
  --samples-per-domain 100 --seed 20260915
```

Paczka Kaggle:

```bash
.venv/bin/python kaggle/prepare_kaggle_moe_benchmark_assets.py \
  --experiment-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --benchmark-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/independent_benchmark_v2_sealed \
  --model-dir models/gemma-3-270m --workspace-root . \
  --output-dir kaggle_gemma_moe_benchmark_assets_v2
```

## 8. Ograniczenia i interpretacja

1. Gemma 3 270M jest małym modelem bazowym; niska dokładność zadań może wywołać efekt podłogi i utrudnić wykrycie różnic.
2. Publiczna kontaminacja pretrainingu pozostaje nieznana.
3. Benchmark nadal częściowo splata domenę z formatem zadania.
4. NLL mierzy preferencję odpowiedzi referencyjnej, nie zawsze poprawne rozumowanie.
5. Brak bezpiecznego pass@1 dla MBPP ogranicza wniosek o jakości programowania.
6. Sto przykładów na domenę wystarcza do PoC, ale daje szerokie przedziały ufności dla małych różnic accuracy.
7. Wynik korzystny tylko w NLL, bez poprawy accuracy lub exact match, nie powinien być opisywany jako wzrost kompetencji domenowej.

Najbardziej uczciwy wniosek będzie zależał od zgodności kilku sygnałów: różnicy NLL, accuracy/exact match, bootstrapowego CI oraz poprawności routingu. Benchmark ma sprawdzić zachowanie i kompetencję gotowego MoE, a nie dostarczyć kolejnej pętli strojenia.
