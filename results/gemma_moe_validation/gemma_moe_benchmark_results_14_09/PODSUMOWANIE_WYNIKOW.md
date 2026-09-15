# Walidacja niezależna Gemma 3 270M MoE — wyniki i wnioski

Data analizy: 14 września 2026 r.  
Zakres: 600 zapieczętowanych przykładów, po 100 dla sześciu domen.  
Porównanie: niezmieniony model bazowy Gemma 3 270M kontra przygotowany
hard-routed MoE z sześcioma ekspertami MLP warstwy 9, zachowującymi po
1536/2048 neuronów, oraz gęstym fallbackiem.

## Najważniejszy wynik

Niezależny benchmark **nie wykazał poprawy kompetencji MoE względem modelu
bazowego**. Wspólna metryka teacher-forced NLL pogorszyła się w agregacji, a
wyniki zadaniowe pozostały zasadniczo bez zmian lub są zbyt słabe, by
potwierdzić korzyść. Jednocześnie eksperyment ujawnił wyraźny distribution
shift routera: na benchmarku niezależnym znacznie częściej używa fallbacku i
w kilku domenach kieruje pewne tokeny do niewłaściwego eksperta.

Nie jest to wynik negujący techniczną poprawność MoE. Poprzedni holdout PPL
pokazał zachowanie większości jakości przy warunkowym pruningu. Obecny test
pokazuje natomiast, że wyuczona specjalizacja **nie przełożyła się na poprawę
niezależnych zadań kompetencyjnych**.

![Porównanie wyników](gemma_moe_benchmark_results/benchmark_comparison.png)

## 1. Integralność i konfiguracja

W folderze znajdują się kompletne wyniki obu wariantów:

- `base_teacher_forced.json` i `moe_teacher_forced.json`;
- `base_generation.json` i `moe_generation.json`;
- `paired_comparison.json`;
- `domain_summary.csv` i `generation_summary.csv`;
- wykres oraz oryginalna paczka ZIP z Kaggle.

Oba modele oceniono z `batch_size=8` i `max_length=1024`. Żaden z 600 promptów
nie został ucięty. Porównanie NLL jest sparowane po `benchmark_id`; bootstrap
ma 2000 replik i seed 20260914 (sam dobór benchmarku używał seedu 20260915).
Benchmark jest od tej chwili **zużytym zbiorem
ewaluacyjnym**: nie należy dostrajać do niego masek, routera, progu confidence
ani promptów, a następnie raportować ponownej oceny na tych samych zadaniach
jako niezależnej.

SHA-256 najważniejszych plików:

| Plik | SHA-256 |
| --- | --- |
| `base_teacher_forced.json` | `7ec15ec79232a545bc719f96852f099d5872fbd8fc7a72501cbeeadf2cf278a9` |
| `moe_teacher_forced.json` | `3ff9230931df19fe1d834c854a1acab17ca5070d0eac5d35c69edb6591adf81f` |
| `paired_comparison.json` | `d978e13989344b2247a694c1352e994e1d60373f16b44655fc6c992fdd5172ec` |
| `base_generation.json` | `d5b8e51928e065872e270afcbe0c4d5015ff265c95b04e696a30d65abf088651` |
| `moe_generation.json` | `de525e2c4da949a7ab6f21ddff9bc957d694123e42d91a38949738dec42ae6af` |
| oryginalny ZIP | `daf12a80a91289b10893a9e3b990ffae296c5bb3e8ab76eabacd7f4a13d53728` |

## 2. Teacher-forced reference NLL

Niższe NLL jest lepsze. `Δ NLL = MoE − base`; dodatnia wartość oznacza
pogorszenie MoE. Przedziały ufności są bootstrapowane na poziomie przykładów.

| Domena | Base NLL | MoE NLL | Średnie sparowane Δ NLL | 95% CI | Δ PPL |
| --- | ---: | ---: | ---: | --- | ---: |
| prawo i orzecznictwo | 1,6758 | 1,6717 | −0,0040 | [−0,0139; 0,0055] | −0,40% |
| biomedycyna | 1,6388 | 1,6486 | +0,0097 | [+0,0003; +0,0226] | +0,98% |
| sport | 2,9589 | 3,0441 | +0,0841 | [+0,0385; +0,1360] | +8,90% |
| polityka i wiadomości | 1,5417 | 1,5364 | −0,0053 | [−0,0135; 0,0031] | −0,53% |
| matematyka / zapis LaTeX | 2,6649 | 2,7925 | +0,1430 | [+0,1191; +0,1669] | +13,61% |
| Python | 1,0875 | 1,1072 | +0,0215 | [+0,0151; +0,0281] | +1,98% |

Prawo i polityka mają niewielkie ujemne średnie, lecz ich przedziały obejmują
zero — dane są zgodne zarówno z małą poprawą, jak i małym pogorszeniem.
Biomedycyna, sport, matematyka i Python mają CI powyżej zera. Najsilniejsze
pogorszenie dotyczy matematyki, a następnie sportu.

Makrośrednia po 600 równoważnych przykładach wynosi
`Δ NLL = +0,04150`, 95% CI `[+0,03171; +0,05307]`; MoE uzyskał niższe NLL
tylko dla 32% przykładów. Przy agregacji ważonej liczbą tokenów odpowiedzi
NLL wzrasta z 1,20272 do 1,22709, a PPL z 3,3292 do 3,4113, czyli o 2,47%.
Tej agregacji nie należy utożsamiać z makrośrednią: domena Python ma 7483 z
8257 wszystkich tokenów odpowiedzi i dlatego dominuje wynik tokenowy.

!!! note "Wielokrotne porównania"
    Przedziały per domena nie zostały skorygowane za sześć równoległych testów.
    Globalne pogorszenie i duże efekty matematyki/sportu są wyraźne, ale
    granicznego wyniku biomedycyny nie należy nadinterpretować.

## 3. Multiple choice

| Domena | Base accuracy | MoE accuracy | Zmiana | Sparowane przejścia |
| --- | ---: | ---: | ---: | --- |
| prawo | 31% | 30% | −1 p.p. | 3 błędne→poprawne, 4 poprawne→błędne |
| biomedycyna | 23% | 26% | +3 p.p. | 3 błędne→poprawne, 0 poprawnych→błędne |
| sport | 49% | 49% | 0 p.p. | brak zmiany predykcji |
| polityka | 38% | 38% | 0 p.p. | brak zmiany poprawności |

Łącznie accuracy rośnie z 141/400 = 35,25% do 143/400 = 35,75%, czyli tylko
o 0,5 punktu procentowego. Dla prawa dokładny test sparowanych niezgodności nie
wskazuje różnicy (`p=1,0`), a dla biomedycyny `p=0,25`. Zysk biomedycyny jest
więc interesującym sygnałem eksploracyjnym, lecz nie dowodem poprawy.

W sporcie MoE wyraźnie pogarsza NLL odpowiedzi referencyjnej, ale ranking
odpowiedzi pozostaje identyczny dla wszystkich 100 pytań. W polityce zmieniła
się tylko jedna wskazana opcja, bez zmiany poprawności. Oznacza to, że pruning
zmienia kalibrację/prawdopodobieństwa silniej niż decyzje argmax.

## 4. Generacja matematyczna

GSM8K numeric exact match wynosi:

- model bazowy: 1/100 = 1%;
- MoE: 3/100 = 3%.

Były to trzy nowe trafienia MoE i utrata jedynego trafienia modelu bazowego;
żaden przykład nie był poprawny dla obu wariantów. Dokładny test sparowany
daje `p=0,625`. Oba wyniki leżą przy podłodze możliwości modelu 270M, dlatego
różnica dwóch zadań nie jest dowodem poprawy matematycznej. Jest też niespójna
z teacher-forced NLL, gdzie matematyka ma największe istotne pogorszenie.

Generacje często powtarzają treść problemu lub tworzą niedokończone
wyjaśnienia. Ostatnia rozpoznana liczba może pochodzić z powtórzonego promptu,
więc nawet numeric exact match jest słabą metryką jakości rozumowania dla
takiego modelu.

## 5. Generacja Pythona — błąd metryki w historycznym runie

Zapisany raport podaje spadek `python_syntax_valid` z 34% do 15%. Sparowane
przejścia to 27 `poprawna→błędna`, 8 `błędna→poprawna` i 7 poprawnych w obu;
formalnie daje to `p≈0,00188`. **Tego wyniku nie wolno jednak interpretować
jako rzeczywistego spadku poprawności składni**, ponieważ odkryto błąd
ekstrakcji kodu w ewaluatorze.

`extract_python_code()` wyszukuje pierwszy *kompletny* blok pomiędzy parą
potrójnych znaczników backtick.
Model zazwyczaj generuje najpierw surową funkcję, potem znacznik zamykający, a
następnie kolejny blok lub tekst. W takiej sytuacji parser często otrzymuje
prozę pomiędzy znacznikami zamiast pierwszej funkcji. Dodatkowo `ast.parse("")`
nie zgłasza błędu, więc pusty fragment bywa zaliczony jako składniowo poprawny.
Metryka jest zatem silnie zależna od formatowania fence, które samo zmieniło
się między base i MoE.

Diagnostyczna analiza post-hoc, która dla odpowiedzi zaczynającej się od
`def`/`class`/`import` bierze tekst **przed pierwszym znacznikiem zamykającym**,
wymaga niepustego AST i nie uruchamia kodu, daje:

| Wariant | Pierwszy fragment parsowalny | Przejścia sparowane |
| --- | ---: | --- |
| base | 96/100 | 87 wspólnych poprawnych |
| MoE | 91/100 | 4 błędne→poprawne, 9 poprawne→błędne |

Dokładny test tych niezgodności daje `p≈0,267`. Jest to wynik post-hoc, a nie
zamrożona metryka pierwotnego protokołu. Pokazuje jednak, że wartość 34%→15%
jest przede wszystkim artefaktem ekstraktora. Nawet 96%/91% oznacza wyłącznie
parsowalność pierwszego fragmentu — wiele funkcji jest semantycznie błędnych,
niedokończonych albo zwraca niewłaściwy wynik. Benchmark nie wykonuje testów
MBPP, więc **nie ma miary pass@1 ani dowodu kompetencji programistycznej**.

### Status naprawy (14.09.2026)

Błąd został naprawiony w `evaluation/moe_benchmark.py`. Ekstraktor najpierw zachowuje
surowy kod przed pierwszym zamykającym fence, następnie szuka niepustego bloku
`python`, a walidator wymaga co najmniej jednego elementu w AST. Dodano testy
regresyjne dla dokładnego formatu generowanego przez prompt MBPP. Starych liczb
34%/15% nie nadpisano, ponieważ dokumentują wykonany run v2; poprawiona metryka
jest częścią świeżego benchmarku v3.

## 6. Generalizacja routera

| Domena promptu | Routed | Do właściwego eksperta | Trafność wśród routowanych | Fallback | Efektywna szerokość MLP-9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prawo | 24,39% | 23,55% | 96,55% | 75,61% | 93,90% |
| biomedycyna | 5,12% | 3,81% | 74,40% | 94,88% | 98,72% |
| sport | 27,35% | 13,64% | 49,86% | 72,65% | 93,16% |
| polityka | 5,62% | 0,20% | 3,55% | 94,38% | 98,59% |
| matematyka | 43,96% | 43,96% | 100,00% | 56,04% | 89,01% |
| Python | 10,70% | 0,25% | 2,33% | 89,30% | 97,32% |

Łącznie na 58 710 tokenach promptów router wysłał do ekspertów 20,83%, z
czego 87,66% do eksperta zgodnego z etykietą domeny; fallback objął 79,17%.
Średnia efektywna szerokość MLP warstwy 9 wyniosła 94,79%, czyli rzeczywista
redukcja szerokości wykonywanej warstwy tylko 5,21%. Nie jest to redukcja
FLOPs całego modelu.

Agregat 87,66% jest myląco wysoki, bo dominuje go długi prompt prawny oraz
dobrze rozpoznana matematyka. Szczegółowo widać trzy problemy:

1. dla sportu 370 tokenów trafiło do matematyki, a 368 do sportu;
2. dla polityki 377 tokenów trafiło do prawa, a tylko 15 do polityki;
3. dla Pythona 377 tokenów trafiło do matematyki, a tylko 9 do Pythona.

To silny dowód distribution shift między źródłami development a niezależnym
benchmarkiem. Router nauczył się nie tylko domen, lecz także formatów źródeł:
SCOTUS/PubMedQA/DBpedia/MATH/stdlib różnią się od MMLU/BIG-bench/GSM8K/MBPP.
Confidence fallback chroni model przed częścią błędów, lecz jednocześnie
sprawia, że PoC wykorzystuje pruning rzadko i daje niewielką redukcję kosztu.

## 7. Oszczędność obliczeń podczas inferencji

### Najpierw zastrzeżenie o jakości

Nie można powiedzieć bezwarunkowo, że jakość się nie pogorszyła. Accuracy
multiple choice pozostało praktycznie takie samo, a zmiany generacyjnego exact
match są niejednoznaczne. Jednocześnie średnie teacher-forced NLL pogorszyło
się z bootstrapowym CI powyżej zera, szczególnie dla matematyki i sportu.
Uczciwe sformułowanie brzmi: **nie wystąpiła katastrofalna degradacja decyzji
zadaniowych, ale benchmark wykrył mierzalną regresję probabilistyczną**.

### Koszt jednego MLP

Gemma 3 270M ma `D=640`, `H=2048`, 18 warstw oraz gated MLP z trzema
projekcjami liniowymi: `gate_proj`, `up_proj` i `down_proj`. Pomijając mały
koszt GELU i mnożenia bramki, gęsty MLP wykonuje na token:

```text
C_dense = 3 × D × H
        = 3 × 640 × 2048
        = 3 932 160 MAC
```

MAC oznacza operację multiply–accumulate; przy konwencji liczącej mnożenie i
dodawanie oddzielnie odpowiada to około 7,86 mln FLOPs. Ekspert zachowuje
1536/2048 = 75% neuronów:

```text
C_expert = 3 × 640 × 1536 = 2 949 120 MAC
```

Dla tokenu, który rzeczywiście trafia do eksperta, redukcja kosztu MLP
warstwy 9 wynosi więc dokładnie 25%, czyli 983 040 MAC. Liniowy router `640→6`
dodaje około 3840 MAC na **każdy** token, niezależnie od fallbacku. Jest to
0,098% kosztu gęstego MLP tej warstwy.

### Średnia wynikająca z zaobserwowanego routingu

Na 58 710 tokenach promptów do ekspertów trafiło 12 231, czyli 20,83%.
Oczekiwany koszt zmodyfikowanego MLP wynosi zatem:

```text
C_MoE = (1-r) × C_dense + r × 0,75 × C_dense + D × 6
      = 3 731 204 MAC/token,  r = 0,2083
```

Bez routera odpowiada to efektywnej szerokości 94,79%; po doliczeniu routera
oszczędność arytmetyczna wynosi **5,11% kosztu MLP warstwy 9**. Dla
zaobserwowanych 58 710 prompt tokens byłoby to 230,86 mld → 219,06 mld MAC,
czyli około 11,80 mld MAC mniej, gdyby ten sam rozkład routingu reprezentował
pełny przebieg inferencji.

| Domena | Route rate | Oszczędność MLP-9 bez routera | Po koszcie routera |
| --- | ---: | ---: | ---: |
| prawo | 24,39% | 6,10% | 6,00% |
| biomedycyna | 5,12% | 1,28% | 1,18% |
| sport | 27,35% | 6,84% | 6,74% |
| polityka | 5,62% | 1,41% | 1,31% |
| matematyka | 43,96% | 10,99% | 10,89% |
| Python | 10,70% | 2,68% | 2,58% |
| **łącznie** | **20,83%** | **5,21%** | **5,11%** |

Nie widać korzystnego kompromisu jakość–compute dla wszystkich domen.
Matematyka daje największą lokalną oszczędność (10,89% MLP-9), ale zarazem
największą regresję PPL (+13,61%). Sport oszczędza 6,74% kosztu MLP-9 przy
+8,90% PPL. Prawo jest najbliżej korzystnego punktu — około 6,00% mniej MAC
MLP-9 bez wykrywalnej zmiany NLL — lecz CI nadal dopuszcza małą poprawę i małą
regresję. Polityka zachowuje jakość podobnie, ale oszczędza tylko około 1,31%
tego MLP. Nie można więc opisać wyniku jako jednolitej „bezpłatnej” redukcji
compute.

Jest to estymacja na podstawie telemetrii prompt-only. Ewaluator celowo nie
zliczał wielokrotnie routingu kandydatów multiple choice ani kolejnych kroków
generacji, więc 11,80 mld MAC nie jest pomiarem całej pracy wykonanej przez
notebook.

### Udział w całym modelu

Redukcja 5,11% dotyczy tylko jednego MLP. W modelu jest 18 takich warstw, a
poza nimi wykonywane są projekcje attention, operacje attention zależne od
długości kontekstu, normalizacje i projekcja do bardzo dużego słownika
`V=262144`.

Przybliżenie oparte tylko na mnożeniach macierzy daje:

| Mianownik porównania | Szacowana redukcja przy obecnym routingu |
| --- | ---: |
| tylko MLP warstwy 9 | 5,11% |
| wszystkie MLP w 18 warstwach | 0,284% |
| liniowe projekcje całego backbone’u | 0,200% |
| backbone + LM head dla generowanego tokenu | 0,075% |

Ostatnie dwie wartości są raczej górnymi granicami udziału, ponieważ rachunek
nie dodaje macierzy attention `QKᵀ`/`AV`, softmaxów, norm i operacji
pamięciowych. Z drugiej strony podczas prefilla, gdy `logits_to_keep` nie
materializuje głowy słownikowej dla każdego prompt tokenu, właściwszy jest
mianownik backbone’u niż „backbone + LM head”. Dokładny udział zależy więc od
stosunku prefilla do dekodowania i długości kontekstu.

Nawet gdyby **każdy** token trafiał do eksperta, obecna modyfikacja jednej
warstwy dawałaby po koszcie routera najwyżej około:

- 24,90% oszczędności MLP warstwy 9;
- 1,38% wszystkich obliczeń MLP modelu;
- 0,98% liniowych projekcji backbone’u;
- 0,37% projekcji liniowych wraz z LM head podczas generacji.

To pokazuje podstawowe ograniczenie architektury PoC: pruning jednej z 18
warstw nie może przynieść dużej oszczędności całego modelu, nawet przy
idealnym routerze.

### Compute nie oznacza automatycznie krótszego czasu

Powyższe wartości opisują **teoretyczną liczbę operacji**, nie zmierzone
przyspieszenie. Implementacja dzieli tensor indeksowaniem boolowskim, uruchamia
gęsty fallback i kilka mniejszych GEMM-ów sekwencyjnie, a następnie scala
wyniki. Na GPU małe i nieregularne macierze mogą mieć gorsze wykorzystanie
sprzętu niż jeden duży gęsty MLP. Przy marginesie około 5% w jednej warstwie
narzut dispatchu, alokacji i transferów może całkowicie znieść korzyść, a nawet
spowolnić model.

Rachunek MAC traktuje też wszystkie operacje jednakowo, podczas gdy router
działa w float32, a model bazowy używa BF16. Rzeczywisty koszt sprzętowy
routera może być zatem większy, niż sugeruje sam stosunek liczby MAC.

W dostarczonych wynikach nie ma pomiaru czasu ściennego, CUDA events, peak
memory ani energii. Nie można zatem twierdzić, że model jest empirycznie
szybszy lub tańszy. Można powiedzieć wyłącznie, że **warunkowy ekspert redukuje
arytmetyczny koszt zmodyfikowanego MLP, ale obecna konfiguracja daje bardzo
małą potencjalną oszczędność całego forwardu**.

### Pamięć parametrów

Runtime nie oszczędza też pamięci wag. Zachowuje gęsty MLP jako fallback i
dodaje sześć kompaktowych ekspertów. Eksperci zawierają łącznie około
17,69 mln dodatkowych parametrów, czyli około 35,4 MB w BF16 lub 70,8 MB w
FP32, nie licząc drobnego routera. To około 6,6% nominalnych 270 mln parametrów
modelu. Małe pliki masek (łącznie 13 056 bajtów) ograniczają rozmiar
**artefaktu na dysku**, ale po złożeniu eksperci mają własne skopiowane wagi.

Aktywacje pośrednie routowanego tokenu są o 25% węższe, więc lokalnie można
zmniejszyć ich objętość. Fallback dla większości tokenów oraz bufory
indeksowania sprawiają jednak, że bez profilera nie można zadeklarować spadku
peak VRAM.

### Odpowiedź praktyczna

Tak, można mówić o **teoretycznej oszczędności compute**:

> Dla tokenów skierowanych do eksperta koszt arytmetyczny MLP warstwy 9 maleje
> o 25%. Przy route rate 20,83% daje to około 5,11% redukcji MAC tej jednej
> warstwy po uwzględnieniu routera, lecz tylko około 0,20% liniowych obliczeń
> backbone’u i prawdopodobnie poniżej 0,1% pełnego kroku generacyjnego z głową
> słownikową.

Nie należy natomiast pisać, że uzyskano przyspieszenie inferencji. Do takiego
wniosku potrzebny jest osobny benchmark latency/throughput z warm-upem,
synchronizacją GPU, wieloma powtórzeniami i raportem peak VRAM.

## 8. Co można, a czego nie można wnioskować

### Uzasadnione wnioski

- MoE działa technicznie na niezależnych danych i nie powoduje katastrofalnej
  degradacji.
- Nie poprawia ogólnej jakości odpowiedzi referencyjnych; średnie NLL jest
  istotnie gorsze.
- Największa regresja występuje w matematyce i sporcie.
- Accuracy multiple choice jest praktycznie niezmienione.
- Router słabo przenosi się między źródłami danych, szczególnie dla polityki i
  Pythona.
- Fallback ogranicza ryzyko, ale redukuje potencjalny zysk obliczeniowy.
- Pierwotna metryka składni Pythona jest niewiarygodna z powodu błędu
  ekstrakcji; brak funkcjonalnego testu kodu.

### Wnioski nieuzasadnione

- Nie można twierdzić, że ekspert biomedyczny poprawia domenę na podstawie
  trzech dodatkowych odpowiedzi.
- Nie można twierdzić, że MoE poprawia matematykę na podstawie 1%→3% GSM8K.
- Nie można interpretować 34%→15% jako wiarygodnego spadku poprawności kodu.
- Nie można utożsamiać 25% pruningu eksperta z 25% przyspieszeniem modelu.
- Wyniki nie dowodzą, że wybrane neurony kodują wiedzę przyczynowo; mapping był
  korelacyjny.

## 9. Zalecane dalsze kroki

1. Zachować ten benchmark jako zużyty i nie stroić na nim finalnego modelu.
2. Naprawić `extract_python_code()` oraz wymagać niepustego AST. Przeliczenie
   zapisanych generacji jest korektą metryki, ale musi być raportowane jako
   post-hoc; nie wymaga ponownego uruchomienia modelu.
3. Dodać bezpieczny, izolowany runner MBPP/EvalPlus i raportować pass@1.
4. Dla routera przygotować development obejmujący wiele źródeł i formatów na
   domenę albo rozważyć dodatkową klasę „ogólne/OOD”.
5. Oddzielić routing tekstu od routingu tokenu: trenować z kontrastowymi
   fragmentami, neutralnymi tokenami lub poolingiem kontekstu.
6. Porównać kilka progów confidence wyłącznie na nowym development, pokazując
   krzywą coverage–quality–effective width.
7. Użyć 20–50 losowych masek, wielu seedów routera oraz bootstrapu różnicy
   loss per przykład.
8. Jeżeli celem jest wiedza domenowa, dodać interwencje przyczynowe i większy
   model, dla którego benchmark nie jest przy podłodze.
9. Dodać benchmark wydajności: prefill i autoregressive decode osobno,
   batch sizes 1/4/8, kilka długości kontekstu, co najmniej 20 powtórzeń po
   warm-upie, CUDA synchronization, tokens/s, latency i peak allocated VRAM.
10. Jeżeli oszczędność compute ma być celem, objąć pruningiem więcej warstw i
    użyć batched/grouped expert kernels; wymaga to nowego development i
    niezależnego testu jakości.

### Kroki wdrożone w follow-upie v3

Zrealizowano punkty możliwe bez ponownego trenowania modelu: naprawę ekstrakcji
Pythona (2), świeży zbiór testowy bez rekordów v1/v2, prostszą matematykę bez
efektu podłogi, trzy unikalne kolejności odpowiedzi MC z item-level bootstrapem
i dokładnym testem sparowanym oraz sprzętowy benchmark prefill dla batch 1/4/8
z warm-upem, synchronizacją CUDA, throughputem i peak VRAM (część punktów 7 i
9). Nie wdrożono wykonywania kodu MBPP, zmian routera, strojenia confidence,
pruningu wielu warstw ani specjalizowanych kerneli: wymagają izolowanego
runnera lub nowego developmentu i nie mogą być dobierane na zużytym v2.

## Konkluzja do pracy

Najbardziej defensywne podsumowanie brzmi:

> Domenowo warunkowany pruning pojedynczej warstwy Gemma 3 270M utworzył
> działający prototyp MoE, który na wcześniejszym holdoucie zachowywał znaczną
> część jakości modelu bazowego. Na niezależnym benchmarku nie uzyskano jednak
> poprawy kompetencji: średni reference-completion NLL uległ pogorszeniu,
> accuracy pozostało zasadniczo niezmienione, a router wykazał ograniczoną
> odporność na zmianę źródła i formatu danych. Wynik wspiera wartość pipeline'u
> jako demonstracji interpretowalnego, warunkowego pruningu, lecz nie stanowi
> dowodu przewagi jakościowej ekspertów ani przyczynowej lokalizacji wiedzy.
