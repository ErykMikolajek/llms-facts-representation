# Raport z triażu domen — Gemma 3 270M / Gemma Scope 2

Data wykonania: 14 września 2026 r.

Status: **triaż zakończony, domeny niezatwierdzone**. Mapowanie domen na neurony
MLP, pruning, trening routera i składanie MoE celowo nie zostały jeszcze
uruchomione. Następną bramką jest wybór domen przez użytkownika.

## 1. Cel i zakres

Celem triażu było znalezienie powtarzalnych grup cech SAE, których silne
aktywacje występują w tekstach o zbliżonej tematyce lub formie. Wynik nie jest
automatyczną ontologią wiedzy modelu. Jest listą kandydatów do dalszej walidacji
na niezależnych danych, a następnie do mapowania na komponenty MLP.

W tej fazie wykonano:

1. kontrolę kompletności checkpointu analizy;
2. porównanie dwóch reprezentacji cech: promowanych tokenów i kontekstów
   aktywacji;
3. redukcję wymiaru i klasteryzację w kilku konfiguracjach;
4. ocenę stabilności globalnej i stabilności każdego klastra;
5. diagnostyczny przegląd przykładów;
6. test zgodności klastrów z alternatywną reprezentacją cech.

Nie wykonano jeszcze pobierania danych dziedzinowych. Zbiory rozwojowy i
holdout zależą od ostatecznego wyboru domen; zostaną przygotowane automatycznie
dopiero po tej decyzji, z zapisem źródła, wersji, licencji, reguł filtrowania,
hashy i podziału danych.

## 2. Zweryfikowane wejście

Podstawą jest ukończony checkpoint:

`data/gemma_scope2_270m_pilecc/analysis/analysis_checkpoint_gemma_scope2_270m.pt`

- SHA-256: `02b1f3713d0b22f5f6879fdbec66691a8b9ca1ff02f0d4a4124cb68817ac43f5`;
- `completed = True`;
- 51 757 przetworzonych batchy;
- 103 513 z 103 513 sekwencji;
- 99 999 989 tokenów;
- 16 384 cechy SAE, w tym 16 343 zaobserwowane i 41 niezaobserwowanych;
- model: Gemma 3 270M;
- SAE: `gemma-scope-2-270m-pt-res`,
  `layer_9_width_16k_l0_medium`;
- punkt pomiaru: warstwa 9, `resid_post`.

Triaż czyta eksport `feature_analysis.json` wygenerowany z tego checkpointu.
Jego SHA-256 to
`e6b3c83cd671c3a7f8f19f40205570d5885cc02803d0d68ec292c7739dd23167`.
Sprawdzono także zgodność identyfikatorów tokenów z lokalnym tokenizerem
`models/gemma-3-270m`: w losowej próbie 10 000 zapisanych identyfikatorów nie
znaleziono różnic dekodowania. Hash słownika tokenizera to
`389ff14a24126d254022dd87bff5b5cedbda3a57d2de94afc47d17e381dcd5be`.

## 3. Dlaczego zmieniono pierwotną reprezentację

Najpierw pogrupowano cechy na podstawie tokenów promowanych przez kierunek
dekodera SAE po przemnożeniu przez macierz unembeddingu. Sprawdzono warianty
Top-128 z metodami HDBSCAN `eom` i `leaf` oraz Top-32 z `leaf`.

Warianty Top-128 dawały odpowiednio tylko 3–4 klastry albo około 85% szumu.
Top-32 dawał 28 klastrów i około 63% szumu, lecz dominowały w nim fragmenty
tokenów, języki, prefiksy i artefakty formatowania. Przykładowe centroidy były
zdominowane przez elementy takie jak `multi`, `single`, `pre` oraz fragmenty
pisma bengalskiego. Nie był to wiarygodny triaż tematyczny.

Jest to zgodne z ograniczeniem zastosowanego przybliżenia: kierunek cechy z
warstwy 9 jest rzutowany bezpośrednio na unembedding, z pominięciem dalszych
warstw modelu i końcowej normalizacji. Taki wynik opisuje przybliżony kierunek
wyjściowy, a nie pełny skutek cechy na predykcję modelu. Reprezentację tę
zachowano jako widok pomocniczy, ale odrzucono jako podstawę triażu.

## 4. Reprezentacja użyta do właściwego triażu

Dla każdej zaobserwowanej cechy SAE utworzono jeden dokument przez połączenie
jej zapisanych kontekstów o najwyższej aktywacji. Znaczniki wyróżniające token
`[[...]]` usunięto, aby nie tworzyły sztucznego sygnału.

Dokumenty zamieniono na macierz TF-IDF:

- małe litery i normalizacja akcentów Unicode;
- tylko słowa zawierające litery, o długości co najmniej dwóch znaków;
- usunięcie angielskich stop words;
- `min_df = 3`, aby odrzucić jednostkowe artefakty;
- `max_df = 0.20`, aby odrzucić słowa obecne w ponad 20% cech;
- `sublinear_tf = True` i normalizacja L2;
- maksymalnie 50 000 elementów słownika;
- do opisu jednej cechy zachowano 64 terminy o największej wadze.

Otrzymano niepuste wektory dla 16 342 z 16 384 cech. Jest to reprezentacja
leksykalna kontekstów, a więc identyfikuje współwystępowanie z domeną, nie
udowadnia natomiast, że pojedyncza cecha koduje pojęcie dziedzinowe.

## 5. Redukcja wymiaru i klasteryzacja

Macierz TF-IDF redukowano algorytmem Truncated SVD, a następnie klasteryzowano
HDBSCAN z metodą wyboru `leaf`. Wykonano siedem przebiegów:

| przebieg | SVD | ziarno | min. klaster | min. próbek | klastry | szum | ARI wszystkie cechy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bazowy | 50 | 0 | 40 | 10 | 23 | 79,1% | 1,000 |
| SVD-30 | 30 | 0 | 40 | 10 | 21 | 78,2% | 0,746 |
| SVD-80 | 80 | 0 | 40 | 10 | 24 | 79,8% | 0,750 |
| ziarno 1 | 50 | 1 | 40 | 10 | 23 | 80,7% | 0,866 |
| ziarno 2 | 50 | 2 | 40 | 10 | 23 | 79,3% | 0,862 |
| min. klaster 20 | 50 | 0 | 20 | 10 | 40 | 79,7% | 0,887 |
| min. klaster 80 | 50 | 0 | 80 | 20 | 16 | 80,9% | 0,782 |

Pierwsza wersja raportu pokazywała tylko ARI policzone na przecięciu cech
uznanych w obu przebiegach za nie-szumowe. Wartości 0,975–0,999 były poprawne
dla tej warunkowej próby, ale nie opisywały przejść między klastrem a szumem.
Kod poprawiono: raport pokazuje teraz zarówno ARI dla wszystkich cech, jak i
ARI dla wspólnie sklasteryzowanych cech.

Duży udział szumu nie jest błędem HDBSCAN. Oznacza, że przy zadanych progach
około 79–81% cech nie należy do dostatecznie gęstej i powtarzalnej grupy.
Tych cech nie wolno automatycznie przypisać do najbliższej domeny.

## 6. Ocena stabilności pojedynczego klastra

Dla każdego klastra bazowego znaleziono klaster o największym nakładaniu w
każdym z sześciu przebiegów alternatywnych. Nakładanie zmierzono indeksem
Jaccarda. Obliczono także:

- spójność: średni cosinus wektorów cech do znormalizowanego centroidu klastra;
- wynik bazowy: `spójność * log(1 + liczba cech)`;
- wynik stabilizowany: `wynik bazowy * średni najlepszy Jaccard`.

Ranking jest narzędziem porządkowania kandydatów. Nie jest testem istotności ani
automatycznym kryterium akceptacji domeny.

## 7. Kandydaci warci rozważenia

| proponowana etykieta | klaster | cechy | spójność | Jaccard śr. | Jaccard min. | ocena |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| prawo i orzecznictwo | 20 | 161 | 0,276 | 0,889 | 0,802 | bardzo stabilny kandydat tematyczny |
| biomedycyna | 17 | 396 | 0,222 | 0,837 | 0,745 | stabilny, szeroki kandydat tematyczny |
| sport | 16 | 117 | 0,290 | 0,759 | 0,692 | czytelny kandydat tematyczny |
| polityka i wiadomości | 22 | 123 | 0,255 | 0,696 | 0,526 | użyteczny, ale słabiej stabilny |
| matematyka / zapis LaTeX | 15 | 322 | 0,321 | 0,893 | 0,839 | bardzo stabilny, lecz częściowo formalny |
| narracja i dialog | 21 | 789 | 0,222 | 0,809 | 0,592 | stabilny, ale szeroki i stylistyczny |
| hotele i recenzje | 18 | 153 | 0,307 | 0,774 | 0,281 | czytelny, lecz wrażliwy na jedną konfigurację |
| Python | 5 | 149 | 0,315 | 0,718 | 0,577 | dobry pozytywny test, domena formalna |
| Java / Android | 6 | 115 | 0,414 | 0,699 | 0,365 | czytelny, ale niestabilny w części przebiegów |
| HTML | 3 | 142 | 0,379 | 0,747 | 0,628 | domena formatu bardziej niż wiedzy |

Nie rekomenduje się klastra 13 mimo wysokiego miejsca w rankingu. Jego terminy
łączą przepisy, biologię, informatykę i inne tematy, a minimalny Jaccard wynosi
tylko 0,193. Jest to przykład, dlaczego wynik liczbowy nie może zastąpić
przeglądu semantycznego.

## 8. Diagnostyczny przegląd kontekstów

Dla każdego kandydata zapisano po pięć kontekstów o wysokich aktywacjach.
Potwierdzają one czytelność m.in. domen prawa, biomedycyny, sportu, polityki,
recenzji hotelowych, matematyki i kodu.

Przykłady pochodzą jednak z tej samej analizy, która posłużyła do znalezienia
klastrów, i zostały wybrane na podstawie słownictwa klastra. Są przydatne do
debugowania i nadania roboczej nazwy, ale nie są niezależnym zbiorem
walidacyjnym i nie mogą służyć do raportowania jakości routera.

## 9. Walidacja w drugim widoku

Członkostwa klastrów z kontekstów zamrożono i obliczono ich spójność w macierzy
Top-128 promowanych tokenów. Dla każdego klastra utworzono 1000 losowych grup
cech o tym samym rozmiarze, losowanych bez zwracania z niepustych wierszy
drugiej macierzy. Zastosowano jednostronne empiryczne p-value z poprawką
`+1` oraz korektę FDR Benjamini–Hochberga dla 23 testów.

Po korekcie większość interesujących kandydatów tematycznych nie odróżnia się
od losowych grup w reprezentacji logit lens (`q > 0,27`). Istotne pozostają
głównie klastry językowo-formatowe oraz niejednorodny klaster 13. Drugi widok
nie potwierdza więc tematycznej struktury, ale też jej nie obala, ponieważ obie
reprezentacje mierzą inne własności cechy. Wynik traktowany jest jako negatywna
kontrola i powód, by wymagać niezależnej walidacji aktywacyjnej.

## 10. Rekomendowany wybór do proof of concept

Najlepszy kompromis między zgodnością z tematem pracy, stabilnością i kosztem
eksperymentu daje zestaw czterech domen specjalistycznych:

1. **prawo i orzecznictwo** — klaster 20;
2. **biomedycyna** — klaster 17;
3. **sport** — klaster 16;
4. **polityka i wiadomości** — klaster 22.

Do architektury należy dodać eksperta ogólnego/fallback, którego nie należy
utożsamiać z jednym z klastrów HDBSCAN. Jako łatwiejszą kontrolę pozytywną można
dodatkowo wybrać matematykę/LaTeX (15) albo Python (5). Taka kontrola może
jednak zawyżać ocenę routera, ponieważ rozpoznanie notacji lub składni jest
łatwiejsze niż rozpoznanie domeny semantycznej.

## 11. Co stanie się po zatwierdzeniu domen

Po wyborze domen zostanie wykonana osobna faza z bramkami jakości:

1. automatyczne przygotowanie niezależnych korpusów rozwojowych i holdout;
2. kontrola licencji, pochodzenia, duplikatów i przecieku między splitami;
3. uruchomienie Gemmy i SAE na nowych tekstach;
4. pomiar selektywności cech domenowych względem wszystkich pozostałych domen
   i zbioru ogólnego, wraz z bootstrapowymi przedziałami ufności;
5. mapowanie zaakceptowanych cech SAE na neurony MLP jako osobny, empiryczny
   problem — bez utożsamiania indeksu cechy SAE z indeksem neuronu;
6. pruning warstwowy z krzywą budżetu i ablacjami, a nie jedną arbitralną
   wartością progu;
7. trening routera wyłącznie na zbiorze rozwojowym;
8. porównanie bazowego modelu, ekspertów, MoE i kontroli losowych na zamrożonym
   holdoucie;
9. zapis wszystkich konfiguracji, ziaren, hashy oraz metryk jakości i kosztu.

## 12. Ograniczenia badawcze bieżącego wyniku

- Triaż jest eksploracyjny i korzysta z ekstremalnych kontekstów zapisanych
  podczas jednej analizy korpusu.
- Klastry opisują współaktywację cech z tekstami, nie lokalizują jeszcze
  przyczynowo wiedzy faktograficznej.
- Badana jest jedna warstwa i jeden SAE; nie sprawdzono zgodności między
  warstwami ani innymi poziomami rzadkości SAE.
- Wysoki odsetek szumu ogranicza pokrycie przestrzeni cech, ale chroni przed
  wymuszonymi przypisaniami.
- Etykiety domen są nadane po analizie słownictwa i kontekstów; muszą być
  potwierdzone na nowych, wcześniej niewidzianych tekstach.
- Centroidy pełnych wektorów TF-IDF kandydatów mają wysokie podobieństwa
  międzydomenowe (około 0,67–0,85), dlatego nie są dowodem ortogonalności
  ekspertów.

## 13. Odtworzenie

Właściwy triaż:

```bash
.venv/bin/python -m domain_triage.semantic_domain_stability \
  --feature-analysis data/gemma_scope2_270m_pilecc/analysis/feature_analysis.json \
  --tokenizer-name models/gemma-3-270m \
  --layer-num 9 \
  --feature-representation contexts \
  --output-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_research_contexts_20260914
```

Walidacja cross-view:

```bash
.venv/bin/python -m domain_triage.semantic_domain_cross_view \
  --primary-results data/gemma_scope2_270m_pilecc/analysis/domain_triage_research_contexts_20260914/stability_results.json \
  --secondary-matrix data/gemma_scope2_270m_pilecc/analysis/domain_triage_research_20260914/feature_token_matrix.npz \
  --secondary-label promoted_tokens_top128_tfidf \
  --output-json data/gemma_scope2_270m_pilecc/analysis/domain_triage_research_contexts_20260914/cross_view_results.json \
  --output-report data/gemma_scope2_270m_pilecc/analysis/domain_triage_research_contexts_20260914/cross_view_report.md \
  --permutations 1000 \
  --seed 0
```

Najważniejsze artefakty:

- `stability_results.json` — pełne wyniki maszynowe;
- `stability_report.md` — tabela wszystkich 23 klastrów;
- `shortlist.json` — członkostwa cech w kandydatach;
- `candidate_contexts/` — przykłady wyłącznie diagnostyczne;
- `cross_view_results.json` i `cross_view_report.md` — test w drugim widoku.
