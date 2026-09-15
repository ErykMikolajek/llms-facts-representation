# `domain_mapping/validate_domain_sae_selectivity.py`

## Cel bramki

Przed pruningiem skrypt sprawdza, czy suma aktywacji cech SAE wybranego klastra
rzeczywiście odróżnia teksty jego domeny na danych development. Jest to
bramka falsyfikacyjna: nieselektywny klaster nie powinien produkować eksperta
tylko dlatego, że otrzymał przekonującą nazwę z top tokenów.

## Jednostka analizy

`pooled_mlp_activations.npz` zawiera sygnały per token oraz `sample_ids`.
`aggregate_token_signals_by_sample()` liczy średnią sygnału w obrębie tekstu.
Metryki i bootstrap działają więc na tekstach, a nie na silnie skorelowanych
tokenach traktowanych jako niezależne obserwacje.

Dla każdej domeny obliczane są:

- ROC-AUC i average precision one-vs-rest;
- standaryzowana różnica średnich (SMD);
- bootstrapowy 95% CI AUC i AP;
- pairwise AUC wobec każdej innej domeny.

SMD używa skończonej regularyzacji mianownika przy zerowej wariancji. Dzięki
temu ścisły JSON nie zawiera niestandardowych `Infinity`/`NaN`.

## Wynik bieżącego eksperymentu

Wszystkie sześć domen przeszło warunek `AUC >= 0.60` i dolna granica 95% CI
większa niż 0.50. AUC wyniosły 0,998; 1,000; 0,999; 0,957; 0,999 i 0,995.
Najmniejszy pairwise AUC, 0,808, dotyczył polityki kontra prawo.

## Artefakty i funkcje

| Element | Zawartość / odpowiedzialność |
| --- | --- |
| `selectivity_results.json` | konfiguracja, wyniki domen, pary i globalne `all_passed` |
| `domain_selectivity.csv` | płaski rekord per domena |
| `pairwise_auc.csv` | pełne porównania domen parami |
| `selectivity_report.md` | raport do inspekcji |
| `standardized_mean_difference()` | efekt standaryzowany ze stabilnym edge case |
| `bootstrap_binary_metrics()` | resampling tekstów i percentylowe CI |
| `evaluate_selectivity()` | wszystkie metryki i decyzja bramki |

## Interpretacja

Selektywność predykcyjna nie jest przyczynowością. Wynik może rozpoznawać
styl datasetu, format LaTeX lub składnię Pythona. Dopiero kontrolowany pruning,
baseline'y i niezależny benchmark mówią, czy wybrane neurony zachowują
funkcję związaną z domeną.

