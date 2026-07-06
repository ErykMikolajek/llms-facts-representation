# Architektura Projektu

Ten dokument opisuje aktualną architekturę repozytorium `llms-facts-representation` oraz stan prac nad koncepcją **SAE-guided MoEfication** dla `roneneldan/TinyStories-1M`.

Główna idea: najpierw uczymy Top-K SAE na aktywacjach warstwy 4, potem grupujemy cechy SAE w domeny semantyczne, mapujemy te domeny na fizyczne neurony MLP i eksportujemy domenowe MLP-only eksperci przez trwałe zerowanie wag. Router i pełne MoE są jeszcze etapami przyszłymi.

## Widok Wysokiego Poziomu

```mermaid
flowchart LR
  rawData["TinyStories CSV"] --> sequencing["Dataset Sequencing"]
  sequencing --> tokenSeqs["Token Sequences"]
  tokenSeqs --> activationCollection["Activation Collection"]
  activationCollection --> layerActivations["Layer 4 Hidden States"]
  layerActivations --> saeTraining["TopK SAE Training"]
  saeTraining --> saeCheckpoint["SAE Checkpoint"]
  saeCheckpoint --> featureAnalysis["Feature Analysis"]
  saeCheckpoint --> domainTriage["Semantic Domain Triage"]
  domainTriage --> domainValidation["Domain Validation Sets"]
  domainValidation --> mlpMapping["MLP SAE Mapping"]
  mlpMapping --> pruning["Domain MLP Pruning"]
  pruning --> mlpExperts["Sparse MLP Experts"]
  mlpExperts --> futureMoE["Future Hard Routed MoE"]
```

## Obecne Warstwy Systemu

### 1. Dane I Tokenizacja

Plik: `dataset_sequencing.py`

Wejście:

- `data/tinystories_dataset/train.csv` albo `validation.csv`,
- kolumna `text`,
- tokenizer `EleutherAI/gpt-neo-125M`.

Wyjście:

- `data/tinystories_dataset/sequenced/tokens_seqs_padded.npy`,
- `data/tinystories_dataset/dataset_info.json`.

Proces:

```mermaid
flowchart TD
  csvInput["CSV text rows"] --> filterText["Filter empty text"]
  filterText --> sentenceSplit["Sentence Split"]
  sentenceSplit --> tokenize["Tokenizer"]
  tokenize --> packSeqs["Pack to 256 tokens"]
  packSeqs --> padSeqs["Pad sequences"]
  padSeqs --> tokensNpy["tokens_seqs_padded.npy"]
```

### 2. Ekstrakcja Aktywacji

Plik: `activations_collecting.py`

Ekstrahowane jest `outputs.hidden_states[layer_num + 1]`, czyli hidden state po bloku transformera. Dla warstwy 4 i TinyStories-1M ma wymiar `64`.

```mermaid
flowchart TD
  tokensNpy["tokens_seqs_padded.npy"] --> tokenDataset["TokenDataset"]
  tokenDataset --> baseModel["TinyStories 1M"]
  baseModel --> hiddenStates["outputs.hidden_states"]
  hiddenStates --> layer4["Layer 4 Hidden State"]
  layer4 --> activationMemmap["TEMP activations memmap"]
  activationMemmap --> activationsNpy["activations_layer_4.npy"]
```

Wyjście:

- `data/tinystories_dataset/activations/activations_layer_4.npy`,
- tymczasowy memmap i plik progress do wznawiania.

### 3. Top-K SAE

Plik: `autoencoder_training.py`

Architektura:

- wejście `d_model=64`,
- słownik `d_sae=4096`,
- aktywność Top-K z `k=8`,
- dekoder `W_dec` o kształcie `4096 x 64`,
- loss MSE rekonstrukcji hidden state.

```mermaid
flowchart LR
  layerVector["Hidden State 64D"] --> center["Subtract b_dec"]
  center --> encoder["Linear Encoder 64 to 4096"]
  encoder --> relu["ReLU"]
  relu --> topK["TopK k equals 8"]
  topK --> sparseActs["Sparse SAE Features"]
  sparseActs --> decoder["Decoder W_dec"]
  decoder --> recon["Reconstructed Hidden State"]
```

Wyjście:

- `topk_sae_layer_4.pt`,
- `topk_sae_layer_4_best.pt`,
- opcjonalne checkpointy krokowe.

### 4. Analiza Cech SAE

Plik: `features_analysis.py`

Ten etap analizuje aktywacje cech i logit lens.

```mermaid
flowchart TD
  saeCheckpoint["SAE Checkpoint"] --> sparseActs["Sparse Feature Activations"]
  tokenSeqs["Token Sequences"] --> contexts["Activation Contexts"]
  sparseActs --> stats["Activation Stats"]
  sparseActs --> examples["Top Examples"]
  saeCheckpoint --> logitLens["W_dec times W_U.T"]
  logitLens --> promotedTokens["Promoted Suppressed Tokens"]
  stats --> textReport["features_analysis.txt"]
  examples --> textReport
  promotedTokens --> textReport
  stats --> jsonReport["features_analysis.json"]
  examples --> jsonReport
  promotedTokens --> jsonReport
```

Aktualny stan po poprawkach:

- zapisuje raport tekstowy,
- zapisuje strukturalny JSON,
- używa parametru `layer_num`, a nie globalnej stałej,
- tworzy katalog `analysis/`, jeśli nie istnieje.

## Domeny Semantyczne

Plik: `semantic_domain_triage.py`

Cel: zgrupować cechy SAE w nadrzędne domeny przez podobieństwo logit-lens.

```mermaid
flowchart TD
  saeCheckpoint["SAE Checkpoint"] --> decoderDirs["SAE Decoder Directions"]
  baseModel["TinyStories LM Head"] --> unembedding["Unembedding Matrix"]
  decoderDirs --> logitLens["Logit Lens Matrix"]
  unembedding --> logitLens
  logitLens --> tokenFilter["Semantic Token Filter"]
  tokenFilter --> featureTokenMatrix["Feature Token Sparse Matrix"]
  featureTokenMatrix --> svd["Truncated SVD"]
  svd --> hdbscan["HDBSCAN"]
  hdbscan --> candidates["Cluster Candidates"]
  candidates --> orthogonalSelect["Orthogonal Domain Selection"]
  orthogonalSelect --> domainsJson["domains.json"]
  orthogonalSelect --> validationLexicons["Domain Lexicons"]
  validationLexicons --> validationSets["domain_validation JSONL"]
```

Artefakty:

- `feature_token_matrix.npz`,
- `logit_lens_top_tokens.jsonl`,
- `domains.json`,
- `feature_domain_assignments.csv`,
- `domain_report.md`,
- `domain_validation/domain_<id>.jsonl`,
- `domain_validation/all_domains_balanced.csv`,
- `domain_validation/validation_summary.json`.

Guardraile po poprawkach:

- częste tokeny funkcyjne są domyślnie odrzucane z reprezentacji domen,
- skrypt domyślnie wymaga co najmniej 3 wybranych domen,
- `domains.json` i raport zawierają diagnostykę: liczba klastrów, noise fraction, udział tokenów funkcyjnych w top tokenach,
- `--allow-underfilled-domains` pozwala zapisać słaby wynik tylko diagnostycznie.

## Dwa Rodzaje Mapowania

W repozytorium istnieją dwa poziomy mapowania i nie należy ich mylić.

### Mapowanie Residual Stream

Plik: `topographic_mlp_sae_mapping.py`

To etap eksploracyjny. Mimo nazwy zmienna `mlp_activations` oznacza tutaj 64-wymiarowy hidden state warstwy, czyli tę samą przestrzeń, w której trenowany był SAE.

```mermaid
flowchart LR
  domainText["Domain Text"] --> modelForward["Model Forward"]
  modelForward --> hiddenState["Layer Hidden State 64D"]
  hiddenState --> sae["SAE"]
  sae --> domainSignal["Aggregated SAE Domain Signal"]
  hiddenState --> corr64["Pearson MI over 64 dims"]
  domainSignal --> corr64
```

### Mapowanie Fizycznych Neuronów MLP

Plik: `domain_mlp_activation_mapping.py`

To właściwy etap dla pruning. Hook na wejściu `mlp.c_proj` przechwytuje post-aktywacje MLP, czyli fizyczne neurony wewnętrznej warstwy feed-forward.

```mermaid
flowchart TD
  domainValidation["domain_<id>.jsonl"] --> tokenizeBatch["Tokenize Batch"]
  tokenizeBatch --> modelForward["TinyStories Forward"]
  modelForward --> cProjHook["Hook on c_proj Input"]
  cProjHook --> postMlp["Post MLP Activations 256D"]
  modelForward --> layerHidden["Layer Hidden State 64D"]
  layerHidden --> saeForward["SAE Forward"]
  saeForward --> domainFeatures["Domain Feature IDs"]
  domainFeatures --> aggregateSignal["Sum Mean Max Domain Signal"]
  postMlp --> correlation["Pearson and Mutual Information"]
  aggregateSignal --> correlation
  correlation --> mappingCsv["MLP Neuron Correlations CSV"]
  postMlp --> mappingNpz["Domain MLP Activations NPZ"]
  aggregateSignal --> mappingNpz
```

Guardraile po poprawkach:

- mapping odrzuca domenę z pustym albo prawie pustym sygnałem SAE,
- `--allow-zero-domain-signal` pozwala zapisać taki przypadek wyłącznie diagnostycznie,
- summary zapisuje diagnostykę sygnału domenowego.

## Pruning Ekspertów MLP

Plik: `domain_mlp_pruning.py`

Wejście:

- `domain_mlp_mapping/domain_<id>_mlp_activations.npz`,
- `domain_mlp_mapping/domain_<id>_mlp_neuron_correlations.csv`.

Proces:

```mermaid
flowchart TD
  mappingNpz["MLP Activations and SAE Signal"] --> tau["Estimate Tau"]
  corrCsv["Neuron Correlations CSV"] --> scores["Neuron Scores"]
  tau --> keepMask["Build Keep Mask"]
  scores --> keepMask
  keepMask --> cFc["Zero c_fc Neuron Rows"]
  keepMask --> cProj["Zero c_proj Neuron Columns"]
  cFc --> expertState["MLP Expert State Dict"]
  cProj --> expertState
  expertState --> expertPt["domain_mlp_expert.pt"]
  keepMask --> maskNpy["domain_mask.npy"]
  keepMask --> metadata["domain_pruning.json"]
```

Metody progu:

- `per-domain-null` - permutuje sygnał domeny i bierze percentyl null distribution,
- `quantile` - bierze percentyl rzeczywistych score neuronów.

Guardraile po poprawkach:

- puste domeny domyślnie kończą się błędem,
- eksport eksperta z `0` zachowanymi neuronami jest blokowany,
- `--allow-empty-experts` służy tylko do diagnostyki,
- `--empty-signal-policy prune-all` bez `--allow-empty-experts` nie utworzy przypadkiem pustego eksperta.

## Aktualny Stan Zgodności Z SAE-Guided MoEfication

Zgodne albo częściowo zgodne:

- TinyStories-1M jako model bazowy,
- aktywacje warstwy 4,
- Top-K SAE `64 -> 4096`,
- logit lens,
- HDBSCAN na macierzy cecha-token,
- walidacje per domena,
- korelacja i MI między neuronami MLP a sygnałem domen SAE,
- pruning wag `c_fc` i `c_proj` do ekspertów,
- trening liniowego routera domenowego,
- składanie hard-routed MoE z pełnych MLP ekspertów,
- walidacja PPL dla bazy, pojedynczych ekspertów i MoE.

Niezgodne albo brakujące:

- obecne lokalne artefakty miały tylko 2 domeny zamiast 3-5,
- jedna domena miała zerowy sygnał SAE,
- pełny eksperyment MoE wymaga ponownego uruchomienia na 3-5 niepustych domenach,
- pierwsza wersja `HardRoutedMLP` jest poprawnościowa, ale nie zoptymalizowana wydajnościowo,
- eksperci są pełnymi MLP z wyzerowanymi wagami, a nie jeszcze materializowanymi rzadkimi podgrafami.

## Architektura MoE

Pierwsza wersja MoE składa się z trzech modułów: `router_training.py`, `moe_assembly.py` i `moe_validation.py`.

```mermaid
flowchart TD
  inputTokens["Input Tokens"] --> transformerPrefix["Transformer Blocks Before Layer 4 MLP"]
  transformerPrefix --> routerInput["Router Input Hidden State"]
  routerInput --> router["Linear Router"]
  router --> top1["Top1 Domain ID"]
  routerInput --> expertBank["Domain Expert Bank"]
  top1 --> expertBank
  expertBank --> selectedMlp["Selected Sparse MLP Expert"]
  selectedMlp --> transformerSuffix["Transformer Blocks After Layer 4 MLP"]
  transformerSuffix --> logits["LM Logits"]
```

### Router Training

Plik: `router_training.py`

Router jest liniowym klasyfikatorem domen:

- wejście: tensor przekazywany do `model.transformer.h[layer_num].mlp`,
- kształt wejścia: `[batch, seq_len, d_model]`, zwykle `[B, S, 64]`,
- wyjście: `[batch, seq_len, n_domains]`,
- etykieta: domena tekstu z `domain_validation/domain_<id>.jsonl`, rozciągnięta na wszystkie nie-padding tokeny.

Artefakty:

- `router/router.pt` - checkpoint z `router_state_dict`, `domain_ids`, `domain_names`, `d_model`, `n_domains`, `layer_num`,
- `router/router_metrics.json` - historia treningu i metryki walidacyjne,
- `router/router_report.md` - czytelny raport per domena.

### MoE Assembly

Plik: `moe_assembly.py`

Moduł ma dwa tryby:

- `build_single_expert_model()` podstawia jeden `domain_<id>_mlp_expert.pt` za MLP warstwy 4.
- `build_hard_routed_moe()` ładuje `router.pt`, bank ekspertów i zastępuje MLP wrapperem `HardRoutedMLP`.

`HardRoutedMLP` wykonuje Top-1 routing per token:

```mermaid
flowchart TD
  mlpInput["MLP input: B,S,D"] --> routerLinear["Linear Domain Router"]
  routerLinear --> routerLogits["Router logits: B,S,N"]
  routerLogits --> top1["Argmax per token"]
  mlpInput --> expertBank["Full MLP Expert Bank"]
  top1 --> masks["Token masks per expert"]
  expertBank --> expertOutputs["Expert outputs"]
  masks --> combine["Scatter combine"]
  expertOutputs --> combine
  combine --> mlpOutput["MLP output: B,S,D"]
```

Guardraile:

- router `d_model` musi zgadzać się z `model.config.hidden_size`,
- `layer_num` i `model_name` routera oraz ekspertów muszą być zgodne,
- eksperci diagnostycznie puste są blokowane bez jawnej flagi.

### PPL Validation

Plik: `moe_validation.py`

Walidacja liczy causal LM loss i perplexity dla:

- modelu bazowego na ogólnym `validation.csv` i per domena,
- pojedynczego eksperta na jego własnej domenie,
- hard-routed MoE na ogólnym `validation.csv` i per domena.

Artefakty:

- `moe_validation/ppl_results.json`,
- `moe_validation/ppl_report.md`.

## Przepływ Eksperymentalny Po Poprawkach

```mermaid
flowchart LR
  step1["Run main.py"] --> step2["Run semantic_domain_triage.py"]
  step2 --> checkDomains["Require 3 to 5 Nonempty Domains"]
  checkDomains --> step3["Run domain_mlp_activation_mapping.py"]
  step3 --> checkSignals["Require Nonzero SAE Signals"]
  checkSignals --> step4["Run domain_mlp_pruning.py"]
  step4 --> checkExperts["Require Nonempty Experts"]
  checkExperts --> routerStep["Run router_training.py"]
  routerStep --> moeStep["Run moe_assembly.py"]
  moeStep --> pplStep["Run moe_validation.py"]
```

## Znane Ryzyka

- HDBSCAN może zwrócić za mało klastrów przy zbyt dużym `min_cluster_size`.
- Nadmierne filtrowanie tokenów może odrzucić zbyt dużo cech, ale zbyt słabe filtrowanie daje domeny typu `the / it / one`.
- Domenowe walidacje są oparte na leksykonach top tokenów, więc mogą nie łapać parafraz.
- Prawdziwy pruning powinien używać `domain_mlp_activation_mapping.py`, a nie eksploracyjnego `topographic_mlp_sae_mapping.py`.
- Router jest trenowany tokenowo, ale etykiety pochodzą z poziomu tekstu, więc raport metryk trzeba interpretować ostrożnie.
- Pierwsza wersja `HardRoutedMLP` może liczyć kilku ekspertów na całym batchu, więc nie powinna być traktowana jako optymalizacja wydajności.
