# Router i MoE

Trening routera domenowego, składanie `HardRoutedMLP`, zamrażanie protokołu
holdout i tworzenie weryfikowalnego bundle runtime. Eksperci są kompaktowymi
wariantami pojedynczej warstwy MLP, a nie niezależnie trenowanymi modelami.

```bash
python3 -m moe.router_training --help
python3 -m moe.moe_assembly --help
```
