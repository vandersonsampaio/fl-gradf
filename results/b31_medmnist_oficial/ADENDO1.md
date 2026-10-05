# Adendo 1 — B3.1 + B3.2: ataques, margem, desvios do BloodMNIST e código

**Data:** 2026-10-05, depois do B3.0 e antes de qualquer run do B3.1. Ordem do `PREREGISTRO.md` §8: pré-registro (`315b880`) → B3.0 → este adendo (commit) → disparo.

## 1. Resultado do B3.0 (regras do §3, aplicadas mecanicamente)

Detalhes em `results/b30_medmnist_sanity/RESULTADO.md`.
- **Convergência:** atendida, no limite. T_Blood = 77,83%; drift final de +1,79 p.p. (< 2).
- **Ataques incluídos:** **LMP e EB** (quedas de 56,9 e 60,8 p.p. no FedAvg).
- **Margem:** **M = ±3,00 p.p.** (fórmula: 7,19 p.p.; limitada ao teto de 3,0). **Declaração do §3:** a equivalência nessa margem é **fraca**. O BloodMNIST tem erro de base ~7× o do MNIST no mesmo regime oficial.
- **Grade:** 2 condições × 2 ataques × 10 sementes = **40 runs**, intercalados.

## 2. Desvios da adaptação ao BloodMNIST (PREREGISTRO §2: "listados no adendo do B3.0")

1. **Clientes:** **96** em vez de 100. O código oficial cria `int(num_clients / num_class)` clientes por grupo de classe, o que dá 12 × 8. `subsample_rate` = 10/96, mantendo **10 clientes por rodada**; 20 atacantes.
2. **Passos locais:** cada cliente tem ~124 imagens de treino, ou seja, **1 batch de 64 por época local** (`drop_last` oficial), contra ~9 no MNIST.
3. **Escala da recompensa:** a perda é somada sobre ~53 batches de teste (contra ~156 no MNIST), então o limiar oficial de reset (recompensa < −80) fica relativamente mais difícil de atingir. Mantido como no código oficial.
4. **Extrator:** `MNISTClassifier` com 3 canais e 8 saídas, treinado centralmente no treino do BloodMNIST (Adam 1e-3, 15 épocas, semente 0; 90,5% no teste), congelado. Hash abaixo.
5. **Dados:** `.npz` oficial do MedMNIST v2 (Zenodo 10519652), sem o pacote `medmnist`, para não alterar as versões do venv oficial. Normalização por canal; aumentos como no ramo MNIST oficial.
6. **Caminho MNIST intacto:** o shim só age quando `dataset == "BloodMNIST"`.

## 3. Métrica e agregação

Ver `ADENDO0_metrica_agregacao.md`: a acurácia é medida no test set global (desbalanceado no BloodMNIST), e a agregação do AdaAggRL não usa tamanho de amostra.

## 4. Execução

- **Lançador:** `scripts/passo2_oficial/run_grid_b31.sh "LMP EB" 6`. A fila é intercalada (td3/fixed alternados por semente e ataque) e o lançador grava o PGID.
- **Janela de execução** (regra do autor): `scripts/janela_execucao.sh` suspende às 7h e retoma às 18h, de segunda a sexta. **Exceção de 05/10:** o autor pediu que a execução continue hoje; por isso o controlador do B3.1 é ativado só às 18h de 05/10.
- **Custo estimado:** ~8–11 h por run com 6 processos (B3.0), ou seja, 40 runs em ~7 levas, ~60–75 h de GPU. Com a janela, o fim previsto fica por volta do fim de semana de 10–11/10.

## 5. Código e artefatos no congelamento
91451db16a765b51197f8ec066912712cfbbb097df496b95069881ac9d0beb3b  scripts/passo2_oficial/bloodmnist_shim.py
89a86ebef9ead31341426bc2a18cde05e6412e0eeeaf4ab094b3e9cc6e0655b8  scripts/passo2_oficial/run_b3.py
1602c9486ce11abf9c4391c3f01e008f55c56ab3443b6d1acfe821d4410d73f0  scripts/passo2_oficial/run_grid_b31.sh
dff2220a1ee52664e7d9149f57df7a089536f8c82040ae974798616d4fbda81d  scripts/passo2_oficial/analisar_b31.py
2253b777eff9b192964a80c6ff2b75b7055c148e720a8083458eccb4cae68451  scripts/passo2_oficial/treinar_extrator_bloodmnist.py
c1546628896d91af7c5c77bb0fb58c95878dab846e0ee4ae8c99bb81c303a203  scripts/janela_execucao.sh
3e154c823ed1a75ac125de4bc4ebc509934f783c5d2e3c42daa35403600b33a1  data/models/extract_feature_bloodmnist.pt
062023e186f537e26b3c21ea3b2614ddfc475e8a14825dfd20663bbb7e37bddc  data/raw/medmnist/bloodmnist.npz
