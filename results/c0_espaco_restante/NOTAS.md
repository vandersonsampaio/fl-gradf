# C.0 — espaço restante: notas de interpretação

**Data:** 2026-09-30 · Descritivo (roadmap §5, C.0), sementes exploratórias 42–51, framework próprio (MNIST, 10 clientes, 2 bizantinos, 15 rodadas, root 100).
**Arquivos:** `analise.txt` / `espaco_restante_por_celula.csv` (teto = FedAvg sem ataque) e `analise_teto_corrigido.txt` / `espaco_restante_teto_corrigido.csv` (teto = melhor das 5 regras sem ataque; gap também contra regras fixas sob ataque). Scripts: `scripts/c0_espaco_restante.py`, `scripts/c0_teto_corrigido.py`.

## 1. O teto é o FedAvg sem ataque em todos os alphas

Acurácia sem ataque (média de 10 sementes):

| α | FedAvg | FLTrust | Krum | Median | Trimmed-Mean |
|---|---|---|---|---|---|
| 0,05 | **0,854** | 0,721 | 0,328 | 0,616 | 0,765 |
| 0,10 | **0,855** | 0,757 | 0,546 | 0,740 | 0,831 |
| 0,50 | **0,905** | 0,828 | 0,884 | 0,896 | 0,899 |

A correção **não mudou o teto**. A hipótese que motivou a correção ("o FedAvg não converge em 15 rodadas e o Krum avança mais rápido") **estava errada**: sem ataque, o Krum é o pior de todos em α ≤ 0,1 (0,33 e 0,55).

## 2. As duas células com gap negativo são artefato do ataque, não falta de convergência

Em `fltrust_aligned`, com α = 0,05 e α = 0,1, o Krum **sob ataque** chega a ~0,895, acima do teto (0,854), embora sem ataque fique em 0,33–0,55. Isso só se explica se o próprio ataque ajuda o Krum. O `fltrust_aligned` é um ataque informado que alinha o update malicioso ao update do servidor no root; o Krum escolhe esse update (o mais "central"), que carrega informação do dataset raiz e funciona como um passo quase centralizado. **Essas 2 células não medem robustez** e devem ficar fora do mapa de espaço restante (ou ser tratadas como um caso patológico do ataque nesse regime).

## 3. Mapa do espaço restante (gap global = teto − melhor método existente, incluindo regras fixas)

- **> 5 p.p. (4/21):** `label_flipping` α=0,05 e 0,1 (~14 p.p.), `sign_flipping` α=0,05 (12 p.p.), `low_mag_backdoor` α=0,05 (7 p.p.).
- **2–5 p.p. (6/21):** `gaussian_noise`, `krum_collusion`, `trim_attack` em α=0,05 (~4,9); `sign_flipping` α=0,1 (4,6); `trim_attack` α=0,1 (3,6); `low_mag_backdoor` α=0,1 (2,4).
- **≤ 2 p.p. (9/21):** praticamente todo α=0,5 (0,5–1,9), mais `gaussian_noise` e `krum_collusion` em α=0,1.
- **Excluídas (artefato, §2):** `fltrust_aligned` α=0,05 e 0,1.

Contra o **esqueleto** (fixed/sr_only/AdaAggRL) o espaço é maior em 3 células onde uma regra fixa clássica já faz melhor: `label_flipping` α=0,5 (Trimmed-Mean reduz o gap de 13,1 para 1,9 p.p.), `low_mag_backdoor` α=0,1 (6,1 → 2,4) e α=0,5.

## 4. Leitura para o Portão C0

- O espaço restante **não é pequeno em todo lugar**: está concentrado em **alta heterogeneidade (α ≤ 0,1)** e em **ataques de rótulo e sinal** (`label_flipping`, `sign_flipping`), exatamente onde o S_R tem sinal fraco (Paper 1). Em α = 0,5 quase tudo está a ≤ 2 p.p. do teto.
- Implicação para o GRADF-v2 (R1/R2): há ganho de acurácia possível **nas células de alta heterogeneidade com ataques que o sinal por cliente não detecta**. Fora delas, o diferencial teria de vir de custo e privacidade (R3/R4).
- O roadmap não fixou um limiar numérico para "pequeno"; os cortes de 1/2/5 p.p. acima são descritivos. **A decisão do Portão C0 é do autor.**
- Limitação: 15 rodadas, MNIST, 10 clientes; o teto de FedAvg sem ataque em α ≤ 0,1 (0,85) é baixo em termos absolutos, então "espaço restante" aqui é relativo a esse regime curto.

---

## 5. Critério do Portão C0 (declarado em 2026-09-30, ANTES de rodar o teto oráculo)

Conforme `references/c0_fechamento.md`, Passo 1. Registro local com hash em `NOTAS.sha256`, sem commit (o autor decidiu não commitar neste momento).

- **Teto oráculo:** FedAvg sem ataque agregando **só os 8 clientes honestos** (índices 2–9; os bizantinos são os índices 0 e 1 em `byzantine_ids`), com as **mesmas partições** do C.0 (sem reparticionar). Métrica idêntica à do C.0: acurácia do modelo global final média sobre os `X_test` dos **10** clientes. α ∈ {0,05; 0,1; 0,5} × sementes 42–51 × 15 rodadas = 30 runs.
- **Célula com espaço:** gap contra o teto oráculo **> 2 p.p.** com **IC95 inteiro acima de 0** (limiar fixado aqui).
- **Células excluídas:** `fltrust_aligned` em α = 0,05 e α = 0,1 (artefato do ataque, §2). Restam 19 células válidas.
- O critério é aplicado ao **gap global** (teto oráculo − melhor método existente, incluindo regras fixas). O gap do esqueleto é reportado ao lado, sem critério próprio.
- Previsão registrada (não é critério): espaço grande em `label_flipping` α ≤ 0,1 e `sign_flipping` α = 0,05; limítrofe em `sign_flipping`/`trim_attack` α = 0,1 e `low_mag_backdoor` α = 0,05; no teto em `gaussian_noise`, `krum_collusion`, `trim_attack` α = 0,05 e quase todo α = 0,5.

---

## 6. Decisão do Portão C0 (2026-09-30, aplicação do critério da §5)

Hash da §5 conferido antes da aplicação: `d0e07ea0…bc55d` (igual ao de `NOTAS.sha256`). Resultado completo em `analise_teto_oraculo.txt` e `espaco_restante_teto_oraculo.csv`; script `scripts/c0_teto_oraculo.py`.

### 6.1 Aplicação mecânica do critério

**Células COM espaço (2/19):** `label_flipping` α = 0,1 (gap 8,0 p.p., IC95 4,0–12,0) e `label_flipping` α = 0,05 (6,0 p.p., IC95 5,5–6,5). → **alvo do R1** (acurácia) no GRADF-v2.

**Células SEM espaço (17/19):** todas as demais. → diferencial do GRADF-v2 nelas é **R3/R4** (custo e privacidade com desempenho igual). Caso limítrofe: `sign_flipping` α = 0,05 (gap 4,2 p.p., mas IC95 −0,4 a 8,7).

**Esqueleto contra regra clássica** (o melhor método da célula é uma regra fixa, não o esqueleto): `label_flipping` α = 0,5 (Trimmed-Mean; gap do esqueleto 12,4 p.p. contra 1,3 do global), `low_mag_backdoor` α = 0,1 e 0,5 (Trimmed-Mean), `fltrust_aligned` α = 0,5 (Median). → motiva o **R2** e a direção 1 do C.2 (filtro por cliente com memória seguido de regra robusta); resultado **obrigatório no P2**: o esqueleto domina só onde o S_R separa os clientes.

**Previsão da §5 contra resultado:** `label_flipping` α ≤ 0,1 confirmou; `sign_flipping` α = 0,05 **não** (IC cruza 0); o "no teto" previsto para α = 0,05 e α = 0,5 se confirmou.

### 6.2 Ressalva: o teto oráculo não é teto em α ≤ 0,1 (achado, não mudança de critério)

| α | teto oráculo (8 honestos) | FedAvg sem ataque (10) |
|---|---|---|
| 0,05 | 0,772 | 0,854 |
| 0,10 | 0,790 | 0,855 |
| 0,50 | 0,899 | 0,905 |

Em α ≤ 0,1, retirar os 2 clientes custa ~7–8 p.p.: com heterogeneidade forte, eles têm classes pouco representadas nos demais, e o teste (IID) cobra isso. Por isso **10 das 17 células "sem espaço" têm gap negativo com IC95 inteiro abaixo de 0** (até −5,4 p.p.): métodos sob ataque superam o "teto", porque vários ataques do simulador (`gaussian_noise`, `trim_attack`, `krum_collusion`…) partem do update verdadeiro do atacante e ainda carregam parte da informação dos dados dele.

Consequências, declaradas:
- **As 2 células com espaço são robustas:** `label_flipping` α ≤ 0,1 tem espaço contra os dois tetos (8,0/6,0 p.p. contra o oráculo; 14,6/14,2 contra o FedAvg com 10).
- **Em α = 0,5 a classificação "sem espaço" é confiável:** os dois tetos quase coincidem (0,899 contra 0,905) e os gaps são ≤ 1,3 p.p. em ambos.
- **Em α ≤ 0,1, "sem espaço" não está demonstrado** para as 7 células além de `label_flipping`: nenhum dos dois tetos é um limite superior válido ali (o FedAvg com 10 superestima o alcançável sob ataque; o oráculo com 8 subestima). O critério foi aplicado como congelado; **reinterpretar essas células é decisão do autor** e pediria um teto que preserve a informação dos dados dos atacantes (por exemplo, um "oráculo de ataque": os mesmos clientes enviando o update honesto), a registrar antes de rodar.

### 6.3 Limitações declaradas
- Com 15 rodadas, o gap mistura robustez com velocidade de convergência → resolver no **B2.7** (incluir as células do C.0 na grade de horizonte).
- **ASR do `low_mag_backdoor` não foi logado** (nem na ablação nem no exp9): a acurácia limpa pode esconder um backdoor bem-sucedido. Registrado como limitação.
- Seleção do Krum no `fltrust_aligned`: não logada; confirmação do artefato fica para a Fase C (roadmap: C.3).

→ **Portão C0 fechado pelo critério congelado**, com a ressalva da §6.2 sobre α ≤ 0,1 pendente de decisão do autor.
