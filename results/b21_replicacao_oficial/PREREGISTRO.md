# Pré-registro — B2.1 + B2.2: replicação confirmatória no AdaAggRL oficial

**Status:** FINAL, congelado antes de qualquer execução da grade (hash em `PREREGISTRO.sha256`).
**Data:** 2026-09-27
**Roadmap:** `references/roadmap_tese_gradf_v2.md` §4, experimentos B2.1 (confirmatório) e B2.2 (hipóteses mecanísticas, confirmatório junto com B2.1).
**Antecedente:** Passo 2 (`results/frente1_passo2_oficial/`, sementes 100–104): H1 inconclusiva por um reset tardio; política TD3 exploratoriamente independente da entrada.
**Registro:** local, com hash; o commit fica a critério do autor. Mudanças depois do início da grade só como adendo datado e hasheado.

## 1. Perguntas

- **B2.1:** no código e no horizonte publicados, a ação fixa é equivalente (±1,0 p.p.) ao TD3 do AdaAggRL?
- **B2.2:** a política TD3 aprendida é essencialmente a inicial e independente da entrada?

## 2. Código

Idêntico ao Passo 2 (código oficial em `external/AdaAggRL`, commit `27b9c18`; venv `external/.venv_adaaggrl`; ajustes mínimos listados em `scripts/passo2_oficial/run_oficial.py`). Runner novo `scripts/passo2_oficial/run_b21.py`, que reusa o do Passo 2 sem alterá-lo e só acrescenta logging: estados observados antes de cada ação e checkpoints do ator do TD3 no passo 0 e a cada 50 passos. Condições e hiperparâmetros idênticos ao Passo 2 (`fixed` = [0,475]×5; `td3` como no `main.py` oficial).

## 3. Escolha da métrica primária (declarada)

A métrica primária foi **escolhida depois de inspecionar as sementes gastas 100–104**, para neutralizar o artefato de reset tardio visto no Passo 2 (o ambiente oficial reinicializa o modelo quando a recompensa < −80). Nas sementes 100–104, o desvio-padrão das diferenças fixed − td3 foi:

| métrica | sd das diferenças | TOST (sementes gastas, **não confirmatório**) |
|---|---|---|
| média 451–500 (Passo 2) | 1,84 p.p. | p = 0,18 |
| mediana 451–500 | 1,14 p.p. | p = 0,03 |
| **mediana 401–500** | **0,32 p.p.** | p < 0,001 |

Justificativa, independente do resultado: a mediana de 100 rodadas não condiciona em resets (que são consequência da ação e não podem ser excluídos sem viés), e um reset tardio contamina tipicamente ~30 das 100 rodadas, abaixo do ponto de ruptura da mediana. A confirmação vem **só** das sementes novas 105–114.

## 4. Grade

- MNIST, q = 0,5, 500 rodadas, 100 clientes, 10% por rodada, 20 atacantes (iguais ao Passo 2).
- Ataques: LMP e EB. Condições: `td3` e `fixed`.
- **Sementes: 105–114** (novas, reservadas no roadmap para confirmação no código oficial).
- Total: 10 × 2 × 2 = **40 runs**, 6 em paralelo, ~60 h de relógio (medido no Passo 2: ~9 h por lote de 6).
- Truncagem em 500 passos (o td3 do SB3 roda 501, Adendo 1 do Passo 2).

## 5. B2.1: hipótese e critérios

**Métrica primária:** mediana da acurácia no teste nas rodadas 401–500.
**Unidade:** par (ataque, semente), n = 20. D = métrica(fixed) − métrica(td3).

**H1.** A ação fixa é equivalente ao TD3.
- TOST pareado (t), margem ±1,0 p.p., α = 0,05 → rejeitadas as duas nulas: **equivalência confirmada**.
- Wilcoxon pareado p < 0,05 com D < 0 → **o TD3 contribui**.
- Wilcoxon p < 0,05 com D > 0 → a fixa é melhor; reportar como "não contribui", sem afirmar "atrapalha" sem replicação.
- Nenhum → **inconclusivo**; reportar Δ, IC95 e d.
- Por ataque: Δ, IC95, d, Wilcoxon com Holm (descritivo).

**Secundárias:** o mesmo procedimento com a média 451–500 (métrica do Passo 2, para continuidade); nº de resets e massa de peso nos atacantes por condição (descritivo).

## 6. B2.2: hipóteses mecanísticas

σ_a = 0,1 × 0,95 / 2 = **0,0475**: desvio-padrão do ruído de exploração do SB3 convertido para unidades de ação. Família H3–H5, **Holm**, α = 0,05, testes unilaterais.

**H3 (replicação do achado exploratório).** Para cada semente, r = correlação de Pearson entre as ações **executadas** pelo td3 sob EB e sob LMP, rodadas 101–500, 5 dimensões achatadas. Confirmada se o Wilcoxon unilateral em atanh(r) − atanh(0,9) > 0 tiver p < 0,05 (n = 10).
*Limitação declarada:* as ações executadas incluem a mesma sequência de ruído nas duas condições (mesma semente). H3 replica o achado, mas o teste limpo da independência da entrada é H5.

**H4 (a política não se afasta da inicial).** Para cada run td3, drift = média de |π₅₀₀(s) − π₀(s)| sobre os estados observados nas rodadas 401–500 e as 5 dimensões, com π determinística (sem ruído) reconstruída dos checkpoints. Confirmada se o Wilcoxon unilateral em drift − σ_a < 0 tiver p < 0,05 (n = 20).

**H5 (a política não depende da entrada).** Para cada run td3, S_swap = média de |π₅₀₀(s_t) − π₅₀₀(s′_t)|, onde s′_t é o estado da mesma rodada no run do **outro ataque** com a mesma semente, t em 401–500. Confirmada se o Wilcoxon unilateral em S_swap − σ_a < 0 tiver p < 0,05 (n = 20).

**Reportados sem teste:** S_shuffle (estado de outra rodada sorteada do mesmo run), S_swap do ator inicial π₀ e a distância de π₀ ao centro, como referências.

## 7. Portão (roadmap §4, B-b)

- H1 com equivalência → a tese do P2 ("o TD3 não contribui") fica confirmatória no código oficial.
- H1 inconclusiva → reportar "sem evidência de contribuição", com B2.2 como evidência principal.
- H4 e H5 confirmadas → "a política não aprende e não depende da entrada" passa a confirmatório.
- O steelman (B2.3) continua necessário, qualquer que seja o resultado.

## 8. Regras

- Análise pré-escrita em `scripts/passo2_oficial/analisar_b21.py`, rodada uma única vez, com os 40 runs completos.
- Runs que falharem por erro de ambiente são repetidos com a mesma semente e registrados. Colapsos e resets contam como resultado.
- Nenhuma acurácia é inspecionada antes do fim da grade. O monitoramento reporta só saúde e progresso.
