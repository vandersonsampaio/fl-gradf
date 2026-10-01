# Pré-registro — B2.6: decomposição do esqueleto (memória, forma da ponderação, sinal)

**Status:** **FINAL**, aprovado pelo autor em 2026-09-30 e congelado antes de qualquer run (hash em `PREREGISTRO.sha256`).
**Data:** 2026-09-30
**Roadmap:** `references/roadmap_tese_gradf_v3.md` §4.1 (ordem 5), confirmatório; agenda da SLR, item 8 ("separar granularidade e memória; nenhum estudo fez").
**Registro:** local, com hash; o commit fica a critério do autor.

## 1. Pergunta

O P2 mostra que o TD3 não contribui (B2.1–B2.3) e que o headroom vem de um **esqueleto fixo**: sinal por cliente, limiar e memória. **Quais peças do esqueleto importam?**
- a memória (penalidade persistente);
- a forma da ponderação (suave contra máscara binária);
- o sinal (S_R da inversão de gradiente contra `cos_server`, que não reconstrói dados).

## 2. Regime e referência

- **Regime:** framework próprio, idêntico à ablação e ao C.0 — MNIST, 10 clientes, bizantinos [0, 1], 15 rodadas, root 100, 3 alphas × 7 ataques = 21 células. **Células válidas: 19** (exclui `fltrust_aligned` α ≤ 0,1, artefato do ataque; C.0).
- **Sementes: 52–61** (novas, reservadas para confirmação no framework próprio).
- **Referência (esqueleto base) = `sr_only`:** o score é o S_R da inversão de gradiente; a ponderação suave é min–max, com limiar δ = max(w̃)·b, b = 0,5; a memória é λ^h com λ = 2. É o mesmo da ablação. Escolhido porque os cues MMD não acrescentaram nada (ablação, H2).

## 3. Variantes

Cada variante muda **uma** peça em relação a `sr_only`.

| id | peça alterada | definição |
|---|---|---|
| `sr_only` | — (referência) | como acima |
| `sr_nomem` | memória desligada | λ = 1 (sem penalidade) |
| `sr_memof` | memória do código oficial | peso × 0,9^flag com o flag **anterior** ao decremento (λ ≈ 1,11, auditoria §3.5) |
| `sr_bin` | máscara binária | clientes acima de δ recebem peso uniforme; mesma memória λ^h |
| `cosserver_only` | sinal sem inversão | score = cosseno até o update do servidor no root; replicação **confirmatória** do exploratório da ablação |
| `sr_cosserver` | sinais combinados (**exploratório**) | score = média de S_R e `cos_server`, com regra fixa |
| `sr_b025`, `sr_b075` | limiar (**sensibilidade**) | b = 0,25 e 0,75; reportado como sensibilidade, nunca usado para escolher configuração |

## 4. Métrica e unidade

- **Métrica:** acurácia global na rodada 15 (a mesma da ablação e do C.0: média sobre os `X_test` dos 10 clientes). Sem resets neste framework.
- **Unidade: semente.** Para cada semente, a média sobre as 19 células válidas. D = variante − `sr_only`, n = 10.

## 5. Hipóteses confirmatórias (família H1–H4, Holm, α = 0,05)

- **H1 (a memória contribui):** `sr_only` − `sr_nomem` > 0. Wilcoxon **unilateral**.
- **H2 (a força da memória importa):** `sr_only` − `sr_memof` ≠ 0. Wilcoxon bilateral. Se for significativa, a λ = 2 da nossa reprodução não é neutra em relação ao código oficial.
- **H3 ("contínuo" não importa):** `sr_bin` ≈ `sr_only`. **TOST ±1,0 p.p.** (IC90 reportado com o IC95).
- **H4 (sinal sem inversão equivale):** `cosserver_only` ≈ `sr_only`. **TOST ±1,0 p.p.**

**Por ataque** (descritivo): Δ, IC95, d e Wilcoxon com Holm sobre os 7 ataques, para cada hipótese.

**Exploratório, sem critério:**
- `sr_cosserver` − `sr_only`, e `sr_cosserver` − melhor de (`sr_only`, `cosserver_only`), por ataque (a hipótese da complementaridade);
- sensibilidade a b;
- perfil por ataque de S_R contra `cos_server`.

## 6. Leitura prevista para o P2

- **H1 confirmada** → "a memória importa" entra nos princípios de projeto (roadmap §4.4).
- **H3 confirmada** → a forma contínua da ponderação não é o que importa. Junto com o B2.8, fecha o eixo "discreto contra contínuo".
- **H4 confirmada** → um sinal **sem reconstrução de dados** substitui o S_R neste regime. Isso sustenta o R3 do GRADF-v2, com o custo de exigir o dataset raiz (`decisao_root_dataset_lgpd.md`).

## 7. Custo

- **Só CPU, em paralelo com a GPU.** Variantes com inversão ≈ 62 s por run; `cosserver_only` ≈ 10 s.
- **Grade:** 8 variantes × 21 células × 10 sementes = 1.680 runs ≈ 25 h de processo, **~3–4 h** com 8 processos (`nice 19`, 2 threads).

## 8. Regras

- Implementação em arquivo **novo** (`scripts/b26_decomposicao.py`), reusando o learner da ablação sem alterá-la. Nada em `src/` muda.
- O extrator aleatório é sempre construído, como na ablação, para preservar o RNG e o pareamento por semente entre variantes.
- Análise pré-escrita e testada com fixtures sintéticas antes do congelamento; rodada uma única vez, com a grade completa.
- Nenhuma acurácia é olhada antes da grade completa.

## 9. Decisões do autor (2026-09-30) e validação antes do congelamento

1. **Referência = `sr_only`.**
2. **Margem do TOST** em H3/H4: **±1,0 p.p.**
3. **Unidade:** média das 19 células válidas por semente, **n = 10**.
4. **Exploratórios incluídos** (`sr_cosserver`, `sr_b025`, `sr_b075`).

Validação (sementes gastas, antes do congelamento):
- `weights_for` coincide com `compute_weights_and_penalty` de `src/` para `sr_only` e `sr_nomem` em 200 casos aleatórios;
- o `sr_only` do B2.6 **reproduz exatamente** o `sr_only` da ablação (semente 42, α = 0,5, `sign_flipping`: 0,889400 = 0,889400);
- a análise, testada com fixtures sintéticas descartadas e efeitos plantados, reagiu corretamente (H1 detectada com −3 p.p.; H3/H4 equivalentes com Δ ≈ 0).

Execução: `scripts/run_grid_b26.sh` (80 jobs = 8 variantes × 10 sementes, 8 processos, `nice 19`, 2 threads), iniciado **depois** do término do B2.8 para reduzir a contenção de CPU com o B2.3b na GPU.
