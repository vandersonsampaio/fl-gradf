# Pré-registro: Passo 2, o TD3 contribui no AdaAggRL oficial?

**Status:** FINAL. A grade foi fixada em 2026-09-25, depois da medição de custo (semente 0, descartada) e antes de qualquer execução confirmatória.
**Data:** 2026-09-25
**Plano:** `references/proximos_passos_pos_ablacao.md`, Passo 2.
**Auditoria que embasa as escolhas:** `references/auditoria_fidelidade_adaaggrl.md`.
**Registro:** local e sem commit, por decisão do autor. O arquivo não deve ser editado depois do início da grade; qualquer mudança posterior entra como adendo datado no fim.

## 1. Pergunta

No código publicado do AdaAggRL (`github.com/yjEugenia/AdaAggRL`, commit `27b9c18`), no horizonte publicado (500 rodadas), a política TD3 produz acurácia melhor que uma ação fixa e que uma ação aleatória?

## 2. Código e ajustes

- Código oficial intacto em `external/AdaAggRL`, venv isolado `external/.venv_adaaggrl` com as versões do `requirements.txt` oficial (torch 2.3.0, SB3 2.3.2, numpy 1.26.4, gym 0.26.2), mais o gymnasium 0.29.1 que o SB3 exige.
- Runner: `scripts/passo2_oficial/run_oficial.py`. Os ajustes mínimos estão listados no docstring dele: shim do `inversefed`, adaptador Gymnasium, sementes, SummaryWriter no-op e `--dataset MNIST`. Nenhuma linha da lógica oficial (ambiente, ataques, recompensa, TD3) é alterada.
- A recompensa oficial usa o conjunto de teste (auditoria §2.1). É mantida, porque a pergunta é sobre o método publicado. O vazamento só pode favorecer o TD3.

## 3. Condições

| condição | ação |
|---|---|
| `td3` | TD3 do SB3 exatamente como no `main.py` (MlpPolicy [256,128], lr 1e-5, buffer 1000, batch 64, `train_freq` 3, ruído N(0; 0,1), γ 0,99, `learning_starts` 100) |
| `fixed` | [0,475]×5, o centro do Box [0; 0,95]^5: a[:4] constante dá softmax uniforme, e a₅ = 0,475 é a média das ações do próprio TD3 na fase aleatória |
| `random` | uniforme em [0; 0,95]^5 durante as 500 rodadas |

## 4. Grade

- Dataset MNIST, 500 rodadas, 100 clientes, 10% por rodada, 20 atacantes (padrões oficiais).
- **q = 0,5** (não-IID moderado, um dos três valores do artigo). O regime da ablação 42–51 é não-IID, e o q = 0,1 padrão do `main.py` é IID.
- **Ataques:** LMP e EB. O IPM oficial fica **fora**: é um update nulo (auditoria §2.2) e custaria mais cerca de 22 h.
- **Sementes:** 100, 101, 102, 103, 104 (novas, nunca usadas no projeto).
- **Total:** 2 ataques × 3 condições × 5 sementes = **30 runs**.
- **Custo medido** (semente 0, EB/LMP/IPM, runs descartados): cerca de 27 s por rodada com 1 processo e cerca de 64 s por rodada por processo com 6 processos em paralelo (3 threads cada, GPU RTX 3050 a 98%). Isso dá cerca de 1,5 h de máquina por run e **cerca de 44 h** para a grade.
- **Ordem de execução:** todas as combinações das sementes 100–102 primeiro, depois 103–104. É só agendamento: as 5 sementes estão comprometidas e **nenhuma análise confirmatória é feita antes da grade completa**.
- **Paralelismo:** 6 processos, `OMP_NUM_THREADS=3`. Não afeta a lógica.

## 5. Métricas

- **Primária:** acurácia média no conjunto de teste nas 50 últimas rodadas (451–500), pelo `history['acc']` oficial.
- **Secundárias:** acurácia na rodada 500; curva completa; massa de peso atribuída aos atacantes reais por rodada; número de resets (recompensa < −80); disparos da regra `sim_lc ≥ 0,9`; trajetória das ações do TD3.

## 6. Hipóteses e critérios

Seja D = métrica(`fixed`) − métrica(`td3`), por par (ataque, semente).

**H1 (tese central).** O TD3 não supera a ação fixa.
- Teste primário: TOST pareado (t) sobre os pares (ataque, semente) dos ataques principais, com margem **±1,0 p.p.** e α = 0,05.
  - TOST rejeita as duas hipóteses nulas → **equivalência**; a tese central se confirma no método publicado.
  - Wilcoxon pareado com p < 0,05 e D < 0 → **o TD3 contribui** no horizonte longo; o artigo muda para "o RL só se paga depois de N rodadas".
  - Wilcoxon com p < 0,05 e D > 0 → a ação fixa é melhor. Reportar como "não contribui"; **não** afirmar que o TD3 atrapalha sem replicação.
  - Nenhum dos casos → **inconclusivo**; reportar Δ, IC 95% e d, sem declarar equivalência.
- Por ataque: Δ, IC 95%, d e Wilcoxon com Holm sobre os ataques (descritivo, dado o n pequeno).

**H2 (secundária).** `random` − `td3`, com o mesmo procedimento. Serve para saber se a política aprendida é distinguível de ação nenhuma.

**Análise exploratória, não confirmatória:**
- a curva de D ao longo das rodadas (o TD3 só age por política depois da rodada 100);
- a trajetória das ações do TD3 e a **distância média até o centro do Box (0,475)**. No smoke test (semente 0, `learning_starts=2`, descartado), a política não treinada já emite ações perto de 0,47, porque o tanh na inicialização é ≈ 0. A ação `fixed` coincide, portanto, com o TD3 "antes de aprender", e a pergunta vira se a política se afasta do ponto inicial com cerca de 66 atualizações a lr 1e-5;
- a massa de peso nos atacantes reais, por condição.

## 7. Regras

- Nenhum parâmetro do código oficial é ajustado depois de ver resultados da grade.
- Runs que falharem por erro de ambiente são repetidos com a mesma semente e registrados. Runs com NaN ou colapso contam como resultado.
- Resets são contabilizados e não descartados.

---

## Adendo 1 (2026-09-25, 21:35): truncagem em 500 rodadas

**Motivo:** o primeiro run `td3` concluído (EB, semente 100) tem **501 passos**. O SB3 2.3.2 coleta em blocos de `train_freq=3`, e `learn(total_timesteps=500)` só para ao fim de um bloco (498 → 501). O `main.py` oficial se comporta da mesma forma. As condições `fixed` e `random` têm exatamente 500 passos.
**Mudança:** na análise, todos os runs são truncados nos **500 primeiros passos** antes de calcular qualquer métrica. A métrica primária fica, para todas as condições, na janela das rodadas 451–500, como pré-registrado. A análise registra `n_steps_raw` para transparência.
**Informação vista antes do adendo:** só a contagem de passos e o tempo dos runs. **Nenhuma acurácia ou comparação entre condições foi olhada.**
**Arquivo alterado:** `scripts/passo2_oficial/analisar.py` (função `load`).
