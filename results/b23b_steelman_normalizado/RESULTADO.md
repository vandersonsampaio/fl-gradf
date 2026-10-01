# Resultado — B2.3b: último steelman do TD3 (recompensa normalizada)

**Data:** 2026-10-01
**Plano:** `PLANO.md` (hash em `PLANO.sha256`; `PLANO.md`, `run_b23b.py`, `analisar_b23b.py`, `analisar_b23.py` e `analisar_b21.py` conferidos contra os hashes antes da análise: OK).
**Grade:** 10/10 runs (sementes 100–104 × LMP/EB; `VecNormalize` só na recompensa, lr 1e-4, `learning_starts` 10), 30/09 18:55 → 01/10 09:50, 0 falhas. Pareado com `fixed` e `td3` do Passo 2 e com o steelman do B2.3.
**Análise:** `scripts/passo2_oficial/analisar_b23b.py` → `analise.txt`, `resumo_runs.csv`, `mecanismo.csv`, `drift_por_checkpoint.csv`. **Exploratório** (sementes gastas).

---

## 1. Portão B-a definitivo (critério fixado no PLANO §5)

| condição | resultado |
|---|---|
| 1. C1 B2.3b − fixed com Δ > 0 e p < 0,05 | **NÃO**: Δ = +0,95 p.p. (IC95 −0,87 a +2,78), Wilcoxon p = 0,066 |
| 2. a política se afastou da inicial (drift_500 > σ_a = 0,0475) | **SIM**: mediana 0,174 |
| 3. a política depende do estado (sd_estados > σ_a) | **NÃO**: mediana 0,0134 (máx. 0,0231) |

→ **VEREDITO: o steelman NÃO supera a ação fixa. O Portão B-a fica definitivo e a busca de configuração termina** (cláusula de "último steelman"). **O B2.3c não roda.**

Comparações (primária, mediana 401–500; n = 10 pares):
- **C1 contra fixed:** +0,95 p.p., não significativo. Na secundária (média 451–500): **Δ = 0,00 p.p.** (IC95 ±0,42).
- **C2 contra o td3 publicado:** +0,77 p.p., p = 0,56.
- **C3 contra o steelman do B2.3:** +11,2 p.p., p = 0,010. A normalização **eliminou os colapsos** do B2.3: resets extras sob EB caíram de 15,8 para 0,8 por run, no nível da ação fixa (0,6).

Em resumo, bem condicionado, o TD3 fica **no nível da ação fixa e do TD3 publicado**, sem ganho e sem colapso.

## 2. Mecanismo (exploratório)

- **A política se move continuamente e não satura:** a mediana do drift cresce de forma quase linear ao longo do treino (0,009 no passo 50; 0,047 no 200; 0,13 no 400; **0,174 no 500**). A ação final é um **ponto interior** do Box que varia por semente (por exemplo, LMP s100 [0,79; 0,71; 0,35; 0,75; 0,31]), e não um canto como no B2.3.
- **A dependência do estado cresce, mas continua pequena:** o sd_estados (mediana) por checkpoint, calculado sobre os mesmos estados das rodadas 401–500, foi:

  | passo | 0 | 100 | 200 | 300 | 400 | 500 |
  |---|---|---|---|---|---|---|
  | sd_estados | 0,0045 | 0,0048 | 0,0059 | 0,0085 | 0,0113 | **0,0134** |

  Ela **triplica** em 500 rodadas, mas fica em **~1/3,5 do ruído de exploração** (σ_a = 0,0475). O passo 0 corresponde à sensibilidade de uma rede aleatória.
- **Extrapolação (especulativa):** no ritmo das últimas 200 rodadas (~+0,0025 a cada 100 rodadas), o sd_estados só chegaria a σ_a por volta da rodada ~1.900. **Não dá para excluir** que, nesta configuração, uma dependência do estado surja num horizonte muito maior que o publicado. Mesmo que surja, nada indica que ela traria ganho de acurácia: na secundária, Δ = 0,00 contra a fixa.

## 3. Quadro dos dois steelmen (para o P2)

| | TD3 publicado (B2.1/B2.2) | steelman B2.3 (lr 1e-3, recompensa bruta) | **steelman B2.3b (lr 1e-4, recompensa normalizada)** |
|---|---|---|---|
| contra fixed | equivalente (TOST) | −10,2 p.p. (colapsos) | +0,95 p.p., n.s. (secundária: 0,00) |
| drift_500 | 0,0097 | 0,47 (satura em ~200 rodadas) | 0,17 (ainda crescendo) |
| dependência do estado | igual à de uma rede aleatória (S_swap 0,0055) | nula (canto constante, sd 0,001) | pequena e crescente (sd 0,0134; 3× o inicial) |
| resets sob EB | 2,8 | 15,8 | 0,8 |

**Leitura para o P2:** em três regimes de aprendizado, o TD3 **nunca supera a ação fixa** em 500 rodadas. O motivo varia:
- no regime publicado, ele não aprende;
- com lr alto e recompensa bruta, degenera num canto e desliga a defesa;
- bem condicionado, aprende devagar uma política quase constante, que empata com a fixa.

O único sinal de aprendizado dependente do estado (B2.3b) é pequeno, cresce devagar e não se converte em ganho. Isso reforça o B2.7 (horizonte) como lacuna a reportar: a pergunta "a partir de que N o RL se paga?" não está respondida para orçamentos muito maiores que o publicado.

## 4. Limitações

- **Exploratório:** sementes gastas; n = 10 pares; linhas de base do Passo 2 já inspecionadas.
- **Uma configuração.** Pela cláusula de término, nenhuma outra será testada nesta linha.
- O sd_estados por checkpoint e a extrapolação são análises **não pré-registradas**, feitas depois de a grade estar completa.
- **Horizonte de 500 rodadas,** o do código publicado.
