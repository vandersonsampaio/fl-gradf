# Resultado — B2.3: steelman do TD3 no AdaAggRL oficial

**Data:** 2026-09-30
**Plano:** `PLANO.md` (hash em `PLANO.sha256`; `PLANO.md`, `analisar_b23.py`, `analisar_b21.py` e `analisar.py` conferidos contra os hashes antes da execução: OK). Agendamento: `EXECUCAO.md`.
**Grade:** 10/10 runs (sementes 100–104 × LMP/EB; TD3 com lr 1e-3 e `learning_starts` 10), 2026-09-29 21:01 → 2026-09-30 16:47, 0 falhas. Pareado com `fixed` e `td3` do Passo 2.
**Análise:** `scripts/passo2_oficial/analisar_b23.py` → `analise.txt`, `resumo_runs.csv`, `mecanismo.csv`, `drift_por_checkpoint.csv`. **Exploratório** (sementes gastas).

---

## 1. Portão B-a (critério fixado no PLANO antes dos dados)

| condição | resultado |
|---|---|
| 1. C1 steelman − fixed com Δ > 0 e p < 0,05 | **NÃO**: Δ = **−10,22 p.p.** (IC95 −24,3 a +3,8), Wilcoxon p = 0,084 |
| 2. a política se afastou da inicial (drift_500 > σ_a = 0,0475) | **SIM**: mediana 0,470 (~10× o ruído) |
| 3. a política usa a entrada (S_swap > σ_a) | **NÃO**: mediana 0,0011 |

→ **VEREDITO: o steelman NÃO supera a ação fixa. A tese do P2 fica fortalecida.**

Detalhe por ataque (primária, mediana 401–500):
- **LMP:** steelman ≈ fixed (Δ = −0,09 p.p.) e ≈ td3 (Δ = −0,23 p.p.). A ação é irrelevante sob LMP, como no Passo 2.
- **EB:** Δ = −20,3 p.p. contra fixed, puxado por **dois colapsos** (sementes 100 e 103: 0,476 e 0,436). Nas outras 3 sementes o steelman fica em 0,88–0,97.
- C2 (steelman − td3 oficial): Δ = −10,4 p.p., Wilcoxon p = 0,002. O steelman é **pior** que a configuração publicada.
- Resets extras sob EB: steelman 15,8 por run, contra td3 2,8 e fixed 0,6.

## 2. Mecanismo: a política aprende uma ação constante de canto (exploratório)

- **O drift cresce rápido e satura:** mediana 0,10 no passo 50, 0,31 no 100 e 0,47 do passo 200 em diante. A política chega à borda do Box em ~200 rodadas e para ali.
- **A ação final é um canto do Box [0; 0,95]^5, a mesma para qualquer estado:** o desvio-padrão da ação sobre os estados das rodadas 401–500 fica entre 0,0001 e 0,02. Exemplos: LMP s100 = [0,95; 0,95; 0; 0,95; 0], EB s103 = [0; 0; 0; 0; 0], EB s104 = [0; 0; 0; 0,95; 0,95].
- **O colapso sob EB coincide com limiar a₅ ≈ 0:** EB s100 (a₅ = 0,00) e EB s103 (a₅ = 0,00) colapsaram, com 38 e 36 resets. Com δ = max(k)·a₅ ≈ 0, o filtro só exclui o cliente de menor score, e os atacantes do EB (com boost) entram na agregação. Quando o canto aprendido tem a₅ alto (EB s102: 0,94; s104: 0,95) ou intermediário (s101: 0,25), a defesa funciona. **Sob LMP**, mesmo a₅ ≈ 0 não colapsa (s100, s103), coerente com a massa quase nula nos atacantes do LMP em todas as condições.
- **S_swap ≈ 0,001:** a política final responde ao estado **ainda menos** que o ator inicial do B2.2 (0,0055), o que é típico de saturação do tanh.

**Leitura:** dar ao TD3 um orçamento de aprendizado real (lr 100×, aquecimento 10×) não produz uma política condicionada ao estado. Produz uma **ação constante degenerada**, escolhida quase ao acaso entre os cantos do Box (varia por semente), que em 2 de 5 sementes sob EB desliga a filtragem.

## 3. Consequências (roadmap §4, Portão B-a)

- **B-a:** "steelman não supera a fixa" → a tese do P2 fica forte. Somado ao B2.1 + B2.2 (equivalência confirmada; política publicada não aprende nem usa a entrada), o quadro é:
  - **orçamento publicado:** o TD3 não sai da inicialização, o que equivale a uma ação fixa no centro;
  - **orçamento generoso:** o TD3 sai da inicialização, mas para um **canto constante**, também independente da entrada, e com risco de colapso.
- Em nenhum dos regimes o TD3 aprende o que justificaria usar RL (ação dependente do estado).

## 4. Limitações declaradas

- **Exploratório:** sementes gastas, n = 10 pares, linhas de base do Passo 2 já inspecionadas.
- **Uma única configuração.** A saturação nos cantos sugere um problema de condicionamento, provavelmente a escala da recompensa: o oficial usa a **soma** da loss sobre ~156 batches do teste, com magnitudes de dezenas a centenas, sem normalização. Um revisor pode pedir um steelman com **recompensa normalizada** (ou observações normalizadas, ou lr menor que 1e-3). O correto é afirmar "esta configuração favorável não supera a fixa; o aprendizado degenera em ação constante de canto", e não "nenhuma configuração supera".
- **Possível passo seguinte (a decidir, não pré-registrado):** B2.3b com recompensa normalizada e lr 1e-4, mesmas sementes, ~16 h.
