# Adendo 2 — B2.7b revisto: de onde vem o Δ do TD3, da constante deslocada ou do ruído?

**Data:** 2026-10-02 23:02:53 -0300, antes de qualquer run do B2.7b.
**Declaração:** este adendo foi escrito **depois de ver o B2.7a** (`b27a_analise.txt`): a política do TD3 em H = 150 não depende do estado (sd_estados ~0,0005 ≪ σ_a = 0,15, igual ao ator inicial). O ator deriva até uma constante perto do centro (drift ~0,06). Por isso o B2.7b passa a ser **explicativo**, não um teste de "o TD3 supera a melhor constante". O B2.7c já está descartado (condição 1 do PLANO §4 não atendida, 0/3).

## Pergunta

De onde vem o Δ(td3_ref − fixed) em H = 150: de uma **constante deslocada** ou do **ruído de exploração** (o TD3 executa π(s) + N(0; 0,15) em toda rodada)?

## Sistemas (substituem os braços do PLANO §3)

3 células (`label_flipping` α 0,05 e 0,1; `low_mag_backdoor` α 0,05) × sementes 42–51 × H = 150:

| sistema | o que isola |
|---|---|
| **td3_frozen** | **controle principal:** o mesmo agente, mesmo aquecimento e mesmo processo de ruído, **sem aprender o ator**. O `train_step` roda normalmente (crítico, buffer e RNG consumidos como no td3_ref) e o ator (W, b e alvos) é restaurado ao inicial depois de cada passo. Isso equivale a lr do ator = 0. A atualização do ator não consome RNG, então a sequência de ruído é a mesma do td3_ref. |
| **fixed_td3mean** | o efeito da constante deslocada: ação constante e determinística = média **por célula** de a_TD3 (π₁₅₀ nos estados 101–150, `a_td3.csv`) |
| **fixed_b025**, **fixed_b075** | sensibilidade ao limiar: a = [0,5]×4, b = 0,25 e 0,75, como no plano |
| fixed (centro) e td3_ref | já existem no B2.7 (`results/b27_horizonte/grade_raw.csv`); o td3_ref foi reproduzido exatamente no B2.7a |

- **Grade:** 4 × 3 × 10 = **120 runs**. Com a GPU ocupada pelo B2.4r, os runs com inversão custam ~1.600–2.000 s, o que dá **~5–6 h de relógio com 12 processos**. A estimativa de ~2 h subestima a disputa de CPU.
- **Verificações (smoke de 15 rodadas):**
  - a constante [0,5]×5 pelo caminho novo reproduz o `fixed` do B2.7 (|Δ| = 0);
  - no td3_frozen, o ator fica idêntico ao inicial após 8 passos de treino.

## Critérios (fixados agora, antes de qualquer run do B2.7b)

Unidade: semente, n = 10 por célula. Δ pareado, IC95 t, H = 150.

1. **"O ganho é exploração, não aprendizado"** se:
   - o IC95 de Δ(td3_ref − td3_frozen) **contém 0 nas 3 células**, **e**
   - td3_frozen − fixed > 0 em **pelo menos 1** delas.

   Operacionalizado como Δ > 0 com IC95 > 0, a mesma regra de significância do B2.7.
2. **"Constante deslocada"** se fixed_td3mean ≈ td3_ref, operacionalizado como: o IC95 de Δ(td3_ref − fixed_td3mean) **contém 0 nas 3 células**. A TOST ±1 p.p. é reportada como descritiva.

Os dois critérios não são mutuamente exclusivos e são reportados juntos.

**Secundário (critério original do PLANO §3):** td3_ref contra a melhor constante em retrospecto entre {fixed, fixed_b025, fixed_b075, fixed_td3mean}, com Δ > 0 e IC95 > 0 em ≥ 2 de 3 células.

**Também reportados:** fixed_td3mean − fixed; td3_ref contra b = 0,25 e b = 0,75.

## Código no congelamento
1c30f5bc8ff69ca495971ca8c3d02ac7298939f0be38465c436c002a5d06143f  scripts/b27b_explicativo.py
5cc052c70579c1fa4e86649ba8c2e96215cb8b829fe9d836f80c6838a9d4d747  scripts/run_grid_b27b.sh
