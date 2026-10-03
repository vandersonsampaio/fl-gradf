# Adendo 1 — B2.5: ε escolhido no sanity

**Data:** 2026-10-01, antes de qualquer run da grade (hash em `ADENDO1.sha256`).
**Plano:** `PLANO.md` §3 (commit `1e92014`). Aplicação mecânica: `scripts/passo2_oficial/analisar_b25.py sanity` → `sanity.txt`.

## Resultado do sanity (semente 100, 100 rodadas; média 81–100)

| condição | acurácia 81–100 | critério |
|---|---|---|
| FedAvg sem ataque | 93,38% | referência |
| FedAvg, ε = 2 | 87,48% (−5,90 p.p.) | degrada ✔ |
| FedAvg, ε = 10 | 29,91% (−63,48 p.p.; 8 resets) | degrada ✔ |
| fixed, ε = 2 | massa média nos atacantes 0,1408 (92 rodadas com atacantes reais) | não excluído ✔ |
| fixed, ε = 10 | massa média nos atacantes 0,0103 | não excluído ✔ (no limite) |

O td3 não entrou no sanity.

## Escolha

**ε = 2.** Os dois candidatos satisfazem os dois critérios, e a regra de desempate fica com o menor.

Observações, descritivas e sem efeito na escolha:
- Com ε = 2, o IPM degrada o FedAvg só um pouco acima do limiar (5,9 p.p.), e o fixed dá peso substancial aos atacantes (~0,14 contra ~0,2 do uniforme). É o regime em que a ponderação **pode** fazer diferença.
- ε = 10 seria quase totalmente excluído pelo fixed (0,0103, no limite do critério).

## Grade

20 runs: IPM real ε = 2 × {td3, fixed} × sementes 105–114, 500 rodadas, 5 em paralelo na GPU:
`bash scripts/passo2_oficial/run_grid_b25.sh grade 2`.
Análise: `analisar_b25.py grade --eps 2`, uma vez, com a grade completa.
