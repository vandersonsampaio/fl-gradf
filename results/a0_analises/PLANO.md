# Plano — A0: análises baratas (roadmap v4 §4.2)

**Status:** FINAL antes dos runs e das análises (hash em `PLANO.sha256`). Descritivo/exploratório.
**Data:** 2026-10-01
**Roadmap:** `references/roadmap_tese_gradf_v4.md` §4.2 e §7.2 (item 1: decide as direções 1 e 6 do P3).

## Itens

**(a) Massa de peso nos atacantes** (`scripts/a0_analises.py massa` / `analisar`)
- O B2.6 não gravou pesos por cliente, então são **runs novos**: `sr_only` e `sr_bin` × 8 células × sementes **52, 53 e 54** = **48 runs** (15 rodadas, CPU).
- **Células:** as 8 com gap negativo e IC95 inteiro abaixo de 0 no C0 (métodos sob ataque acima do oráculo FedAvg-8): `gaussian_noise`, `krum_collusion` e `trim_attack` em α = 0,05 e 0,1; `sign_flipping` e `low_mag_backdoor` em α = 0,1.
- **Verificação:** o learner herda o do B2.6 e só **registra** pesos e `is_byz`. A acurácia final de cada run deve reproduzir exatamente a do B2.6 (máx. |Δ| < 1e-9). Se não reproduzir, o item (a) é invalidado e reportado como tal.
- **Métrica:** massa = soma dos pesos normalizados dos clientes bizantinos por rodada, média nas 15 rodadas e por célula. Referência: FedAvg uniforme dá 0,2.
- **Regra de decisão** (direção 6 do P3, "aproveitar em vez de excluir"): **aproveitamento real** se a massa média for ≥ **0,05** em pelo menos metade das 8 células, para a variante considerada. Caso contrário, os gaps negativos do C0 **não** se explicam por aproveitamento dos atacantes, e a direção 6 perde a motivação.

**(b) Teto da seleção de sinais** (dados do B2.6, sementes 52–61, 19 células válidas)
- **Três estimativas**, todas como média sobre as 19 células por semente:
  1. max por semente entre `sr_only` e `cosserver_only` (**viesada para cima**: máximo de dois valores ruidosos);
  2. escolha do sinal por célula, in-sample;
  3. escolha por célula **deixando a semente de fora** (LOSO, sem viés de seleção).
- Comparadas com `sr_only`, `cosserver_only`, `sr_cosserver`, `sr_bin` e `sr_b025`, e com o melhor sinal único por semente.
- **Uso:** dimensiona o headroom que a direção 1 do P3 disputa. A estimativa **de referência é a LOSO**.

**(c) Concordância entre os sinais por cliente** (**preliminar**)
- **Fonte:** registro da ablação (`results/frente1_ablacao_adaaggrl/detectores/`), que cobre só a **trajetória td3, semente 42**, nas 21 células.
- **Medidas:** Spearman médio por rodada entre S_R e `cos_server` entre os clientes, e AUC de cada sinal para separar os atacantes.
- **Uso:** candidato a seletor não aprendido. É preliminar porque usa uma única semente e outra trajetória.

**(d) Tabela-síntese dos três regimes do TD3** (`scripts/passo2_oficial/a0_regimes_td3.py`, venv oficial)
- **Regimes:** publicado (B2.1), B2.3 e B2.3b.
- **Medidas:** mediana de drift e de `sd_estados` por checkpoint (0–500), sobre os estados das rodadas 401–500. Usa os checkpoints intermediários **locais**.
- **Uso:** figura do P2.

## Custo e regras

- **(a):** 48 runs de ~62 s de processo, em 3 processos (um por semente), `nice 19`.
- **(b) a (d):** minutos.
- **(a) roda antes do B2.7,** para não disputar CPU.
- Nenhum resultado é usado para ajustar os planos do B2.5 e do B2.7, que estão congelados à parte.
