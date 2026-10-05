# Resultado — B2.8: teto oracular da seleção por rodada (eixo da granularidade)

**Data:** 2026-09-30
**Plano:** `PLANO.md` (hash em `PLANO.sha256`; `PLANO.md` e `scripts/b28_oraculo_por_regra.py` conferidos contra o hash antes da análise: OK).
**Grade:** 210/210 runs (sementes 42–51 × 3 alphas × 7 ataques), 30/09 18:54 → ~19:15, CPU, sem erros.
**Análise:** `scripts/b28_oraculo_por_regra.py analisar` → `analise.txt`, `oraculo_por_celula.csv`. **Exploratório.**

---

## 1. Veredito pelo critério do PLANO

- **Células-alvo** (o esqueleto supera a melhor regra fixa estática, W com IC95 > 0): **14/19**.
- Nelas, o oráculo por rodada fica **abaixo** do esqueleto (Q com IC95 < 0) em **8/14** e **acima** em **2/14** (`label_flipping` α = 0,1; `sign_flipping` α = 0,1).
- **VEREDITO PRÉ-REGISTRADO: "A GRANULARIDADE EXPLICA".** Na maioria das células-alvo, nem uma escolha ideal de regra a cada rodada, feita olhando o teste, alcança a filtragem por cliente.

## 2. Robustez do veredito (exploratório, não pré-registrado)

O veredito **depende de três células de α = 0,5 com diferenças mínimas**: `gaussian_noise` −0,12 p.p., `sign_flipping` −0,27 e `krum_collusion` −0,47. Todas têm IC95 < 0, mas são irrelevantes na prática.

| exigência sobre \|Q\| | abaixo do esqueleto | acima | leitura |
|---|---|---|---|
| > 0 (critério do PLANO) | 8/14 | 2/14 | "a granularidade explica" |
| > 1 p.p. (relevância prática) | 5/14 | 2/14 | **inconclusivo** (< 7) |

**Leitura calibrada:** a evidência forte está concentrada em **α ≤ 0,1**, com ataques de modelo em que o S_R separa os clientes. Ali o oráculo fica **4–14 p.p. abaixo** do esqueleto:
- `gaussian_noise` α = 0,05: −13,7 p.p.;
- `gaussian_noise` α = 0,1: −11,3 p.p.;
- `krum_collusion` α = 0,05: −10,8 p.p.;
- `krum_collusion` α = 0,1: −8,9 p.p.;
- `trim_attack` α = 0,05: −5,9 p.p.

Em α = 0,5, esqueleto e oráculo por rodada ficam praticamente empatados. O P2 deve reportar o veredito do critério **junto com** essa ressalva.

## 3. Padrões por tipo de ataque

- **Ataques de modelo com heterogeneidade forte** (`gaussian_noise`, `krum_collusion`, `trim_attack` em α ≤ 0,1): a ponderação por cliente vence com folga até o oráculo. Nenhuma regra única por rodada iguala excluir os clientes certos.
- **Ataques de rótulo** (`label_flipping`): o oráculo **supera** o esqueleto (α = 0,1: +8,9 p.p.; α = 0,5: +11,9 p.p., esta fora das células-alvo porque ali o esqueleto já perde para o Trimmed-Mean). É coerente com o C.0: onde o S_R não separa os clientes, a escolha da regra importa mais que o filtro por cliente.
- **`low_mag_backdoor`:** o oráculo supera o esqueleto (+5,4 a +5,7 p.p. em α ≤ 0,1). A célula não é alvo em α = 0,1 porque ali o esqueleto não supera a regra estática. A ASR não foi medida.
- **Ganho da seleção por rodada sobre a melhor regra fixa (G):**
  - grande em α ≤ 0,1: até +19,7 p.p. em `sign_flipping` α = 0,1 e +17,5 em `label_flipping` α = 0,1;
  - pequeno em α = 0,5: ≤ 2,7 p.p.;
  - negativo em `gaussian_noise` α ≤ 0,1 (−7,1 e −7,7): o guloso por rodada perde para a melhor regra fixa, uma limitação típica do guloso.

## 4. O que o oráculo escolhe

- **α ≤ 0,1:** esmagadoramente **FLTrust** (89–140 de 150 rodadas por célula), a única regra do arsenal ancorada no dataset raiz. **Clustering** vem em segundo nos ataques de sinal e de trim.
- **α = 0,5:** a escolha varia por ataque: Trimmed-Mean (`gaussian_noise`), Clustering (`trim_attack`, `sign_flipping`, `label_flipping`), FedAvg (`low_mag_backdoor`).
- Implicação: o teto da seleção por regra em alta heterogeneidade depende do **sinal do servidor (root)**. É o mesmo trade-off de privacidade do `cos_server` (B2.6/H4; `decisao_root_dataset_lgpd.md`).

## 5. Consequências para o P2 e o P3

- **P2 (eixo da granularidade):** a diferença que o P1 atribuiu a "discreto contra contínuo" é, **em α ≤ 0,1 com ataques de modelo**, uma diferença de **granularidade**: uma regra para todos contra um peso por cliente. Nem a escolha ideal de regra alcança o filtro por cliente. **Não** é uma afirmação geral: em ataques de rótulo, a seleção por regra (idealizada) supera o esqueleto, e em α = 0,5 os dois empatam.
- **P3 (direção 1 do C.2, dois níveis):** o filtro por cliente vence onde o sinal separa, e a regra robusta vence onde não separa. Isso reforça a arquitetura de **filtro por cliente seguido de regra robusta**.

## 6. Limitações

- **Exploratório:** sementes gastas; referências da ablação e do exp9 já conhecidas.
- **O oráculo usa o teste para escolher:** é um teto, não um método. Isso só fortalece as células em que ele perde.
- **Guloso:** ótimo por rodada, não na trajetória (visto nos G negativos em `gaussian_noise` α ≤ 0,1).
- **15 rodadas:** gap misturado com convergência; o B2.7 trata do horizonte.
- **ASR** do `low_mag_backdoor` não medida.

## Nota de 2026-10-05 (ponderação × robustez)

- Dois dos 7 braços do oráculo por rodada (FedAvg e FedProx) ponderam por tamanho de amostra; os outros 5 e o esqueleto não.
- Sob rótulos enviesados por cliente (α ≤ 0,1) e teste global balanceado, **a ponderação por tamanho de amostra do FedAvg padrão custa, sozinha e sem ataque, até ~4,5 p.p.** em relação à ponderação uniforme (oráculo-8, α 0,1, H = 150: 85,0% × 89,5%; FedAvg-10: 1,0–1,8 p.p.). Tabelas que comparam defesas com o FedAvg padrão (e os braços FedAvg/FedProx do B2.8 e do B2.7) devem descontar esse efeito, para não atribuir à robustez da defesa o que é só ponderação.
- No B2.8, o oráculo escolhe a regra pela própria acurácia, a cada rodada, e por isso pode evitar FedAvg/FedProx quando a ponderação por amostra prejudica.
- A métrica (acurácia no test set global IID; test sets dos clientes são partes iguais) **não** favorece por si a ponderação uniforme: a sensibilidade de métrica do B2.8s é vazia por construção e reproduziu o veredito (`results/b28s_sensibilidade_metrica/RESULTADO.md`).
