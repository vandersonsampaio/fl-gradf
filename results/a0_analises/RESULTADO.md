# Resultado — A0: análises baratas (roadmap v4 §4.2)

**Data:** 2026-10-01. Plano congelado em `PLANO.md` (hash em `PLANO.sha256`, commit `79f9ce1`) antes dos runs e das análises.
**Saídas:** `analise.txt` (itens a–c), `concordancia_sinais.csv`, `massa/*.csv` (48 runs), `analise_regimes_td3.txt` e `regimes_td3_por_checkpoint.csv` (item d).

## (a) Massa de peso nos atacantes → direção 6 do P3 sem motivação

- **Verificação:** os 48 runs reproduzem exatamente o B2.6 (máx. |Δacc| = 0). O registro de pesos não perturbou a trajetória.
- **Regra pré-registrada:** "aproveitamento real" exige massa média ≥ 0,05 em ≥ 4 das 8 células. Resultado: **2/8 células**, tanto no `sr_only` quanto no `sr_bin` → **sem aproveitamento relevante.**
- Em 6 células (`gaussian_noise`, `krum_collusion` e `trim_attack`, em α = 0,05 e 0,1), a massa é **0,0000–0,0001**: os atacantes são excluídos por completo. O ganho sobre o oráculo FedAvg-8 nessas células (gap negativo do C0) vem, portanto, de **como os honestos são ponderados**, não de usar os atacantes. *Precisão de 2026-10-04:* concretamente, da ponderação **uniforme** dos honestos contra a ponderação **por tamanho de amostra** do oráculo. Com a métrica de média uniforme sobre clientes, o oráculo-8 uniforme reproduz os melhores métodos (`results/c0b_espaco_h150/VERIFICACOES.md` §3; `results/c0_espaco_restante/CORRECAO_2026-10-04.md`).
- As 2 exceções não são "aproveitamento" no sentido da direção 6:
  - `low_mag_backdoor` α = 0,1: massa ≈ 0,19, perto do uniforme (0,2). O S_R não separa esse ataque (AUC 0,51 no item c), então os atacantes **passam**, não são explorados.
  - `sign_flipping` α = 0,1: massa 0,06–0,07, concentrada nas rodadas 2–7 (0,08–0,29), que caem a ~0 depois da rodada 8. É um transiente de exclusão tardia.
- **Leitura:** os gaps negativos do C0 **não** se explicam por aproveitamento dos atacantes. A direção 6 ("aproveitar em vez de excluir") perde a motivação empírica.
- *Nota:* nessas variantes a semente muda pouco a trajetória (a partição é fixa; a semente só afeta o root e os agentes). No `sr_bin`/`sign_flipping`, as 3 sementes dão a mesma massa por rodada.

## (b) Teto da seleção de sinais → headroom da direção 1 é ~+1,7 p.p.

Média sobre as 19 células válidas do B2.6, por semente (n = 10):

| estimativa | acurácia (%) | IC95 |
|---|---|---|
| máx. por semente (viesado para cima) | 84,30 | 83,72–84,88 |
| escolha por célula, in-sample | 83,59 | 82,82–84,36 |
| **escolha por célula, LOSO (referência)** | **83,24** | 82,61–83,87 |
| sr_b025 (melhor sinal único fixo) | 82,47 | 82,40–82,53 |
| sr_cosserver | 82,11 | 81,01–83,21 |
| sr_bin | 81,92 | 81,52–82,32 |
| cosserver_only | 81,32 | 80,48–82,16 |
| sr_only | 80,90 | 80,51–81,28 |

- **Ganho da escolha de sinal por célula (LOSO) sobre o melhor sinal único por semente: +1,75 p.p. (IC95 +1,31 a +2,19).** É o headroom que a direção 1 do P3 disputa. É pequeno, mas tem IC inteiro acima de zero. Contra o `sr_b025` fixo, a margem cai para ~+0,8 p.p.
- O padrão de escolha é legível: **cos_server** em `label_flipping`, `sign_flipping` e `low_mag_backdoor` (ataques que o S_R não separa); **S_R** em `gaussian_noise`, `krum_collusion` e `trim_attack` com α baixo.

## (c) Concordância S_R × cos_server (preliminar: semente 42, trajetória td3)

- **Spearman médio por rodada entre clientes:** de −0,36 a +0,35. Os dois sinais são **quase ortogonais**.
- **AUC complementar:**
  - o **S_R** separa `gaussian_noise` (≈ 1,0), `krum_collusion` e `trim_attack` (0,86–0,97);
  - o **cos_server** separa `label_flipping` (0,80–1,0) e `sign_flipping` (0,83–0,99);
  - nenhum dos dois separa bem `low_mag_backdoor`;
  - em `fltrust_aligned`, o cos_server é **invertido** (AUC 0), artefato conhecido do ataque.
- **Leitura:** um seletor não aprendido (por exemplo, "usar o sinal de maior separação") é plausível, porque os sinais falham em ataques diferentes. A confirmação exige mais sementes.

## (d) Três regimes do TD3 (figura do P2)

Mediana entre runs, estados das rodadas 401–500, σ_a = 0,0475:

| checkpoint | drift publicado | drift B2.3 | drift B2.3b | sd_estados publicado | sd_estados B2.3 | sd_estados B2.3b |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0,0046 | 0,0047 | 0,0045 |
| 100 | 0 | 0,314 | 0,023 | 0,0046 | 0,0195 | 0,0048 |
| 250 | 0,0035 | 0,467 | 0,063 | 0,0046 | 0,0010 | 0,0072 |
| 500 | **0,0097** | **0,470** | **0,174** | **0,0047** | **0,0011** | **0,0134** |

- **Publicado:** a política quase não se move (drift ≪ σ_a) e não depende do estado.
- **B2.3 (lr 1e-3):** a política se move muito, satura num canto do Box até o passo ~200 e fica **ainda menos** dependente do estado (sd 0,001).
- **B2.3b (lr 1e-4, recompensa normalizada):** o drift cresce de forma monotônica e o `sd_estados` triplica (0,0045 → 0,0134). Mesmo assim, fica a ~¼ de σ_a em 500 rodadas. É a única configuração com dependência de estado crescente, que é a limitação de horizonte já registrada no roadmap (B2.7b cortado).

## Decisões para o P3

1. **Direção 6** (aproveitar atacantes): **sem motivação** pelo critério pré-registrado.
2. **Direção 1** (seleção de sinal): headroom real, mas modesto (~+1,7 p.p. sobre o melhor sinal único por semente; ~+0,8 sobre o `sr_b025`). Os sinais são complementares por tipo de ataque.
