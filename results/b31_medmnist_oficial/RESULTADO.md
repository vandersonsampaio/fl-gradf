# Resultado — B3.1 + B3.2: fixed contra TD3 no AdaAggRL oficial, BloodMNIST

**Data:** 2026-10-09
**Pré-registro:** `PREREGISTRO.md` (`315b880`), com os adendos 0, 1 e 2. `analisar_b31.py` conferido contra o hash do Adendo 1 antes da execução (`dff2220a…`, idêntico).
**Grade:** 40/40 runs (sementes 135–144 × LMP/EB × fixed/td3; BloodMNIST, q = 0,5, 500 rodadas), de 2026-10-05 17h27 a 2026-10-09 01h46. Fila intercalada, 6 processos na GPU.
- **Execução:** nenhuma falha e **nenhum run repetido** (§6 do pré-registro).
- **Pausas:** a janela 7h–18h seg–sex foi aplicada pelo controlador. As exceções do autor (retomadas manuais em 05–08/10) estão registradas em `results/janela_execucao.log`. Pausar com SIGSTOP não altera a trajetória de um run, só o relógio.

**Análise:** `scripts/passo2_oficial/analisar_b31.py --margem 3.0 --ataques LMP EB`, rodada uma única vez com a grade completa. Ela gerou `analise.txt`, `resumo_runs.csv` e `mecanismo.csv`.

---

## 1. B3.1 — resultado confirmatório

**Métrica primária:** mediana da acurácia nas rodadas 401–500. D = fixed − td3, n = 20 pares (ataque × semente), margem M = ±3,00 p.p. (Adendo 1; **equivalência fraca**, pela regra do §3).

| métrica | Δ (p.p.) | IC90 | IC95 | d | TOST ±3,0 | Wilcoxon | veredito |
|---|---|---|---|---|---|---|---|
| **primária (mediana 401–500)** | **+0,13** | (−0,64; +0,90) | (−0,80; +1,06) | +0,07 | **p < 0,0001** | p = 0,70 | **EQUIVALENTE** |
| AUC 1–500 | +0,12 | (−0,30; +0,54) | (−0,39; +0,63) | +0,11 | p < 0,0001 | p = 0,50 | EQUIVALENTE |
| AUC 251–500 | +0,09 | (−0,49; +0,68) | (−0,61; +0,80) | +0,06 | p < 0,0001 | p = 0,29 | EQUIVALENTE |

A primária e as duas AUCs concordam, então o resultado **não** é "sensível a resets".

**Por ataque (descritivo, Holm):**

| ataque | métrica | Δ (p.p.) | IC95 | d | Wilcoxon (Holm) |
|---|---|---|---|---|---|
| LMP | primária | −0,47 | (−1,94; +1,00) | −0,23 | 0,70 (0,70) |
| EB | primária | +0,73 | (−0,57; +2,03) | +0,40 | 0,28 (0,55) |
| LMP | AUC 251–500 | −0,27 | (−1,74; +1,20) | −0,13 | 0,85 (0,85) |
| EB | AUC 251–500 | +0,45 | (+0,09; +0,82) | +0,88 | 0,027 (0,055) |

→ **VEREDITO PRÉ-REGISTRADO: EQUIVALÊNCIA CONFIRMADA** (M = ±3,0 p.p.).
- O IC90 da primária fica **bem dentro de ±1 p.p.** Ou seja, a equivalência também valeria com a margem do MNIST, embora isso não seja o teste pré-registrado.
- Nenhuma diferença por ataque é significativa depois do Holm. O caso mais próximo (EB, AUC 251–500, Holm 0,055) aponta para a fixa e é descritivo.

## 2. B3.2 — hipóteses mecanísticas (Holm sobre H3–H5, σ_a = 0,0475)

| hipótese | estatística | resultado |
|---|---|---|
| **H3** corr. das ações EB × LMP > 0,9 | mediana r = 0,598 (por semente: 0,657 · 0,558 · 0,555 · 0,454 · 0,551 · 0,643 · 0,621 · 0,575 · 0,629 · 0,665) | Holm p = 1,00 → **NÃO confirmada** |
| **H4** drift \|π₅₀₀ − π₀\| < σ_a | mediana 0,0122; máx. 0,0189 (≈ 1/4 do ruído) | Holm p < 0,001 → **CONFIRMADA** |
| **H5** S_swap \|π₅₀₀(s) − π₅₀₀(s′)\| < σ_a | mediana 0,0210; máx. 0,0290 | Holm p < 0,001 → **CONFIRMADA** |

**Descritivo:**

| ataque | drift | sd_estados | S_swap | S_shuffle |
|---|---|---|---|---|
| EB | 0,0124 | 0,0676 | 0,0208 | 0,0284 |
| LMP | 0,0120 | 0,0095 | 0,0210 | 0,0110 |

sd_estados é o desvio da saída de π₅₀₀ entre os estados das rodadas 401–500. No MNIST (B2.2): drift 0,0097, S_swap 0,0055, S_shuffle 0,0061.

**Leitura:**
- **H4 se repete:** em 500 rodadas a política se afasta da inicialização cerca de 1/4 do próprio ruído de exploração, uma ordem parecida com a do MNIST.
- **H5 se repete na regra pré-registrada:** trocar a observação pela do outro ataque muda a ação bem menos que σ_a. Em valor absoluto, porém, o S_swap é ~4× o do MNIST (0,021 contra 0,0055).
- **Sob EB, π₅₀₀ varia mais entre estados (sd 0,068 > σ_a)** que sob LMP (0,0095). A explicação mais provável está no §3: com resets a cada ~23 rodadas, os estados observados mudam muito ao longo do run e a mesma rede quase parada produz saídas mais espalhadas. Isso é exploratório. A variação ainda não vira diferença de desempenho (§1).
- **H3 falha:** as ações executadas sob EB e LMP com a mesma semente correlacionam ~0,6, contra ~0,99 no MNIST. No B2.1 já se notava que essa correlação vem da **sequência de ruído compartilhada** e cai quando os runs têm histórias de reset diferentes (s110, r = 0,899). No BloodMNIST os resets são ~20 por run, contra 0,2–2,2 no MNIST (§3), e é isso que dessincroniza os runs. Essa interpretação é exploratória: o resultado confirmatório é só que H3 não se repete.

## 3. Achados exploratórios (não confirmatórios): regime de resets periódicos

- **Todos os 40 runs ficam num regime de resets periódicos.**
  - Média de resets por run: ~20 (fixed 19,95; td3 21,05).
  - O intervalo mediano entre resets é de **23–24 rodadas**, igual nas quatro células.
  - **100% dos runs** têm reset dentro da janela 401–500.
  - No MNIST (B2.1), a média era de 0,2–2,2 resets por run.
- **Nível de acurácia** (média da primária):

  | condição | EB | LMP |
  |---|---|---|
  | fixed | 32,7% | 30,9% |
  | td3 | 31,9% | 31,3% |

  Isso fica acima do FedAvg sob ataque no B3.0 (17–21%), mas muito abaixo do FedAvg sem ataque (77,8%). **A equivalência do §1 é entre dois sistemas que operam no mesmo regime degradado:** nem a fixa nem o TD3 seguram o BloodMNIST sob LMP ou EB no ambiente oficial.
- **Relação com o Adendo 2:** a escala de recompensa é ~3× menor no BloodMNIST, e o limiar de reset (−80) foi mantido. O que se observa não é ausência de resets, mas resets regulares nas duas condições. O mecanismo exato (por que o ciclo tem ~23 rodadas) não foi investigado.
- **td3 contra fixed em resets:** ~1 reset a mais por run no td3 (Wilcoxon pareado descritivo p = 0,044). A diferença é pequena diante do regime comum e não aparece na acurácia.
- **Massa de peso nos atacantes:** mediana ~0,016 nas quatro células, contra ≤ 0,0001 no MNIST. O esqueleto de agregação deixa passar um pouco de peso aos atacantes no BloodMNIST, e de forma igual com ou sem TD3.

## 4. Consequências (Portão B′, roadmap e pré-registro §7)

- **Portão B′: a regra é atendida.** A equivalência (H1) e o mecanismo (H4, H5) se repetem.
  - O P2 pode afirmar, em dois datasets (um deles de saúde), que o TD3 do AdaAggRL publicado é equivalente a uma ação fixa e que a política não se afasta da inicialização nem passa a depender da entrada.
- **Ressalvas obrigatórias ao citar o B3.1:**
  1. A margem de ±3,0 p.p. é uma **equivalência fraca** (Adendo 1), embora o IC90 observado fique dentro de ±1 p.p.
  2. A equivalência acontece num **regime degradado e de resets periódicos** (~31% de acurácia, um reset a cada ~23 rodadas). Ela diz que o TD3 não acrescenta nada à fixa, **não** que algum dos dois defende bem o BloodMNIST no ambiente oficial.
  3. **H3 não se repete.** A identidade quase perfeita das ações entre ataques, vista no MNIST, é específica do regime com poucos resets e não deve ser apresentada como propriedade geral.
- **Não altera** as conclusões do P2 no MNIST (B2.1–B2.8). Estende o escopo do achado principal a um dataset médico.

## 5. Arquivos

- `analise.txt`: saída integral da análise pré-registrada.
- `resumo_runs.csv`: por run, a primária, as AUCs, os resets, o reset em 401–500 e a massa nos atacantes.
- `mecanismo.csv`: por run td3, drift, sd_estados, S_swap e S_shuffle.
- `raw/`: os 40 JSONs, as `obs` e os checkpoints do ator. Os checkpoints intermediários não são versionados (pré-registro §8).
- Os eventos da janela de execução ficam em `results/janela_execucao.log`.
