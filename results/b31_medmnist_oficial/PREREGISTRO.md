# Pré-registro — B3.1 + B3.2: fixed contra TD3 no AdaAggRL oficial, BloodMNIST

**Status:** FINAL, congelado **antes de qualquer run do B3.0 e do B3.1** (hash em `PREREGISTRO.sha256`). Mudanças depois disso só como adendo datado e hasheado.
**Data:** 2026-10-04
**Roadmap:** `references/roadmap_tese_gradf_v4_1.md` §4.1 (B3.0, B3.1 + B3.2) e `references/roadmap_adendo_saude.md` §3 (Fase B′).
**Antecedentes:**
- B2.1 + B2.2 (`results/b21_replicacao_oficial/`): equivalência fixed ≈ td3 no MNIST; política parada e independente da entrada.
- B2.4r (`results/b24r_limiar_oficial/`): o código oficial na GPU não é bit-reprodutível, e um reset tardio isolado muda a métrica primária de uma semente em ~8 p.p.

## 1. Perguntas

- **B3.1:** no código e no horizonte publicados, com um dataset de saúde (BloodMNIST), a ação fixa é equivalente ao TD3 do AdaAggRL dentro da margem M (§3)?
- **B3.2:** a política TD3 aprendida continua essencialmente a inicial e independente da entrada (H3–H5 do B2.2)?

## 2. Dataset e adaptação (feita no B3.0, congelada antes do B3.1)

- **BloodMNIST** (MedMNIST v2): 28×28 RGB, 8 classes. Splits oficiais de treino e teste; o de validação não é usado.
- **Regime oficial inalterado:** 100 clientes, 10% por rodada, 20 atacantes, q = 0,5, lr 0,05, 500 rodadas, partição `_build_groups_by_q` do código oficial.
- **Adaptação mínima,** feita fora do código oficial (shim ou monkeypatch no runner, como nos experimentos anteriores; `external/AdaAggRL` não é editado):
  - ramo BloodMNIST no `construct_dataloaders`;
  - modelo oficial com 3 canais de entrada e 8 saídas;
  - **extrator de features novo** para a inversão de gradiente e as pistas MMD, porque o `extract_feature.pt` oficial é MNIST com 1 canal. Ele é treinado uma vez no B3.0 com a mesma arquitetura e procedimento do oficial (adaptados a 3 canais e 8 classes), congelado e com o hash registrado em adendo antes do B3.1.
- Qualquer outra mudança necessária é listada no adendo do B3.0, com justificativa.

## 3. O que o B3.0 precisa entregar, e as regras que usam isso (fixadas agora)

**B3.0 (sanity, desenvolvimento):** FedAvg com agregação uniforme (o modo `fedavg` do runner do B2.5) por 500 rodadas, sementes **130–132**:
- BloodMNIST: sem ataque, LMP e EB (9 runs);
- **MNIST: sem ataque** (3 runs), como referência para a regra da margem.

Métrica de cada run: T = mediana da acurácia nas rodadas 401–500.

**Regra de convergência (portão do B3.0):**
- O FedAvg sem ataque no BloodMNIST "converge" se a média de T nas 3 sementes for ≥ 50% (4× o acaso de 12,5%) **e** a média das acurácias nas rodadas 451–500 diferir da média nas 401–450 em menos de 2 p.p.
- Se não convergir: **parar e consultar o autor** antes do B3.1.

**Regra de inclusão dos ataques:**
- Um ataque entra no B3.1 se o FedAvg sob esse ataque ficar, na média das 3 sementes, **≥ 5 p.p. abaixo** do FedAvg sem ataque (T). É o mesmo limiar do sanity do B2.5.
- Se só um ataque passar, o B3.1 roda só com ele (n = 10) e isso é declarado.
- Se nenhum passar: parar e consultar o autor.

**Regra da margem M (equivalência):**
- No MNIST, a margem de ±1,0 p.p. corresponde ao nível de erro de um FedAvg sem ataque. No BloodMNIST, ela escala com o erro relativo:

  M = 1,0 p.p. × (100 − T_Blood) / (100 − T_MNIST)

  - T_Blood e T_MNIST são as médias de T do FedAvg sem ataque nas sementes 130–132.
  - M é arredondada **para cima** ao próximo múltiplo de 0,25 p.p. e limitada a **[1,0; 3,0] p.p.**
  - Se a fórmula der mais de 3,0, M = 3,0, com a declaração de que a equivalência nessa margem é fraca.
- M é calculada **mecanicamente** pelo script de análise do B3.0 e registrada em adendo (com hash e commit) **antes** de disparar o B3.1. Nenhum run de fixed ou td3 entra nessa conta.

## 4. Grade do B3.1 + B3.2

- **Condições:** `fixed` = [0,475]×5 (o centro, como no B2.1) e `td3` (como no `main.py` oficial: lr 1e-5, ruído 0,1, `train_freq` 3, `batch_size` 64, `buffer_size` 1000, rede [256, 128]).
- **Ataques:** LMP e EB (os que passarem na regra do §3).
- **Sementes:** **135–144**. Total: 10 × 2 × 2 = **40 runs** (ou 20 se só um ataque passar).
- **Execução intercalada:** a fila alterna as condições por semente e ataque (td3 LMP, fixed LMP, td3 EB, fixed EB, para cada semente). Assim cada leva paralela da GPU mistura fixed e td3, e as duas condições ficam sujeitas às mesmas variações de carga, temperatura e ordem. É proibido rodar todos os runs de uma condição antes da outra.
- **Registro:** igual ao B2.1:
  - estados observados antes de cada ação;
  - checkpoints do ator nos passos 0, 50, …, 500;
  - massa de peso nos atacantes;
  - resets;
  - parciais a cada 25 rodadas.
- **Truncagem:** 500 passos (o td3 do SB3 roda 501).

## 5. Métricas e hipóteses

**Unidade:** par ataque × semente (n = 20; ou n = 10 com um só ataque). D = métrica(fixed) − métrica(td3).

**Métrica primária:** mediana da acurácia nas rodadas 401–500 (a mesma do B2.1).

**Sensibilidade por AUC (obrigatória, reportada junto):** área sob a curva de acurácia, normalizada:
- média da acurácia nas rodadas 1–500;
- média nas rodadas 251–500.

Ela captura o custo dos resets ao longo de todo o run e não depende de onde cai a janela.

**H1 (B3.1).** A ação fixa é equivalente ao TD3.
- TOST pareado (t), margem ±M, α = 0,05, na primária → **equivalência confirmada**. Reportar IC90 e IC95.
- **Sensibilidade:** a mesma TOST nas duas AUCs. Se a primária e as AUCs discordarem no veredito, o resultado é reportado como **"sensível a resets"**, e a equivalência só é afirmada com a ressalva.
- Wilcoxon pareado p < 0,05 com D < 0 → **o TD3 contribui**; com D > 0 → a fixa é melhor ("não contribui").
- Nenhum → **inconclusivo**; reportar Δ, IC90, IC95 e d.
- **Por ataque** (descritivo): Δ, IC95, d, Wilcoxon com Holm.

**B3.2 (mecanismo, família H3–H5, Holm, α = 0,05, unilaterais; σ_a = 0,1 × 0,95 / 2 = 0,0475; idênticas ao B2.2):**
- **H3:** correlação das ações executadas entre EB e LMP da mesma semente (rodadas 101–500) > 0,9. Só se os dois ataques entrarem.
- **H4:** drift = média |π₅₀₀(s) − π₀(s)| sobre os estados das rodadas 401–500 < σ_a.
- **H5:** S_swap = média |π₅₀₀(s_t) − π₅₀₀(s′_t)| com o estado do outro ataque < σ_a. Só se os dois ataques entrarem; senão, reportar o S_shuffle como descritivo.

**Resets:** são consequência da ação e **contam como resultado**. Nunca são excluídos da primária nem das AUCs. Reportar, por condição, o nº de resets por run e a fração de runs com reset dentro da janela 401–500, com Wilcoxon pareado descritivo do nº de resets.

## 6. Cláusula de não-reprodutibilidade na GPU

- O código oficial na GPU **não é bit-reprodutível** entre execuções (B2.4r ADENDO1 e RESULTADO). A semente fixa a inicialização e a partição, não a trajetória. O pareamento por semente pareia a **configuração**, e cada run é uma amostra da distribuição daquela configuração.
- **Nenhum run com JSON final completo é rodado de novo,** por nenhum motivo (resultado inesperado, reset, colapso). Só runs que **falharem sem JSON final** (erro de ambiente, processo morto) são repetidos com a mesma semente, e cada repetição é registrada no `grid.log` e no RESULTADO.
- A variabilidade entre execuções (no B2.4r, ~8,6 p.p. na primária de uma semente por causa de um reset tardio) é declarada como fonte de variância incluída nos IC. É também a razão de a sensibilidade por AUC ser obrigatória.

## 7. Portão B′ (roadmap)

- **Equivalência (H1) e mecanismo (H4, H5) se repetem** → o P2 afirma o resultado em dois datasets, um deles de saúde.
- **Não se repetem** → reportar como limite de escopo do achado, com a direção e o mecanismo observados.
- **"Sensível a resets"** → o P2 afirma o mecanismo e reporta a equivalência com a ressalva.

## 8. Regras

- A análise do B3.1 + B3.2 é pré-escrita (script com hash em adendo antes do disparo) e rodada uma única vez, com a grade completa.
- Nenhuma acurácia do B3.1 é inspecionada antes do fim da grade. O monitoramento reporta só saúde e progresso.
- **Ordem obrigatória:**
  1. este pré-registro commitado;
  2. B3.0 implementado e rodado;
  3. adendo com dataset, extrator, ataques incluídos, M e hashes;
  4. commit;
  5. disparo do B3.1.
- Custo: o do B3.0 calibra o do B3.1 (estimativa do roadmap: ~42 h de GPU para o BloodMNIST).
- Logs e checkpoints intermediários do ator não são versionados (só os passos 0 e 500).
