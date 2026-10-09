# Pré-registro: ablação do AdaAggRL (Frente 1, pós-V0)

**Data:** 2026-09-24, escrito **antes** de rodar a grade (passo 3).
**Não commitado, por instrução do autor.** Este arquivo e os resultados ficam isolados em `results/frente1_ablacao_adaaggrl/`. O timestamp do arquivo e a ordem registrada na sessão servem de registro.
**Origem:** `references/frente1_v0_sanidade_extrator.md` §4 (o TD3 não aprende em 15 rodadas, e o S_R domina) e as duas propostas do autor: (1) AdaAggRL sem RL; (2) S_R contra detectores triviais.
**Código:** `scripts/frente1_ablacao_adaaggrl.py`. Não altera nada em `src/`.

## 0. O que já foi visto antes deste registro (transparência)

- **V0:** 1 semente, 3 células, com cues, faixas e ações do TD3 (ver o relatório do V0).
- **Passo 1, AUC dos detectores ao longo da trajetória de referência:** semente 42, **21 células** (`passo1_auc_detectores.csv`). O `cos_median`, declarado antes de ver os dados como o detector trivial da variante confirmatória, teve a **pior** AUC média (0,465). O `cos_server` teve a melhor entre os triviais (0,704).
  - Consequência: `cosmed_only` continua **confirmatória**, como estava declarada. `cosserver_only` entra como **exploratória**, porque foi escolhida depois de ver dados da semente 42, que também faz parte da grade.
- **Checagem de metodologia, que não é resultado:**
  - a trajetória `td3` do script reproduz o AdaAggRL oficial da semente 42 com diferença de 0,0 nas 21 células;
  - a função de headroom reproduz os **9/21** documentados para a referência (e dá 3/21 para o Random).

## 1. Protocolo, idêntico ao da referência

MNIST, root=100, sementes 42–51 (10), α ∈ {0,5; 0,1; 0,05}, 7 ataques (`fltrust_aligned`, `gaussian_noise`, `krum_collusion`, `label_flipping`, `low_mag_backdoor`, `sign_flipping`, `trim_attack`), `n_rounds=15`, `n_clients=10`, bizantinos = clientes {0, 1}. A métrica é a acurácia global da última rodada (média das acurácias de teste por cliente).

**Referência (`td3_ref`):** AdaAggRL com extrator `random`, reaproveitado de `results/tables/exp10_selector_comparison_variantb_full_seed{42..51}_raw.csv`, sem recalcular. É a referência recomendada pelo V0.
**Random:** do mesmo arquivo.
**Melhor regra fixa:** `exp9_dominance_grid_10seeds_ALL_root100_raw.csv`.

## 2. Variantes

Todas mantêm o mecanismo do AdaAggRL: normalização min–max de ŵ = S·a, limiar δ = max(w̃)·b, contador h, penalidade λ^h com λ=2, e agregação ponderada dos parâmetros completos. O extrator `random` é sempre construído, para preservar o RNG e o pareamento por semente.

| variante | score por cliente | ação | inversão de gradiente | status |
|---|---|---|---|---|
| `fixed` | (S_R, S_cl, S_cg, S_lg), como na referência | **a = [0,5; 0,5; 0,5; 0,5], b = 0,5**, sem TD3 | sim | confirmatória (H1) |
| `sr_only` | S_R | a = [1, 0, 0, 0], b = 0,5 | sim, sem extrator nem MMD | confirmatória (H2) |
| `cosmed_only` | cosseno até a mediana coordenada a coordenada | a = [1, 0, 0, 0], b = 0,5 | **não** | confirmatória (H3) |
| `cosserver_only` | cosseno até o update do servidor no root | a = [1, 0, 0, 0], b = 0,5 | não | **exploratória** |

## 3. Hipóteses e testes

**Unidade de análise:** semente. Nos testes agregados, primeiro calcula-se a média por semente sobre as 21 células (n=10), e só depois o teste. Por ataque, a média por semente é sobre os 3 α.

- **Equivalência:** TOST pareado (t) com margem **±0,01** de acurácia, α=0,05. Há equivalência quando o IC 90% da diferença média fica dentro de ±0,01.
- **Diferença:** Wilcoxon signed-rank pareado, bilateral, α=0,05. Também reportamos Δ e o d de Cohen pareado.
- **Por ataque:** Holm-Bonferroni sobre os 7 ataques, separadamente para os p do TOST e para os p do Wilcoxon.
- **Headroom:** contagem de células em que a diferença para a melhor regra fixa supera a soma dos desvios padrão, pelo mesmo critério do exp9 e do Passo Zero. É **descritiva**, não é teste.

**H1. O TD3 não contribui neste regime:** `fixed` − `td3_ref`.
- Agregado equivalente (p_TOST < 0,05) → **H1 confirmada**. O §4 do V0 deixa de ser observacional.
- Wilcoxon agregado com p < 0,05 e `fixed` pior → o TD3 contribui. H1 é refutada.
- Wilcoxon agregado com p < 0,05 e `fixed` melhor → o TD3 atrapalha. H1 é refutada no sentido forte.
- Nenhum dos casos anteriores → **inconclusivo**. A margem de ±0,01 não é resolvível com n=10.

**H2. Os cues MMD não contribuem além do S_R:** `sr_only` − `td3_ref`, com as mesmas regras.

**H3. A inversão de gradiente não contribui (um detector trivial basta):** `cosmed_only` − `td3_ref` e `cosmed_only` − `sr_only`, com as mesmas regras.
- *Previsão registrada depois do passo 1:* H3 deve **falhar** com o `cos_median`, que teve AUC média de 0,465 e AUC ≈ 0 contra ataques informados.
- Se falhar, isso **não** refuta a ideia geral de que "detectores triviais bastam". Refuta apenas este trivial. A variante exploratória `cosserver_only` informa sobre a ideia geral, mas não a confirma.

**Leituras para a Frente 2:** um diferencial de privacidade só é sustentável com uma variante sem inversão que fique **equivalente ou melhor** que `td3_ref` no agregado e que também supere o Random. Um resultado exploratório (`cosserver_only`) exige replicação em sementes novas antes de virar afirmação.

**Comparações descritivas adicionais:** cada variante contra Random, agregado e por ataque, com Holm.

## 4. Não fazer

- Não trocar a margem, os valores de a e b, nem a lista de variantes confirmatórias depois de ver resultados da grade.
- Não reportar `cosserver_only` como confirmatória.
