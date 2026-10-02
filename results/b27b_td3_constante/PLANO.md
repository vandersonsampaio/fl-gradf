# Plano — B2.7a/b: o TD3 em H = 150 depende do estado ou só achou uma constante melhor?

**Status:** FINAL antes de qualquer run e **antes de ver o B2.7a** (hash em `PLANO.sha256`). **Exploratório.** O código ainda não existe: ele será escrito, testado e terá o hash registrado em adendo **antes** do disparo, sem mudar nada deste plano.
**Data:** 2026-10-02
**Roadmap:** `references/roadmap_tese_gradf_v4_1.md` §4.1 (B2.7a, B2.7b, B2.7c) e §7.2 (item 1).
**Antecedente:** B2.7 (`results/b27_horizonte/`). O TD3 (`td3_ref`) passou no critério só em 1 de 8 células em H = 150 (`label_flipping` α 0,05, +6,65 p.p.), entre 72 testes sem correção. Além disso, a comparação foi contra o **centro** (`fixed`, b = 0,5), enquanto LinUCB e DQN foram comparados contra a **melhor constante em retrospecto**.

## 1. Células, horizonte e sementes

- **Células:** as 3 com inclinação positiva do Δ(TD3 − fixed) por log H no B2.7:
  - `label_flipping` α 0,05;
  - `label_flipping` α 0,1;
  - `low_mag_backdoor` α 0,05.
- **H = 150**, sementes **42–51** (as do B2.7, para o pareamento).
- **Regime:** idêntico ao B2.7: learner da ablação, MNIST logístico, 10 clientes, bizantinos [0, 1], root 100, `keras.utils.set_random_seed(seed)` antes de cada sistema.

## 2. B2.7a: mecanismo do TD3 (descritivo)

O B2.7 não gravou ações nem os parâmetros do ator. Por isso o `td3_ref` é rodado de novo nas 3 células × 10 sementes (**30 runs**), com registro por rodada:
- estado médio (`mean_state`);
- ação executada;
- parâmetros do ator (`W_actor`, `b_actor`) nas rodadas 0, 50, 100 e 150.

**Verificação:** a acurácia em H = 15, 50 e 150 deve reproduzir exatamente (|Δ| < 1e-9) a do `td3_ref` no `grade_raw.csv` do B2.7. Se não reproduzir, o B2.7a é invalidado e reportado como tal.

**Medidas por run**, sobre os estados observados nas rodadas 101–150, com π determinística (sem ruído):
- **drift** = média de |π₁₅₀(s) − π₀(s)|, sobre estados e as 5 dimensões;
- **sd_estados** = média, sobre as 5 dimensões, do desvio-padrão de π₁₅₀(s) entre os estados;
- **ação média aprendida** a_TD3 = média de π₁₅₀(s) sobre esses estados (vetor de 5 dimensões), e a distância dela ao centro [0,5; 0,5; 0,5; 0,5; 0,5].

**Referência:** σ_a = **0,15**, o desvio-padrão do ruído de exploração do agente TD3 do framework próprio (`exploration_sigma`), já em unidades de ação [0; 1]. Ele é diferente do σ_a = 0,0475 do código oficial.

## 3. B2.7b: o TD3 contra a melhor constante em retrospecto

**Braços constantes**, nas mesmas 3 células, H = 150, sementes 42–51. É o mesmo esqueleto do `fixed`, com detector completo, ação constante e sem TD3:
- `fixed_b025`: a = [0,5; 0,5; 0,5; 0,5], b = 0,25;
- `fixed_b075`: a = [0,5; 0,5; 0,5; 0,5], b = 0,75;
- `const_td3`: ação constante = a_TD3 do **mesmo** run do B2.7a (célula × semente), ou seja, a constante que o próprio TD3 aprendeu naquele run.

O centro (`fixed`, b = 0,5) vem do B2.7. Total: 3 × 30 = **90 runs**.

**Melhor constante em retrospecto:** por célula, o braço de maior acurácia média em H = 150 entre {`fixed`, `fixed_b025`, `fixed_b075`, `const_td3`}. É uma referência otimista para o fixo, a mesma assimetria exigida do LinUCB e do DQN no B2.7.

**Critério (fixado agora, antes do B2.7a):** **"o TD3 aprende algo além de uma constante"** se Δ = TD3 − melhor constante (pareado por semente, IC95 t) for > 0 com IC95 inteiro acima de 0 em **pelo menos 2 das 3 células**.

**Também reportados:**
- Δ contra cada braço;
- qual braço é o melhor por célula;
- Δ(`const_td3` − `fixed`), que indica se a constante aprendida já explica o ganho do TD3 sobre o centro.

## 4. B2.7c (condicional)

O B2.7c (confirmação: 3 células, H = 150, **sementes 82–91**, pré-registrado à parte) só roda se **as duas** condições valerem:
1. **B2.7a:** a mediana entre sementes de sd_estados > σ_a = 0,15 em pelo menos 2 das 3 células;
2. **B2.7b:** o critério do §3 é atendido.

Se o B2.7c rodar, ele tem prioridade sobre o C0b (roadmap: pausar o C0b).

## 5. Leitura para o P2

- **B2.7b não atendido:** reportar "o ganho do TD3 em horizonte longo equivale a ajustar uma constante (o limiar)", com o B2.7a como mecanismo.
- **B2.7b atendido, mas sd_estados ≤ σ_a:** o TD3 achou uma ação melhor sem depender do estado. Reportar como achado exploratório, sem confirmação.
- **As duas condições:** rodar o B2.7c antes de qualquer afirmação.

## 6. Custo e regras

- **Custo:** B2.7a, 30 runs de ~600 s; B2.7b, 90 runs. Ao todo ~1,5–2 h de relógio com 10 processos na CPU (`nice 19`, 2 threads, sem GPU).
- **Ordem obrigatória:**
  1. este plano commitado;
  2. código com hash no adendo;
  3. B2.7a;
  4. B2.7b, que depende da a_TD3 do B2.7a.

  O critério do B2.7b não muda depois de ver o B2.7a.
- Nenhuma acurácia do B2.7b é inspecionada antes do fim da grade.
- Logs não versionados.
