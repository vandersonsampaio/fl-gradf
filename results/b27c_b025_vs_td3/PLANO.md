# Plano — B2.7c (reformulado): o limiar fixo b = 0,25 supera o TD3 em H = 150?

**Status:** **CONFIRMATÓRIO**. Plano final, escrito antes do código e de qualquer run (hash em `PLANO.sha256`). O código será escrito, testado em smoke e terá o hash registrado em adendo **antes** do disparo, sem mudar nada deste plano.
**Data:** 2026-10-11
**Roadmap:** `references/roadmap_tese_gradf_v4_1.md` (§4.1 B2.7c; sementes 82–91 reservadas). O B2.7c original (confirmar que o TD3 aprende) foi cancelado no B2.7a (condição 1: 0/3). As sementes ficaram livres e são usadas aqui para outra pergunta.
**Antecedentes:**
- **B2.6** (`results/b26_decomposicao/`, sementes 52–61): o limiar b é o componente de maior efeito. b = 0,25 é a melhor variante (+1,6 p.p. sobre o centro) e b = 0,75 a pior (−10,5 p.p.).
- **B2.7b** (`results/b27b_td3_constante/`, sementes 42–51): td3_ref − fixed_b025 = −6,9 / −4,4 / −4,4 p.p. nas 3 células, com o IC95 inteiro abaixo de 0. **Limitação declarada lá:** "b = 0,25 é escolhido em retrospecto".

## 0. Por que este experimento

O B2.7b declarou que "o TD3 perde para uma constante simples", mas a constante foi escolhida depois de ver os dados (melhor em retrospecto entre 4 braços, com as mesmas sementes). Este experimento **resolve** a questão: fixa b = 0,25 **antes**, a partir do B2.6 (sementes disjuntas), e testa contra o td3_ref em sementes novas (82–91), com teste unilateral pré-registrado.

## 1. Sistemas, células, horizonte e sementes

| sistema | definição |
|---|---|
| **td3_ref** | learner da ablação (`scripts/frente1_ablacao_adaaggrl.build_learner_cls`), modo `td3`, idêntico ao B2.7/B2.7a |
| **fixed_b025** | mesmo esqueleto, ação constante a = [0,5; 0,5; 0,5; 0,5], b = 0,25, sem TD3 (o mesmo caminho do `fixed_b025` do B2.7b) |

- **Células:** as mesmas 3 do B2.7a/b, escolhidas no B2.7 por favorecerem o TD3 (o que torna o teste conservador para a hipótese):
  - `label_flipping` α 0,05;
  - `label_flipping` α 0,1;
  - `low_mag_backdoor` α 0,05.
- **H = 150.** A acurácia é a do modelo global ao fim da rodada 150. Também são gravadas H = 15 e 50, só como descritivas.
- **Sementes 82–91** (n = 10 por célula), pareadas por semente entre os dois sistemas.
- **Regime:** idêntico ao B2.7: MNIST logístico, 10 clientes, bizantinos [0, 1], root 100 (`root_seed = seed`), e `keras.utils.set_random_seed(seed)` antes de cada sistema.
- **Grade:** 2 × 3 × 10 = **60 runs**.

## 2. Hipótese e testes (fixados agora)

Unidade: semente. D = acc(fixed_b025) − acc(td3_ref) em H = 150, pareado por semente.

**H1 (primária, por célula):** E[D] > 0.
- **Teste:** t pareado **unilateral** (H0: E[D] ≤ 0), n = 10, gl = 9.
- **Multiplicidade:** Holm–Bonferroni sobre as 3 células, com α de família = 0,05 (unilateral).
- **Veredito:**
  - **"b = 0,25 supera o TD3" (H1 confirmada):** Holm rejeita H0 em **≥ 2 das 3 células**, e nenhuma célula tem D médio < 0;
  - **"confirmada em todas as células":** Holm rejeita nas 3;
  - **"não confirmada":** caso contrário. Não confirmar **não** é evidência de equivalência e não será reportado assim.

**H1b (secundária, agregada):** E[D̄] > 0, em que D̄ é a média de D sobre as 3 células, por semente (n = 10). Teste t pareado unilateral, α = 0,05, sem correção (é um único teste secundário).

**Reportados (descritivos, sem veredito):**
- D médio, IC95 t bilateral e IC90 unilateral inferior, por célula;
- p de Wilcoxon de postos sinalizados unilateral, como robustez à normalidade;
- d_z de Cohen;
- D em H = 15 e H = 50;
- comparação dos D com os do B2.7b (sementes 42–51), para avaliar se o efeito encolheu (viés de seleção do b em retrospecto).

**Poder** (sd por célula do B2.7b, 3,3–5,3 p.p.):
- se o efeito do B2.7b se repetir: 0,80–0,94 por célula no α de Holm mais rigoroso (0,0167), e 0,93–0,99 em α = 0,05;
- se o efeito cair pela metade: 0,26–0,40 e 0,47–0,63.

O teste agregado H1b existe para esse segundo cenário.

## 3. Verificações (antes do disparo, registradas no adendo)

Smoke de 15 rodadas, semente 42, `label_flipping` α 0,05, pelo código novo:
- o td3_ref reproduz o `td3_ref` do B2.7 em H = 15 (|Δ| = 0);
- o fixed_b025 reproduz o `fixed_b025` do B2.7b em H = 15 (|Δ| = 0).

Se alguma falhar, o código é corrigido **antes** do congelamento. O plano não muda.

## 4. Regras

- Ordem obrigatória:
  1. este plano com hash;
  2. código + smoke + hash no adendo;
  3. grade.
- Nenhuma acurácia das sementes 82–91 é inspecionada antes do fim da grade completa (60/60). Run que falhar é rodado de novo com a mesma semente. Se não houver como rodar de novo, a célula é analisada com o n disponível, o que é declarado.
- **Custo:** ~30 min por run, com 12 processos na CPU (`nice 19`, 2 threads, sem GPU). São 5 ondas, ~2,5–3 h de relógio. Respeita a janela seg–sex 7h–18h (`scripts/janela_execucao.sh`).
- Sem commit. Os arquivos ficam untracked até o autor decidir.
