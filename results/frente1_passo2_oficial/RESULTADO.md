# Resultado — Passo 2: o TD3 contribui no AdaAggRL oficial?

**Data:** 2026-09-27
**Pré-registro:** `PREREGISTRO.md` (hash original e do Adendo 1 em `PREREGISTRO.sha256`)
**Grade:** 30/30 runs (LMP, EB × td3/fixed/random × sementes 100–104; MNIST, q=0,5, 500 rodadas), 2026-09-25 12:46 → 2026-09-27 09:51, 0 falhas.
**Análise:** `scripts/passo2_oficial/analisar.py` → `analise.txt`, `resumo_runs.csv`, `curvas_acc.csv`. Exploratórias rodadas à parte (comandos no §4).

---

## 1. Resultado confirmatório (como pré-registrado)

Métrica primária: acurácia média nas rodadas 451–500 (runs truncados em 500 passos, Adendo 1).

| ataque | semente | fixed | random | td3 |
|---|---|---|---|---|
| EB | 100 | 0,8811 | 0,9163 | 0,8250 |
| EB | 101 | 0,9674 | 0,6995 | 0,9656 |
| EB | 102 | 0,9671 | 0,8142 | 0,9660 |
| EB | 103 | 0,9639 | 0,7969 | 0,9664 |
| EB | 104 | 0,9622 | 0,6630 | 0,9679 |
| LMP | 100 | 0,9659 | 0,9686 | 0,9648 |
| LMP | 101 | 0,9654 | 0,9663 | 0,9696 |
| LMP | 102 | 0,9680 | 0,9679 | 0,9657 |
| LMP | 103 | 0,9631 | 0,9641 | 0,9668 |
| LMP | 104 | 0,9655 | 0,9687 | 0,9675 |

**H1 (fixed − td3), pooled n=10:** Δ = **+0,45 p.p.**, IC95 (−0,87; +1,76), d = +0,24. TOST ±1,0 p.p.: p = 0,18. Wilcoxon: p = 0,56.
→ **VEREDITO PRÉ-REGISTRADO: INCONCLUSIVO.** Não há evidência de que o TD3 seja melhor que a ação fixa (a estimativa pontual favorece a fixa), **mas a equivalência dentro de ±1 p.p. não foi demonstrada**.
- LMP: Δ = −0,13 p.p., IC95 (−0,49; +0,23), Holm p = 0,88.
- EB: Δ = +1,02 p.p., IC95 (−2,19; +4,23), Holm p = 1,00. A largura vem de **um** par (semente 100, ver §2.1).

**H2 (random − td3), pooled n=10:** Δ = −8,00 p.p., IC95 (−17,53; +1,54), d = −0,60. TOST p = 0,93, Wilcoxon p = 0,19 → **INCONCLUSIVO**.
- LMP: Δ = +0,02 p.p. (a ação é irrelevante sob LMP).
- EB: Δ = −16,02 p.p., Holm p = 0,25 (ação aleatória colapsa sob EB em 4/5 sementes, mas n=5 não basta para Wilcoxon + Holm).

## 2. Resultados exploratórios (não confirmatórios)

### 2.1 O único par discrepante é um artefato de reset tardio
O ambiente oficial reinicializa o modelo de FL do zero quando a recompensa < −80 (auditoria §2.5). Em EB, semente 100, **fixed resetou na rodada 421 e td3 na 441** — a janela 451–500 mede a recuperação de um modelo recém-reinicializado, não o regime estável (fixed: mín 0,72 na janela; td3: mín 0,56). Nos outros 9 pares, |Δ| ≤ 0,5 p.p. Excluir o par não é permitido pelo pré-registro; registra-se como a causa da largura do IC.

Contagem de resets extras (5 sementes somadas):

| | fixed | td3 | random |
|---|---|---|---|
| EB | 3 | 14 | 60 |
| LMP | 1 | 3 | 0 |

Os resets do td3 se concentram antes da rodada 122 — na fase de aquecimento de 100 rodadas com ação uniformemente aleatória do SB3. Depois do aquecimento, só 1 reset (s100, rodada 441). **O custo do TD3 sob EB é o aquecimento aleatório, não a política aprendida.**

### 2.2 A política TD3 não se afasta do ponto inicial e não depende do ataque
- Deriva média da ação em relação ao centro do Box (0,475) nas rodadas 101–500: ≤ 0,023 por dimensão (média entre sementes); máximo por semente/dimensão 0,079. Sem tendência entre janelas de 100 rodadas.
- **Mesma semente, ataques diferentes → quase a mesma sequência de ações:** correlação passo a passo das ações após a rodada 100 entre EB e LMP = **0,91 / 0,99 / 0,99 / 0,94 / 0,99** (sementes 100–104), com ações médias iguais até a 3ª casa decimal. Os estados e recompensas diferem muito entre os ataques (p.ex. a regra `sim_lc ≥ 0,9` dispara 1,4×/rodada em EB e ~0 em LMP). A trajetória é determinada pela inicialização da rede e pela sequência de ruído (ambas fixadas pela semente), **não pelo que o agente observa**.
- Coerente com o orçamento calculado na auditoria: ~133 passos de crítico / ~66 de ator com lr 1e-5.

### 2.3 O filtro com memória exclui os atacantes em qualquer condição
Massa de peso agregada aos atacantes reais, média por rodada: ≤ 0,0011 em todas as 6 combinações (inclusive `random`). Sob LMP a ação não importa (as três condições empatam); sob EB, ação aleatória prejudica via resets, e a fixa é tão boa quanto o TD3.

## 3. Leitura e consequências para o plano

**O que se pode afirmar:**
1. (confirmatório) No código e horizonte publicados, **não há evidência de que o TD3 supere uma ação fixa**; a estimativa pontual favorece a fixa (+0,45 p.p.).
2. (confirmatório) A equivalência em ±1 p.p. **não foi estabelecida** com 5 sementes — sobretudo por um reset tardio em EB.
3. (exploratório, forte) A política aprendida **é essencialmente a política inicial mais ruído**: não se afasta do centro e é quase idêntica entre ataques com estados e recompensas muito diferentes.

**O que não se pode afirmar:** "o TD3 é equivalente à ação fixa" (TOST falhou); "o TD3 atrapalha" (nenhum teste significativo).

**Ramos do plano (§6):** nenhum dos dois desfechos pré-registrados se concretizou de forma limpa. A leitura mais honesta é "o RL não contribui neste horizonte, e o mecanismo é que ele não chega a aprender", o que sustenta a tese central em caráter **exploratório**, e não o ramo "o RL só se paga depois de N rodadas" (não há nenhuma tendência de aprendizado ao longo das 500 rodadas).

## 4. Opções de próximo passo (decisão do autor)

1. **Replicação confirmatória da equivalência**, com novo pré-registro: sementes novas (p.ex. 105–114), mesma margem. Declarar como estudo novo, nunca como extensão sequencial das sementes 100–104 sem correção. Considerar pré-registrar também uma métrica robusta a resets (p.ex. janela estável: rodadas sem reset nas 50 anteriores), justificada por §2.1. Custo: ~9 h por lote de 6 runs (só fixed e td3: 20 runs ≈ 30 h).
2. **Confirmar o achado mecanístico (§2.2)** como hipótese pré-registrada: "correlação EB×LMP das ações ≥ 0,9 por semente" e "|deriva| < ruído de exploração". Barato: os dados já existem para as sementes 100–104 (exploratórios), a confirmação exigiria sementes novas — pode rodar junto com a opção 1.
3. **Seguir para os Passos 3–5** tratando o Passo 2 como: "sem evidência de contribuição do TD3; política não aprende no orçamento publicado" (exploratório), e reservar a opção 1 para antes da submissão.

## 4b. Comandos das exploratórias

As checagens do §2 foram rodadas ad hoc sobre `raw/*.json` (resets por run, deriva da ação por janela, correlação EB×LMP das ações por semente). Para reproduzir, ler `steps[:500]` de cada JSON: campos `action`, `acc`, e `resets[1:]`.
