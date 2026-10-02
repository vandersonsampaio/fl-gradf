# Plano — B2.4r: varredura reduzida do limiar a₅ no AdaAggRL oficial

**Status:** FINAL antes de qualquer run (hash em `PLANO.sha256`). **Exploratório.** O código ainda não existe: ele será escrito, testado e terá o hash registrado em adendo **antes** do disparo, sem mudar nada deste plano.
**Data:** 2026-10-02
**Roadmap:** `references/roadmap_tese_gradf_v4_1.md` §4.1 (B2.4r) e §7.2 (item 3).

## 1. Perguntas

1. **Efeito causal do limiar no código publicado:** mudar só a₅ (o limiar δ = max(k)·a₅), com o resto da ação no centro, muda a acurácia sob EB? No framework próprio o limiar é o maior efeito (B2.6: b = 0,75 custa 10,5 p.p.). No código oficial, a evidência atual são só 2 colapsos acidentais do B2.3.
2. **Existe uma constante melhor que o centro?** É a mesma pergunta do B2.7b, agora no código oficial.

## 2. Desenho

- **Código:** oficial (`external/AdaAggRL`, commit `27b9c18`), venv `external/.venv_adaaggrl`, sem nenhuma edição. O runner reusa o `run_b21.py`/`run_oficial.py` e muda só a ação fixa.
- **Regime:** idêntico ao Passo 2 e ao B2.1 (MNIST, q = 0,5, 100 clientes, 10% por rodada, 20 atacantes, 500 rodadas). Ataque **EB** apenas: o LMP é dispensado por ser insensível a a₅.
- **Condições:** ação fixa [0,475; 0,475; 0,475; 0,475; a₅] com **a₅ ∈ {0; 0,1; 0,25; 0,95}**. O Box oficial é [0; 0,95].
  - O **centro** (a₅ = 0,475) vem dos runs `fixed` EB do Passo 2 (`results/frente1_passo2_oficial/raw/`, mesmas sementes). Ele não é rodado de novo.
  - Com a₅ = 0, o cliente de menor score continua excluído, porque k = 0 após o min-max. Isso é uma propriedade do mecanismo oficial, e não um filtro desligado.
- **Sementes:** 100–104 (as mesmas do Passo 2, para o pareamento com o centro).
- **Total:** 4 × 5 = **20 runs**, 5 em paralelo na GPU, **~28–30 h**.
- **Registro:** o mesmo do Passo 2 e do B2.1 (acurácia, perda, ação, massa nos atacantes, nº de excluídos e de atacantes excluídos, resets). Checkpoints parciais a cada 25 rodadas.

## 3. Verificação antes da grade

Rodar o runner novo com a₅ = 0,475 na semente 100 por **25 rodadas** e comparar com o Passo 2. Ele deve reproduzir exatamente (|Δacc| < 1e-9 por rodada) a trajetória do `fixed` EB, semente 100. Se não reproduzir, o centro é rodado de novo nas 5 sementes (+5 runs), por adendo, antes da grade.

## 4. Métricas

- **Primária:** mediana da acurácia nas rodadas 401–500 (a mesma do B2.1 e do B2.5).
- **Secundária, robusta a reset** (princípio do roadmap §2): média da acurácia nas 500 rodadas (área sob a curva).
- **Mecanismo (descritivas):** nº de resets, massa média de peso nos atacantes reais, nº médio de clientes excluídos e de atacantes excluídos por rodada.

## 5. Critérios (exploratórios; n = 5; sem correção de multiplicidade, 4 comparações)

Para cada a₅: Δ = métrica(a₅) − métrica(centro), pareado por semente, com IC95 t.
- **Q1, o limiar tem efeito causal:** em pelo menos um a₅, |Δ| > 2 p.p. com IC95 que exclui 0, na primária **ou** na secundária. Também reportar a forma da resposta (monotônica ou não) e o mecanismo (exclusões, massa, resets).
- **Q2, existe constante melhor que o centro:** em pelo menos um a₅, Δ > 0 com IC95 > 0 na primária **e** Δ > 0 na secundária.
- Wilcoxon reportado como descritivo: com n = 5, o menor p bilateral possível é 0,0625.
- Cascatas de reset contam como resultado, porque são consequência da ação.

## 6. Leitura

- **Q1 sim:** o limiar é causal também no código publicado, o que fecha a evidência da Parte I. Ele é o componente a explicar (R6) e a proteger (R5) no P3.
- **Q2 sim:** o AdaAggRL publicado opera abaixo da melhor constante e o TD3 não a encontra (B2.1/B2.2). Isso reforça "o aprendizado equivale, no máximo, a ajustar o limiar".
- **Q2 não:** o centro já está perto do ótimo sob EB, e o TD3 não teria o que ganhar ajustando a₅.

## 7. Regras

- Análise pré-escrita (script com hash no adendo), rodada uma vez com a grade completa.
- Nenhuma acurácia é inspecionada antes do fim da grade. O monitoramento reporta só saúde e progresso.
- Runs que falharem por erro de ambiente são repetidos com a mesma semente e registrados.
- Logs e checkpoints não versionados, como nos experimentos anteriores.
