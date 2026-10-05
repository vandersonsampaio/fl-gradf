# Resultado — B2.8s: o veredito da granularidade depende da métrica?

**Data:** 2026-10-05.
- Plano: `PLANO.md`, commit `c15937d`.
- Grade: 210 jobs (21 células × sementes 42–51 × 9 sistemas), de 04/10 19h26 a 05/10 00h12. Sem falhas.
- Saídas: `analise.txt`, `celulas_uniforme.csv`, `celulas_ponderada.csv`, `grade_raw.csv` (com as acurácias por cliente).

## A sensibilidade de métrica é vazia: as duas métricas são idênticas por construção

- **Todos os clientes têm test sets do mesmo tamanho (1000 exemplos)**, em todos os α. O `data/download_datasets.py` divide o test set global do MNIST (10.000) **aleatoriamente e em partes iguais** entre os 10 clientes ("shared test set split equally across clients").
- Logo:
  - a média **ponderada pelo tamanho do test set** é matematicamente igual à média **uniforme** (máx. |Δ| = 2e-16 nos 1.890 runs);
  - **as duas são a acurácia no test set global do MNIST, que é IID e balanceado por classe.** A métrica do C0/B2.7/B2.8/C0b **não é uma média "por cliente" com distribuições locais**; é a acurácia global.
- **O critério formal foi atendido** ("granularidade blindada": mesmo veredito, 100% de concordância), **mas isso não tem conteúdo**: as duas linhas da análise são a mesma métrica. As pequenas diferenças em α 0,5 (≤ 0,002 p.p.) vêm de desempates do oráculo guloso, não da métrica.
- **Erro de desenho, declarado:** o plano partiu da premissa (minha) de que a métrica era uma média uniforme sobre test sets locais de tamanhos diferentes. O inventário de ponderação foi feito; os tamanhos e a origem dos test sets não foram verificados antes do disparo.

## O que a grade mostra mesmo assim

- **Reprodução do B2.8 com ressemeadura por sistema (sementes 42–51):** 14 células-alvo, 7 com o oráculo por rodada abaixo do esqueleto e 2 acima → **"a granularidade explica"**. O B2.8 original tinha 8/14 abaixo, com o mesmo veredito. O veredito resiste à mudança de RNG e de implementação das referências: esqueleto e regras estáticas rodados de novo na mesma grade.

## Consequências

1. **A pergunta de fundo continua aberta, mas reformulada.** A preocupação era: "parte da vantagem do filtro por cliente vem de ponderar os honestos uniformemente, e a métrica premia isso". Com um test set global IID, o mecanismo que favorece a ponderação uniforme não é a métrica por cliente. É o **desbalanceamento de classes** que a ponderação por tamanho de amostra introduz quando os clientes grandes têm distribuição enviesada (Dirichlet α ≤ 0,1), avaliado num teste balanceado. Testar isso de verdade exigiria uma métrica sobre **test sets locais com a distribuição de rótulos de cada cliente**, que não existem no pipeline atual (precisariam ser construídos).
2. **O C0b-iii, como planejado, seria igualmente vazio no lado da métrica:** os pares U e P diferem só no teto, que já foi medido na sensibilidade do C0b. Por isso **a fila foi interrompida antes de dispará-lo** (05/10 00h53), à espera de decisão do autor. O D2 continua.
3. **Precisam de correção** os textos de 04/10 (`VERIFICACOES.md` §3, `CORRECAO_2026-10-04.md` do C0, RESULTADOs do C0b, A0 e B2.7) que dizem "a métrica é a média uniforme das acurácias nos test sets dos clientes". O correto: a métrica é a acurácia no test set global IID; o oráculo ponderado por amostra perde porque, com rótulos enviesados por cliente, dá peso demais às classes dos clientes grandes. **As conclusões numéricas (oráculo uniforme como teto, sensibilidade do C0b) não mudam;** muda a explicação do mecanismo.
