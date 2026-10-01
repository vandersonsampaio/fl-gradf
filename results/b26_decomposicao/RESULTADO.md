# Resultado — B2.6: decomposição do esqueleto (memória, forma da ponderação, sinal)

**Data:** 2026-10-01
**Pré-registro:** `PREREGISTRO.md` (hash em `PREREGISTRO.sha256`; `PREREGISTRO.md`, `scripts/b26_decomposicao.py` e `scripts/run_grid_b26.sh` conferidos contra o hash antes da análise: OK).
**Grade:** 80/80 jobs = 1.680 runs (8 variantes × sementes 52–61 × 21 células), 30/09 19:15 → 01/10 03:46, CPU, 0 falhas.
**Análise:** `scripts/b26_decomposicao.py analisar` → `analise.txt`, `por_semente.csv`. Unidade: semente (média das 19 células válidas), n = 10.

---

## 1. Hipóteses confirmatórias (Holm sobre H1–H4, α = 0,05)

| hipótese | Δ (p.p.) | IC95 | IC90 | teste | Holm | resultado |
|---|---|---|---|---|---|---|
| **H1** a memória contribui (`sr_only` − `sr_nomem` > 0) | −0,35 | (−0,82; +0,12) | — | Wilcoxon unilateral, p = 0,97 | 1,00 | **NÃO confirmada** |
| **H2** a força da memória importa (`sr_only` − `sr_memof` ≠ 0) | −0,63 | (−1,08; −0,18) | — | Wilcoxon bilateral, p = 0,010 | **0,039** | **CONFIRMADA** |
| **H3** máscara binária ≈ suave (`sr_bin` − `sr_only`, TOST ±1 p.p.) | +1,02 | (+0,47; +1,57) | (+0,58; +1,47) | TOST, p = 0,54 | 1,00 | **NÃO confirmada** |
| **H4** `cos_server` ≈ S_R (`cosserver_only` − `sr_only`, TOST ±1 p.p.) | +0,42 | (−0,22; +1,06) | (−0,09; +0,94) | TOST, p = 0,035 | 0,105 | **NÃO confirmada** (passa sem correção, cai com Holm) |

## 2. Leitura de cada hipótese

**H1, memória.** Não há evidência de que a penalidade persistente (λ = 2) ajude **em média**. A estimativa pontual favorece desligá-la (−0,35 p.p.). O efeito é **heterogêneo por ataque** (descritivo, Holm sobre 7 ataques) e se cancela no agregado:
- ajuda em `sign_flipping` (+3,4 p.p.);
- atrapalha em `low_mag_backdoor` (−3,2), `fltrust_aligned` α = 0,5 (−1,2) e `label_flipping` (−1,9, não significativo após Holm);
- é neutra nos demais.

**H2, força da memória.** **Confirmada:** a memória fraca do código oficial (0,9^flag, λ ≈ 1,11) é **melhor** que a λ = 2 da nossa reprodução (+0,63 p.p.). O perfil por ataque espelha o do H1: a memória forte prejudica `low_mag_backdoor` e `fltrust_aligned` e ajuda `sign_flipping`. Consequência: a λ = 2 **não era neutra** (auditoria §1, item 2). A memória mais forte não explica headroom; tende a atrapalhar.

**H3, forma da ponderação.** A equivalência **não** foi confirmada porque a diferença **favorece a máscara binária** (+1,02 p.p., IC95 inteiro acima de 0, estimativa pontual acima da margem). *Fora do teste pré-registrado* (o critério era equivalência): a ponderação contínua (suave) **não é vantagem**; um filtro binário com o mesmo limiar faz **igual ou melhor** em 5 dos 7 ataques (`label_flipping` +5,1; `low_mag_backdoor` +2,2; `fltrust_aligned` +1,1; `krum_collusion` +0,5) e pior em `sign_flipping` (−1,9) e `gaussian_noise` (−0,1).

**H4, sinal.** Equivalência **não confirmada** após Holm. No agregado a diferença é pequena (+0,42 p.p., IC90 dentro de ±1 p.p.), mas **esconde perfis opostos por ataque** (todos com Holm < 0,05, exceto `sign_flipping`):
- **S_R vence** nos ataques de modelo, sobretudo em α = 0,05: `gaussian_noise` −4,6; `krum_collusion` −3,5; `trim_attack` −2,7. E vence em `fltrust_aligned` α = 0,5 (−8,6), onde o atacante se alinha exatamente ao update do servidor, que é o próprio sinal do `cos_server`;
- **`cos_server` vence** nos ataques de rótulo e backdoor: `label_flipping` **+11,5** (por exemplo, α = 0,1: 83,4 contra 65,3) e `low_mag_backdoor` +3,8.

Os sinais são **complementares**, e não equivalentes. Isso confirma, agora em sementes novas, o perfil exploratório visto na ablação.

## 3. Exploratório (sem critério)

- **Sinais combinados** (`sr_cosserver` = 0,5 S_R + 0,5 `cos_server`): +1,21 p.p. sobre `sr_only` (IC95 −0,05 a +2,47) e +0,62 sobre o melhor sinal único por semente (IC95 −0,51 a +1,74). A média simples **não** captura a complementaridade, porque herda as perdas de cada sinal (por exemplo, α = 0,05 `gaussian_noise`: 69,9 contra 80,4 do S_R). Uma combinação que escolha ou pondere por ataque fica para a Fase C (direção 2).
- **Limiar b:**
  - b = 0,25: **+1,57 p.p.** (IC95 +1,22 a +1,92);
  - b = 0,75: **−10,45 p.p.** (IC95 −11,4 a −9,5).

  **O limiar é o componente de maior efeito** entre todos os testados, uma ordem de grandeza acima de memória, forma e sinal no agregado. É coerente com o B2.3, em que os colapsos vieram de a₅ ≈ 0, e justifica o B2.4.

## 4. Consequências para o P2 (revisão dos princípios de projeto)

| princípio no roadmap v3 §4.4 | depois do B2.6 |
|---|---|
| "a memória importa" | **Não se sustenta no agregado.** O efeito depende do ataque e se cancela; memória forte (λ = 2) é pior que a fraca do código oficial (H2). Reformular: *a memória não explica o headroom; seu efeito é específico do ataque.* |
| "contínuo contra discreto" (P1) | **A ponderação contínua não é vantagem:** a máscara binária é igual ou melhor. Junto com o B2.8, a diferença P1 é de **granularidade por cliente**, não de continuidade. |
| "o sinal por cliente importa onde separa" | **Reforçado e refinado:** S_R e `cos_server` separam ataques **diferentes**. Nenhum sinal único domina. |
| "o limiar é o componente decisivo" | **Reforçado:** b tem o maior efeito medido (−10,5 a +1,6 p.p.). |

Implicações para o P3:
- **Direção 2 (sinais complementares):** fortemente motivada, mas a combinação precisa ser **seletiva**, não média simples.
- **R3:** o `cos_server` dispensa a inversão, mas exige o root e é vulnerável a ataques alinhados ao servidor.
- **R6:** o limiar como parâmetro central de interpretabilidade.

## 5. Limitações

- 15 rodadas; MNIST; regime do framework próprio (logística, 10 clientes).
- O teste de H3 era de equivalência; a direção "binária melhor" é leitura **não pré-registrada** do IC.
- Os efeitos por ataque usam Holm sobre 7 ataques, mas são descritivos (a família confirmatória é H1–H4).
- `sr_cosserver` usa uma regra de combinação ingênua (média).
