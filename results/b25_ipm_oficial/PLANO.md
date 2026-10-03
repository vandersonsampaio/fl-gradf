# Plano — B2.5: IPM real no AdaAggRL oficial (fixed contra td3)

**Status:** FINAL antes de qualquer run do sanity e da grade (hash em `PLANO.sha256`). Mudanças depois disso só como adendo datado e hasheado.
**Data:** 2026-10-01
**Roadmap:** `references/roadmap_tese_gradf_v4.md` §4.1 (B2.5, prioridade 1): terceiro ataque do paper original; checklist da SLR.
**Antecedente:** B2.1 + B2.2 (`results/b21_replicacao_oficial/`): fixed equivalente ao td3 (±1 p.p.) sob LMP e EB; política TD3 sem drift e sem dependência da entrada.

## 1. Pergunta

Sob um ataque otimizado contra a média (IPM), a ação fixa continua equivalente (±1,0 p.p.) ao TD3 do AdaAggRL no código e no horizonte publicados?

## 2. Achado de auditoria e desvio declarado

- **O IPM oficial é nulo.** Em `exp_environments.step`, a rede é posta em `old_weights` antes de `IPM_attack`, que devolve `craft(old_weights, get_parameters(net), 5, -1)`; como `get_parameters(net) == old_weights`, a diferença é zero e o atacante envia o próprio modelo global. Verificação: `scripts/passo2_oficial/test_b25_ipm.py`, item 2 (norma do update forjado = 0), saída em `teste_unitario.txt`. **Reportado como achado de auditoria**; o IPM oficial não é rodado na grade (seria um ataque de "carona" sem efeito, não o IPM de Xie et al.).
- **Desvio declarado: IPM real** (Xie et al., 2020): delta forjado = −ε × média dos deltas honestos da rodada (atacante onisciente, conhece os updates honestos); peso enviado = `old_weights` + delta forjado; todos os atacantes reais da rodada enviam o mesmo vetor.
- **Implementação sem editar o código oficial** (`scripts/passo2_oficial/run_b25.py`): o ambiente roda no caminho do LMP (o único que recebe a lista de pesos honestos já calculados) e, só neste processo, `exp_environments.LMP_attack` é trocado pela função de IPM real. Os pesos honestos são os dos não atacantes e dos atacantes em primeira participação (que no código oficial treinam honestamente), como no LMP oficial. Todo o resto (agregação, recompensa, resets, sorteios) é o do código oficial.
- **Teste unitário** (item 1 do mesmo script): cos(delta forjado, média honesta) = −1 e razão de normas = ε, para ε = 2 e 10, em pesos sintéticos e no `MNISTClassifier` oficial. **Passou** antes deste plano.

## 3. Escolha de ε (sanity, semente 100, já gasta)

Candidatos: **ε ∈ {2, 10}**. Cinco runs de **100 rodadas**, semente 100 (`run_grid_b25.sh sanity`):
`fedavg` sem ataque; `fedavg` com ε = 2 e 10; `fixed` com ε = 2 e 10.

- `fedavg`: agregação uniforme (troca de `aggeregate` por `average` só neste processo). Sem ataque: mesma partição e mesmos 20 atacantes sorteados (RNG preservado), com a lista de atacantes esvaziada logo após a inicialização.
- **"Degrada o FedAvg":** acurácia média nas rodadas 81–100 pelo menos **5 p.p.** abaixo do run sem ataque (mesma semente).
- **"Não é totalmente excluído":** massa média de peso nos atacantes sob o `fixed` **> 0,01**, calculada só nas rodadas com atacantes reais (referência: ≤ 0,0001 no B2.1).
- **Regra:** escolhe-se o ε que satisfaz os dois critérios; se os dois satisfazem, o **menor**. Se nenhum satisfaz os dois, o menor ε que degrada o FedAvg, **declarando que o resultado pode ser trivial** (o fixed excluiria os atacantes). Se nenhum degrada o FedAvg, caso não previsto: **parar e consultar o autor** antes da grade.
- **O ε é escolhido olhando só o FedAvg e o fixed; o td3 nunca entra no sanity.**
- Aplicação mecânica: `scripts/passo2_oficial/analisar_b25.py sanity` → `sanity.txt`. O ε escolhido é registrado em **Adendo 1** (hasheado e commitado) antes de lançar a grade.

## 4. Grade

- MNIST, q = 0,5, 500 rodadas, 100 clientes, 10% por rodada, 20 atacantes (iguais ao B2.1).
- Ataque: IPM real com o ε do Adendo 1. Condições: `td3` (como no `main.py` oficial) e `fixed` = [0,475]×5.
- **Sementes: 105–114** (confirmação no código oficial; célula nova em relação ao B2.1).
- Total: **20 runs**, 5 em paralelo na GPU, ~27–31 h de relógio.
- Registro igual ao B2.1: estados antes de cada ação, checkpoints do ator a cada 50 passos, massa de peso nos atacantes, resets. Truncagem em 500 passos (td3 do SB3 roda 501).

## 5. Hipótese e critérios (iguais ao H1 do B2.1)

**Métrica primária:** mediana da acurácia no teste nas rodadas 401–500 (a do B2.1, mantida para comparabilidade).
**Unidade:** semente, n = 10. D = métrica(fixed) − métrica(td3).

**H1.** A ação fixa é equivalente ao TD3 sob IPM real.
- TOST pareado (t), margem ±1,0 p.p., α = 0,05 → **equivalência confirmada**. Reporta-se IC90 (o intervalo do TOST) e IC95.
- Wilcoxon pareado p < 0,05 com D < 0 → **o TD3 contribui**.
- Wilcoxon p < 0,05 com D > 0 → a fixa é melhor; reportar como "não contribui".
- Nenhum → **inconclusivo**; reportar Δ, IC90, IC95 e d.

**Secundárias (descritivas):** média 451–500; nº de resets; massa nos atacantes por condição; drift |π₅₀₀ − π₀| e `sd_estados` de π₅₀₀ sobre os estados das rodadas 401–500 (comparados a σ_a = 0,0475, sem teste).

**Poder:** com n = 10, a equivalência só é atingível se o sd das diferenças for pequeno (no B2.1 foi ~0,3 p.p. na métrica primária). Um resultado inconclusivo é reportado como tal.

## 6. Leitura para a tese

- Equivalência → a conclusão do P2 ("o TD3 não contribui") se estende ao terceiro ataque do paper original, com IPM corrigido.
- TD3 contribui → limita a conclusão do P2 a LMP/EB; reportar com destaque.
- Se o ε escolhido não satisfizer o critério de inclusão, a leitura é condicionada à declaração de possível trivialidade (§3).

## 7. Regras

- Análise pré-escrita em `scripts/passo2_oficial/analisar_b25.py` (subcomandos `sanity` e `grade`), rodada uma vez com os runs completos.
- Runs que falharem por erro de ambiente são repetidos com a mesma semente e registrados. Colapsos e resets contam como resultado.
- Nenhuma acurácia da grade é inspecionada antes do fim. O monitoramento reporta só saúde e progresso.
- Logs (`*.runner.log`, `*.stdout.log`) e checkpoints intermediários do ator não são versionados (só os passos 0 e 500).

## 8. Fora do escopo

**B2.7b cortado** (roadmap v4): declarado como limitação. B2.4r e MedMNIST só quando o autor pedir.
