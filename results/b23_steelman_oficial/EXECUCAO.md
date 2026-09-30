# Registro de execução — B2.3

Mudanças de **agendamento** (não de desenho) em relação ao `PLANO.md`. O plano, os 10 runs, o runner (`run_b23.py`), a configuração do steelman e a análise continuam os mesmos e com os hashes de `PLANO.sha256`.

## 2026-09-29 21:01 — antecipação de 2 runs

- **Motivo:** o B2.1 entrou na última semente com só 4 processos, deixando 2 vagas de GPU ociosas por ~8 h. Aproveitá-las encurta o B2.3.
- **O que mudou:**
  - o disparador `run_grid_b23.sh --after-b21` (PID 1185960, armado em 2026-09-28 08:19) foi encerrado antes de iniciar qualquer run;
  - novo escalonador `scripts/passo2_oficial/run_b23_escalonado.sh` (arquivo novo; `run_grid_b23.sh` não foi alterado): roda **LMP e EB da semente 100 imediatamente** e, após o GRID_END do B2.1, os **8 restantes com 4 em paralelo**.
- **Por que não afeta o resultado:** a ordem de execução não entra no desenho; cada run é determinado pelo ambiente oficial e pela semente. O paralelismo muda só o tempo de relógio.
- **Efeito colateral declarado:** durante a sobreposição, os 4 últimos runs do B2.1 ficam um pouco mais lentos (GPU compartilhada por 6 processos, como no resto da grade).
- **Nenhum resultado foi olhado** antes desta mudança (B2.1: só saúde e progresso; B2.3: nenhum run iniciado até então).
