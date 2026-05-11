# Gorit Lab DarkTech Generator

Gerador de faixas musicais no estilo **DarkTech** (Darkpsy + Hitech, 180-200 BPM) assistido por IA, orquestrado por **DeepSeek V4 Pro** e com geração de áudio em **ACE-Step 1.5**.

Reboot 2026 do projeto original (síntese DSP em Python puro), agora baseado em modelos generativos abertos de áudio.

## Status

Pré-alpha. Fase 1 (MVP) implementada.

## Modos de execução

O projeto roda de três formas. Escolha pelo seu hardware.

### 1. Híbrido: UI local + GPU remota no Colab (recomendado para RTX 3050 / hardware modesto)

Sua máquina hospeda a interface, a orquestração com DeepSeek e o mastering (tudo CPU-bound ou via API). A geração de áudio é despachada para um notebook Colab com A100 via Cloudflare Tunnel. **Você usa a UI local, mas a A100 do Colab faz o trabalho pesado.**

**No Colab** (uma vez por sessão):
1. Abra [notebooks/99_colab_worker.ipynb](notebooks/99_colab_worker.ipynb).
2. Runtime > Change runtime type > GPU > A100.
3. Rode as 3 células. A última imprime uma URL `https://*.trycloudflare.com` e a API key da sessão. Deixe essa célula rodando.

**Localmente** (no VSCode):
```bash
cd /home/horningwalter/Projetos/gorit_lab_darktech_generator
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
# editar .env:
#   DEEPSEEK_API_KEY=sk-...
#   DARKTECH_REMOTE_URL=https://abc-123.trycloudflare.com
#   DARKTECH_REMOTE_API_KEY=<a chave que o notebook imprimiu>
darktech-gradio
```

Abra `http://localhost:7860`. O cabeçalho da UI mostrará `Backend: REMOTE ...` confirmando a conexão.

Quando terminar de produzir, feche o notebook do Colab para liberar a A100. A UI local continua aberta; basta religar o Colab depois para retomar.

### 2. Totalmente no Colab (sem máquina local)

Use o notebook [notebooks/00_colab_launcher.ipynb](notebooks/00_colab_launcher.ipynb): instala o pacote, sobe a UI Gradio dentro do próprio Colab e expõe um link público. Útil quando você quer trabalhar de outra máquina.

### 3. Totalmente local (precisa de GPU 8GB+ VRAM)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[ml]"
cp .env.example .env  # apenas DEEPSEEK_API_KEY, deixar DARKTECH_REMOTE_URL vazio
darktech-gradio
```

## Arquitetura

Pipeline neural orquestrado por agente, em quatro camadas com dependência estritamente descendente:

```
[Gradio UI (local)]
    │
    ▼
[Orchestrator Agent (Pydantic AI + DeepSeek V4 Pro)]
    │
    ├──→ [Reference Analyzer]        Essentia + MERT + CLAP    (Fase 2)
    ├──→ [Plan/Arrangement Node]     TrackPlan (Pydantic)
    ├──→ [Stem Generation]           ACE-Step 1.5
    │       ├── local (sua GPU)
    │       └── remoto (Colab A100 via Cloudflare Tunnel)
    ├──→ [Stem Aligner + Looper]     (Fase 2)
    ├──→ [Mix Bus + Mastering]       pedalboard + pyloudnorm   (local)
    └──→ [Critic/Refinement Loop]    (Fase 3)
```

Quem decide se o renderer é local ou remoto é [model_registry.py](src/darktech_generator/generation/model_registry.py), olhando para `DARKTECH_REMOTE_URL`. Nenhuma outra camada precisa saber.

## Custo estimado

- Hobby (sem Colab pago): $0-1/mês com Colab Free T4 + DeepSeek pay-per-use (~$0.01/faixa).
- Produção (recomendado): ~$12/mês com Colab Pro A100 + Drive 100GB + DeepSeek (~$2.50/mês para 100 faixas).
- Treino de LoRA: ~$50 num mês isolado (Colab Pro+).

## Testes

```bash
uv pip install -e ".[dev]"
PYTHONPATH=src pytest tests/unit/ -v
```

11 testes unitários offline (schemas, mastering, exporter).

## Licença

MIT.
