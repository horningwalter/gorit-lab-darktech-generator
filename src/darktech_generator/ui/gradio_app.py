"""Gradio UI for the MVP. Single textbox + generate button + audio player.

Phase 2 adds the reference dropzone, plan inspector, stem solos, and feedback chat.
"""

from __future__ import annotations

import logging

import gradio as gr

from darktech_generator.api import generate_track
from darktech_generator.config import get_settings

logger = logging.getLogger(__name__)


def _backend_status() -> str:
    settings = get_settings()
    if not settings.darktech_remote_url:
        return "Backend: LOCAL GPU (set DARKTECH_REMOTE_URL in .env to offload to Colab)."
    try:
        from darktech_generator.generation.remote_renderer import RemoteRenderer

        info = RemoteRenderer().health()
        gpu = info.get("gpu", {})
        name = gpu.get("device_name", "unknown")
        vram = gpu.get("vram_free_gb")
        suffix = f", {vram} GB free" if vram is not None else ""
        return f"Backend: REMOTE {settings.darktech_remote_url} (GPU: {name}{suffix})"
    except Exception as e:
        return f"Backend: REMOTE {settings.darktech_remote_url} (health check failed: {e})"


def _run(user_prompt: str) -> tuple[str, str, str]:
    if not user_prompt.strip():
        return "", "", "Please describe the track you want."
    result = generate_track(user_prompt)
    plan = result.track_plan
    plan_summary = (
        f"BPM {plan.bpm:.1f} | Key {plan.key} | Duration {plan.duration_seconds:.0f}s | "
        f"Sections: {', '.join(s.name for s in plan.sections)}\n"
        f"Cost: ${result.total_usd:.4f}\n"
        f"Notes: {plan.notes}"
    )
    return result.output_wav_path, plan_summary, "Done."


def build() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="Gorit Lab DarkTech Generator") as demo:
        gr.Markdown(
            "# Gorit Lab DarkTech Generator\n"
            "MVP. Describe a DarkTech track and press Generate. "
            f"Cost cap per session: \\${settings.darktech_cost_hard_cap_usd:.2f}."
        )
        backend = gr.Textbox(
            label="",
            value=_backend_status(),
            interactive=False,
            show_label=False,
            container=False,
        )
        refresh_btn = gr.Button("Refresh backend status", size="sm")
        with gr.Row():
            prompt = gr.Textbox(
                label="Brief",
                lines=4,
                placeholder=(
                    "ex: DarkTech 187 BPM, intro atmosférico longo, drop pesado "
                    "estilo Kindzadza, synth ácido modular."
                ),
            )
        with gr.Row():
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            audio_out = gr.Audio(label="Output", type="filepath")
        with gr.Row():
            plan_box = gr.Textbox(label="Track plan", lines=6, interactive=False)
            status = gr.Textbox(label="Status", lines=2, interactive=False)

        run_btn.click(
            fn=_run,
            inputs=[prompt],
            outputs=[audio_out, plan_box, status],
        )
        refresh_btn.click(fn=_backend_status, outputs=[backend])
    return demo


def launch(share: bool | None = None) -> None:
    settings = get_settings()
    settings.ensure_dirs()
    app = build()
    app.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share if share is None else share,
    )


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
