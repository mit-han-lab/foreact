import argparse
import gradio as gr
import numpy as np
import random
import torch

from pipeline import VisualForesightPipeline
from utils.trainer_utils import find_newest_checkpoint

MIN_SEED = 0
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGES_PER_PROMPT = 4
DEFAULT_IMAGES_PER_PROMPT = 1


def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def make_process_fn(pipeline, view_names):
    n_meta = 7  # prompt, negative_prompt, seed, gs, igs, n_steps, n_images

    def process(*all_args, progress=gr.Progress(track_tqdm=True)):
        (
            prompt,
            negative_prompt,
            seed,
            guidance_scale,
            image_guidance_scale,
            num_inference_steps,
            num_images_per_prompt,
        ) = all_args[:n_meta]
        view_images = all_args[n_meta:]

        input_images = {
            name: img
            for name, img in zip(view_names, view_images)
            if img is not None
        }

        per_view_outputs = pipeline(
            images=input_images if input_images else None,
            caption=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=torch.Generator().manual_seed(seed),
            enable_progress_bar=True,
        ).images

        # Gradio convention: with a single output component, return the raw
        # value (returning a 1-tuple would be passed through to the Gallery
        # as-is, breaking its postprocess). With multiple components, return
        # a tuple so Gradio dispatches per-component.
        view_outputs = [per_view_outputs.get(name, []) for name in view_names]
        if len(view_outputs) == 1:
            return view_outputs[0]
        return tuple(view_outputs)

    return process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint"
    )
    args = parser.parse_args()

    checkpoint_path = find_newest_checkpoint(args.checkpoint_path)
    pipeline = VisualForesightPipeline.from_pretrained(
        checkpoint_path,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )

    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)

    # Drive the UI from the loaded checkpoint's view configuration. Old
    # single-view checkpoints (no `view_names` field) fall back to the config
    # default ("primary",) and render a 1-panel UI; multi-view checkpoints
    # render one input/output per configured view.
    view_names = list(pipeline.config.view_names)
    print(f"Loaded checkpoint with {len(view_names)} view(s): {view_names}")

    css = """
    .gallery-view .grid-wrap {
        max-height: 320px !important;
        overflow: hidden !important;
    }
    .gallery-view .thumbnail-item {
        max-height: 300px !important;
    }
    .gallery-view .thumbnail-item img {
        max-height: 290px !important;
        object-fit: contain !important;
    }
    .gr-block { border-radius: 12px !important; }
    """

    with gr.Blocks(fill_width=True, css=css, title="VisualForesight") as demo:
        gr.Markdown(f"# VisualForesight ({len(view_names)}-view)", elem_id="title")

        # Prompt row — full width
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                max_lines=1,
                placeholder="Describe what you want to change...",
                scale=4,
            )
            generate_btn = gr.Button("Generate", variant="primary", scale=1, min_width=120)

        input_components = []
        with gr.Row(equal_height=True):
            for name in view_names:
                input_components.append(
                    gr.Image(label=f"{name} Input", type="pil", height=320)
                )

        output_components = []
        with gr.Row(equal_height=True):
            for name in view_names:
                output_components.append(
                    gr.Gallery(
                        columns=1, label=f"{name} Output",
                        elem_classes=["gallery-view"],
                        height=320, object_fit="contain", preview=False,
                    )
                )

        # Settings row
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                seed = gr.Slider(
                    label="Seed", minimum=MIN_SEED, maximum=MAX_SEED, step=1, value=0
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                guidance_scale = gr.Slider(
                    1, 30, step=0.5, value=4.5, label="Guidance Scale"
                )
                image_guidance_scale = gr.Slider(
                    1, 30, step=0.5, value=1.5, label="Image Guidance Scale"
                )
            with gr.Row():
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    max_lines=1,
                    value="",
                    scale=3,
                )
                num_inference_steps = gr.Slider(
                    1, 100, step=1, value=8, label="Inference Steps", scale=1,
                )
                num_images_per_prompt = gr.Slider(
                    1,
                    MAX_IMAGES_PER_PROMPT,
                    value=DEFAULT_IMAGES_PER_PROMPT,
                    step=1,
                    label="Number of Images",
                    scale=1,
                )

        inputs = [
            prompt,
            negative_prompt,
            seed,
            guidance_scale,
            image_guidance_scale,
            num_inference_steps,
            num_images_per_prompt,
            *input_components,
        ]
        outputs = output_components

        process_fn = make_process_fn(pipeline, view_names)

        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            api_name=False,
        ).then(
            fn=process_fn,
            inputs=inputs,
            outputs=outputs,
        )

        generate_btn.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            api_name=False,
        ).then(
            fn=process_fn,
            inputs=inputs,
            outputs=outputs,
        )

        demo.launch(share=True)
