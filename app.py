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


def process(
    prompt,
    negative_prompt,
    seed,
    guidance_scale,
    image_guidance_scale,
    num_inference_steps,
    num_images_per_prompt,
    input_image,
    progress=gr.Progress(track_tqdm=True),
):
    images = pipeline(
        image=input_image,
        caption=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        generator=torch.Generator().manual_seed(seed),
        enable_progress_bar=True,
    ).images
    return images


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

    css = """
    #output-gallery .grid-wrap {
        max-height: 480px !important;
        overflow: hidden !important;
    }
    #output-gallery .thumbnail-item {
        max-height: 460px !important;
    }
    #output-gallery .thumbnail-item img {
        max-height: 450px !important;
        object-fit: contain !important;
    }
    .gr-block { border-radius: 12px !important; }
    """

    with gr.Blocks(fill_width=True, css=css, title="VisualForesight") as demo:
        gr.Markdown("# VisualForesight", elem_id="title")

        # Prompt row — full width
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                max_lines=1,
                placeholder="Describe what you want to change...",
                scale=4,
            )
            generate_btn = gr.Button("Generate", variant="primary", scale=1, min_width=120)

        # Input / Output side by side — equal columns
        with gr.Row(equal_height=True):
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil", height=480)
            with gr.Column():
                output_gallery = gr.Gallery(
                    columns=1,
                    label="Generated Images",
                    elem_id="output-gallery",
                    height=480,
                    object_fit="contain",
                    preview=False,
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
            input_image,
        ]

        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=output_gallery,
        )

        generate_btn.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=output_gallery,
        )

        demo.launch(share=True)
