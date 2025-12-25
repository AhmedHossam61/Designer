from diffusers import DiffusionPipeline
import torch
from pathlib import Path
from datetime import datetime
import json


class ImageGenerationAgent:
    """Optimized AI agent for generating images using Stable Diffusion XL"""

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        output_dir="generated_images"
    ):
        print("🤖 Initializing AI Image Generation Agent...")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # ================= GPU DIAGNOSTICS =================
        print("\n" + "=" * 60)
        print("🔍 GPU DIAGNOSTICS")
        print("=" * 60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )
        else:
            self.device = "cpu"
            print("⚠️ CUDA not available, using CPU")

        print(f"📱 Selected device: {self.device}")
        print("=" * 60 + "\n")

        # ================= LOAD MODEL =================
        print("⏳ Loading diffusion model...")
        if self.device == "cuda":
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                use_safetensors=True,
            )

        self.pipe.to(self.device)

        # ================= OPTIMIZATIONS =================
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_tiling()

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✅ xFormers enabled")
            except Exception:
                print("⚠️ xFormers not available")

            print(
                f"GPU memory after loading: "
                f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            )

        # ================= PROMPT TEMPLATES =================
        self.quality_prompt = (
            "high quality, ultra detailed, sharp focus, professional lighting, "
            "cinematic composition, 8k, realistic"
        )

        self.default_negative = (
            "low quality, blurry, noisy, deformed, bad anatomy, "
            "extra fingers, extra limbs, watermark, text, logo"
        )

        print("✅ Agent ready!\n")

    def generate_image(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        seed=None,
    ):
        print(f"🎨 Generating image for: '{prompt}'")

        full_prompt = f"{self.quality_prompt}, {prompt}"
        final_negative = (
            f"{self.default_negative}, {negative_prompt}"
            if negative_prompt
            else self.default_negative
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            if self.device == "cuda":
                print(
                    f"GPU before gen: "
                    f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                )

            with torch.inference_mode():
                result = self.pipe(
                    prompt=full_prompt,
                    negative_prompt=final_negative,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )

            image = result.images[0]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(
                c for c in prompt[:50] if c.isalnum() or c in (" ", "_")
            ).replace(" ", "_")
            filepath = self.output_dir / f"{timestamp}_{safe_prompt}.png"

            image.save(filepath)

            # ================= SAVE METADATA =================
            metadata = {
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": final_negative,
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "model": self.pipe.config._name_or_path,
            }

            with open(filepath.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Image saved: {filepath}")

            if self.device == "cuda":
                print(
                    f"GPU after gen: "
                    f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                )
                torch.cuda.empty_cache()

            del result
            return filepath

        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None

    def interactive_mode(self):
        print("=" * 60)
        print("🎨 AI IMAGE GENERATION AGENT - Interactive Mode")
        print("=" * 60)

        num_steps = 30
        guidance = 7.5
        width, height = 1024, 1024
        seed = None

        while True:
            try:
                prompt = input("\n🖼️ Prompt (or 'exit'): ").strip()
                if prompt.lower() in ("exit", "quit", "q"):
                    break

                negative = input("🚫 Negative (optional): ").strip()
                seed_input = input("🎲 Seed (Enter for random): ").strip()
                seed = int(seed_input) if seed_input else None

                self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    seed=seed,
                )

            except KeyboardInterrupt:
                break


def main():
    agent = ImageGenerationAgent()
    agent.interactive_mode()


if __name__ == "__main__":
    main()
