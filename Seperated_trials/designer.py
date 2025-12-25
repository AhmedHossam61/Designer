from diffusers import DiffusionPipeline
import torch
from pathlib import Path
from datetime import datetime
import os

class ImageGenerationAgent:
    """Simple AI agent for generating images using Stable Diffusion XL"""
    
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0", output_dir="generated_images"):
        """
        Initialize the image generation agent
        
        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save generated images
        """
        print("🤖 Initializing AI Image Generation Agent...")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Detailed CUDA diagnostics
        print("\n" + "="*60)
        print("🔍 GPU DIAGNOSTICS")
        print("="*60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            self.device = "cuda"
        else:
            print("⚠️  WARNING: CUDA not available!")
            print("   Possible issues:")
            print("   1. PyTorch not installed with CUDA support")
            print("   2. NVIDIA drivers not installed")
            print("   3. No compatible GPU detected")
            self.device = "cpu"
        
        print(f"📱 Selected device: {self.device}")
        print("="*60 + "\n")
        
        # Load the diffusion pipeline
        print("⏳ Loading model (this may take a moment)...")
        if self.device == "cuda":
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                variant="fp16"
            )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, 
                use_safetensors=True
            )
        
        self.pipe.to(self.device)
        
        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_vae_tiling()

        # Verify model is on GPU
        if self.device == "cuda":
            print(f"✅ Model loaded on GPU")
            print(f"   GPU memory after loading: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        print("✅ Agent ready! Let's create some images!\n")
    
    def generate_image(self, prompt, negative_prompt="", num_inference_steps=30, guidance_scale=7.5):
        """
        Generate an image based on the user prompt
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Things to avoid in the image
            num_inference_steps: Number of denoising steps (higher = better quality but slower)
            guidance_scale: How closely to follow the prompt (7-9 is typical)
        
        Returns:
            Path to the saved image
        """
        print(f"🎨 Generating image for: '{prompt}'")
        print(f"⚙️  Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        
        try:
            # Monitor GPU usage if available
            if self.device == "cuda":
                print(f"   GPU memory before generation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # Generate the image
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            if self.device == "cuda":
                print(f"   GPU memory after generation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            image = result.images[0]
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
            filename = f"{timestamp}_{safe_prompt}.png"
            filepath = self.output_dir / filename
            
            # Save the image
            image.save(filepath)
            print(f"✅ Image saved to: {filepath}\n")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error generating image: {e}\n")
            return None
    
    def interactive_mode(self):
        """Run the agent in interactive mode, continuously accepting user prompts"""
        print("=" * 60)
        print("🎨 AI IMAGE GENERATION AGENT - Interactive Mode")
        print("=" * 60)
        print("Type your image descriptions and I'll generate them!")
        print("Commands:")
        print("  - Type a description to generate an image")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'settings' to adjust generation parameters")
        print("=" * 60)
        print()
        
        # Default settings
        num_steps = 34
        guidance = 4
        
        while True:
            try:
                user_input = input("🖼️  Describe your image (or command): ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using the AI Image Generation Agent!")
                    break
                
                if user_input.lower() == 'settings':
                    print(f"\n⚙️  Current settings:")
                    print(f"   Steps: {num_steps}")
                    print(f"   Guidance scale: {guidance}")
                    
                    try:
                        new_steps = input(f"   New steps (press Enter to keep {num_steps}): ").strip()
                        if new_steps:
                            num_steps = int(new_steps)
                        
                        new_guidance = input(f"   New guidance scale (press Enter to keep {guidance}): ").strip()
                        if new_guidance:
                            guidance = float(new_guidance)
                        
                        print(f"✅ Settings updated!\n")
                    except ValueError:
                        print("❌ Invalid input, keeping previous settings\n")
                    
                    continue
                
                # Check if user wants to add negative prompt
                negative = input("🚫 Anything to avoid? (press Enter to skip): ").strip()
                
                # Generate the image
                self.generate_image(
                    prompt=user_input,
                    negative_prompt=negative,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance
                )
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

def main():
    """Main function to run the agent"""
    # Initialize the agent
    agent = ImageGenerationAgent()
    
    # Run in interactive mode
    agent.interactive_mode()

if __name__ == "__main__":
    main()