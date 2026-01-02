# from __future__ import annotations

# from typing import Optional

# from pydantic import BaseModel, Field
# from langchain.tools import tool

# # Local tool implementation (already provided by you)
# from designer_opt import ImageGenerationAgent


# # IMPORTANT:
# # - Instantiate the SDXL pipeline ONCE (module-level singleton) to avoid re-loading per image.
# # - Keep output_dir configurable via the tool args if you want, but default is fine.
# _AGENT = ImageGenerationAgent(output_dir="generated_images")


# class GenerateImageArgs(BaseModel):
#     prompt: str = Field(..., description="Diffusion-ready prompt for the image (no text/logos/watermarks).")
#     width: int = Field(1024, ge=256, le=2048)
#     height: int = Field(1024, ge=256, le=2048)
#     steps: int = Field(30, ge=10, le=80)
#     guidance: float = Field(7.5, ge=1.0, le=20.0)
#     seed: Optional[int] = Field(None, description="Optional seed for reproducibility.")


# @tool("designer_opt_generate_image", args_schema=GenerateImageArgs)
# def designer_opt_generate_image(
#     prompt: str,
#     width: int = 1024,
#     height: int = 1024,
#     steps: int = 30,
#     guidance: float = 7.5,
#     seed: Optional[int] = None,
# ) -> str:
#     """
#     Generate an image using designer_opt (SDXL) and return the saved file path as a string.
#     """
#     path = _AGENT.generate_image(
#         prompt=prompt,
#         width=width,
#         height=height,
#         num_inference_steps=steps,
#         guidance_scale=guidance,
#         seed=seed,
#     )
#     return str(path)


from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from langchain.tools import tool

# Local tool implementation (already provided by you)
from designer_opt import ImageGenerationAgent


# IMPORTANT:
# - Instantiate the SDXL pipeline ONCE (module-level singleton) to avoid re-loading per image.
# - Keep output_dir configurable via the tool args if you want, but default is fine.
_AGENT = ImageGenerationAgent(output_dir="generated_images")


class GenerateImageArgs(BaseModel):
    prompt: str = Field(..., description="Diffusion-ready prompt for the image (no text/logos/watermarks).")
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(30, ge=10, le=80)
    guidance: float = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None, description="Optional seed for reproducibility.")


@tool("designer_opt_generate_image", args_schema=GenerateImageArgs)
def designer_opt_generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    guidance: float = 7.5,
    seed: Optional[int] = None,
) -> str:
    """
    Generate an image using designer_opt (SDXL) and return the saved file path as a string.
    """
    path = _AGENT.generate_image(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
    )
    return str(path)
