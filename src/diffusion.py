# ==============================================================================
# Se encarga exclusivamente de preparar el tensor de entrada, cargar stabilityai/sd-x2-latent-upscaler una sola vez y ejecutar la inferencia en GPU/CPU.
# ==============================================================================


import torch
from PIL import Image
from diffusers import StableDiffusionLatentUpscalePipeline

# Configuración global de hardware (se ejecuta una sola vez al importar el módulo)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_DIFFUSION_ID = "stabilityai/sd-x2-latent-upscaler"

print(f"[diffusion.py] Cargando {MODEL_DIFFUSION_ID} en {DEVICE} ({DTYPE})...")

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
    MODEL_DIFFUSION_ID,
    dtype=DTYPE,
)
upscaler.to(DEVICE)

if DEVICE == "cuda":
    upscaler.enable_attention_slicing()


def preparar_imagen_entrada(imagen: Image.Image, tamano_objetivo: int = 256) -> Image.Image:
    """
    Escala la imagen manteniendo su relación de aspecto hasta tamano_objetivo (256px)
    y aplica relleno neutro (zero-padding) para obtener un cuadrado exacto de 256x256.
    Esto garantiza divisibilidad por 128 (VAE x8 * U-Net x16) sin distorsionar objetos.
    """
    imagen = imagen.convert("RGB")
    ancho, alto = imagen.size

    escala = tamano_objetivo / max(ancho, alto)
    nuevo_ancho = max(1, int(round(ancho * escala)))
    nuevo_alto = max(1, int(round(alto * escala)))
    img_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    lienzo = Image.new("RGB", (tamano_objetivo, tamano_objetivo), (128, 128, 128))
    offset_x = (tamano_objetivo - nuevo_ancho) // 2
    offset_y = (tamano_objetivo - nuevo_alto) // 2
    lienzo.paste(img_redimensionada, (offset_x, offset_y))

    return lienzo


def ejecutar_difusion(
    img_base: Image.Image,
    prompt_guia: str,
    pasos_inferencia: int,
    semilla: int,
    escala_guia: float,
) -> Image.Image:
    """
    Aplica super-resolución x2 operando en el espacio latente mediante denoising iterativo.
    """
    generador = torch.Generator(device=DEVICE).manual_seed(int(semilla))
    prompt_efectivo = (
        prompt_guia.strip()
        if prompt_guia and prompt_guia.strip()
        else "archival photograph, sharp details"
    )

    resultado = upscaler(
        prompt=prompt_efectivo,
        image=img_base,
        num_inference_steps=int(pasos_inferencia),
        guidance_scale=float(escala_guia),
        generator=generador,
    ).images[0]

    return resultado