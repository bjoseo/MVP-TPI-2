# ==============================================================================
# Encapsula el modelo de vision openai/clip-vit-base-patch32 para evaluar las imagenes en el espacio semantico compartido y generar la tabla comparativa entre la linea base clasica y la reconstruccion por difusion
# ==============================================================================


import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CLIP_ID = "openai/clip-vit-base-patch32"

print(f"[analysis.py] Cargando {MODEL_CLIP_ID} en {DEVICE}...")

clip_model = CLIPModel.from_pretrained(MODEL_CLIP_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_CLIP_ID)

# Frases evaluadas en inglés para aprovechar la máxima precisión del entrenamiento original de CLIP
ETIQUETAS_CLIP = {
    "sharp photograph with clear and defined details": "Imagen nítida con detalles definidos",
    "blurry, pixelated, low resolution photograph": "Imagen borrosa o pixelada",
    "clear, legible printed or handwritten text": "Texto o letras legibles",
    "human face with clear facial features": "Rostro humano con rasgos definidos",
    "artificial, synthetic or heavily reconstructed texture": "Superficie con textura artificial o reconstruida",
}


def evaluar_imagen_clip(imagen: Image.Image) -> dict[str, float]:
    """
    Calcula probabilidades relativas de similitud semántica (Zero-Shot) entre la imagen
    y las descripciones textuales definidas.
    """
    prompts_en = list(ETIQUETAS_CLIP.keys())
    inputs = clip_processor(
        text=prompts_en, images=imagen, return_tensors="pt", padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    return {
        ETIQUETAS_CLIP[prompt_en]: float(prob)
        for prompt_en, prob in zip(prompts_en, probs)
    }


def comparar_analisis_visual(img_clasica: Image.Image, img_ia: Image.Image) -> list[dict]:
    """
    Compara los puntajes semánticos antes (interpolación clásica) y después (difusión).
    """
    scores_antes = evaluar_imagen_clip(img_clasica)
    scores_despues = evaluar_imagen_clip(img_ia)

    comparacion = []
    for etiqueta_es in ETIQUETAS_CLIP.values():
        v_antes = scores_antes[etiqueta_es] * 100
        v_despues = scores_despues[etiqueta_es] * 100
        comparacion.append({
            "criterio": etiqueta_es,
            "antes": v_antes,
            "despues": v_despues,
            "delta": v_despues - v_antes,
        })
    return comparacion