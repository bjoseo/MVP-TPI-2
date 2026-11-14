import gradio as gr
import torch
from PIL import Image, ImageChops, ImageEnhance
from diffusers import StableDiffusionLatentUpscalePipeline
from transformers import CLIPModel, CLIPProcessor

# ==============================================================================
# 1. CONFIGURACIÓN DE HARDWARE Y CARGA DE MODELOS (100% GRATUITOS)
# ==============================================================================
# Detectamos si hay GPU disponible (en Google Colab T4 será "cuda").
# Usamos float16 en GPU para reducir el consumo de VRAM a la mitad y acelerar la inferencia.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL_DIFFUSION_ID = "stabilityai/sd-x2-latent-upscaler"
MODEL_CLIP_ID = "openai/clip-vit-base-patch32"

print(f"Cargando modelos en {DEVICE} ({DTYPE})...")

# Modelo de Difusión: Opera en el espacio latente para duplicar (x2) la resolución.
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
    MODEL_DIFFUSION_ID,
    dtype=DTYPE,
)
upscaler.to(DEVICE)

# Optimización de memoria para evitar desbordes de VRAM en GPUs gratuitas (T4)
if DEVICE == "cuda":
    upscaler.enable_attention_slicing()

# Modelo de Análisis Visual (CLIP): Proyecta imágenes y texto en un espacio semántico común.
clip_model = CLIPModel.from_pretrained(MODEL_CLIP_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_CLIP_ID)

# Etiquetas para evaluar cómo cambia la percepción semántica antes y después
ETIQUETAS_CLIP = [
    "imagen nítida con detalles definidos y claros",
    "imagen borrosa, pixelada y de baja resolución",
    "texto o letras legibles",
    "rostro humano con rasgos definidos",
    "superficie con textura reconstruida o artificial",
]


# ==============================================================================
# 2. FUNCIONES DE PROCESAMIENTO DIGITAL Y ANÁLISIS
# ==============================================================================
def preparar_imagen_entrada_old_01(imagen: Image.Image, max_lado: int = 256) -> Image.Image:
    """
    Limita el tamaño de entrada a 256px en su lado mayor y asegura dimensiones
    múltiplos de 8 (requisito matemático de los autoencoders del espacio latente).
    Esto evita demoras excesivas y errores de memoria en Colab.
    """
    imagen = imagen.convert("RGB")
    ancho, alto = imagen.size

    if max(ancho, alto) > max_lado:
        escala = max_lado / max(ancho, alto)
        ancho = int(ancho * escala)
        alto = int(alto * escala)

    # Ajuste a múltiplo de 8 para compatibilidad con el espacio latente del VAE
    ancho = max(64, (ancho // 8) * 8)
    alto = max(64, (alto // 8) * 8)
    return imagen.resize((ancho, alto), Image.Resampling.LANCZOS)

# ==============================================================================
# 2. FUNCIONES DE PROCESAMIENTO DIGITAL Y ANÁLISIS
# ==============================================================================
#    Se recomienda usar preparar_imagen_entrada (con múltiplos de 16) para evitar errores de #  skip-connections en la U-Net.
#    Se deja preparar_imagen_entrada_old como referencia histórica de la versión inicial.
#    Se cambiá el divisor 8 por 16 en preparar_imagen_entrada para asegurar compatibilidad con los 4 niveles de downsampling de la U-Net.
# ==============================================================================

# 2. En preparar_imagen_entrada (línea ~49), cambiá el divisor 8 por 16:
def preparar_imagen_entrada_old_02(imagen: Image.Image, max_lado: int = 256) -> Image.Image:
    """
    Limita el tamaño de entrada a 256px en su lado mayor y asegura dimensiones
    múltiplos de 16 (requisito de los 4 niveles de downsampling de la U-Net).
    """
    imagen = imagen.convert("RGB")
    ancho, alto = imagen.size

    if max(ancho, alto) > max_lado:
        escala = max_lado / max(ancho, alto)
        ancho = int(ancho * escala)
        alto = int(alto * escala)

    # Ajuste a múltiplo de 16 para que las skip-connections de la U-Net coincidan
    ancho = max(64, (ancho // 16) * 16)
    alto = max(64, (alto // 16) * 16)
    return imagen.resize((ancho, alto), Image.Resampling.LANCZOS)

# ==============================================================================
# 2. FUNCIONES DE PROCESAMIENTO DIGITAL Y ANÁLISIS
# ==============================================================================
def preparar_imagen_entrada(imagen: Image.Image, tamano_objetivo: int = 256) -> Image.Image:
    """
    Escala la imagen manteniendo su relación de aspecto para que el lado mayor
    mida 256px y completa con relleno neutro (padding) hasta 256x256.
    Esto garantiza divisibilidad exacta por 128 (VAE x8 * U-Net x16) sin deformar
    ni recortar información de la escena.
    """
    imagen = imagen.convert("RGB")
    ancho, alto = imagen.size

    # Escalamos proporcionalmente para que el lado mayor sea tamano_objetivo (256)
    escala = tamano_objetivo / max(ancho, alto)
    nuevo_ancho = max(1, int(round(ancho * escala)))
    nuevo_alto = max(1, int(round(alto * escala)))
    img_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    # Creamos un lienzo cuadrado exacto (256x256) y centramos la imagen
    lienzo = Image.new("RGB", (tamano_objetivo, tamano_objetivo), (128, 128, 128))
    offset_x = (tamano_objetivo - nuevo_ancho) // 2
    offset_y = (tamano_objetivo - nuevo_alto) // 2
    lienzo.paste(img_redimensionada, (offset_x, offset_y))

    return lienzo


def analizar_con_clip(imagen: Image.Image) -> dict[str, float]:
    """
    Evalúa la imagen usando CLIP (Zero-Shot).
    IMPORTANTE: No mide 'veracidad histórica', sino qué tan plausible resulta
    la imagen frente a cada descripción textual en el espacio latente compartido.
    """
    inputs = clip_processor(
        text=ETIQUETAS_CLIP, images=imagen, return_tensors="pt", padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        # Softmax convierte los logits de similitud coseno en probabilidades relativas
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    return {etiqueta: float(prob) for etiqueta, prob in zip(ETIQUETAS_CLIP, probs)}


def generar_mapa_diferencias(img_clasica: Image.Image, img_ia: Image.Image) -> Image.Image:
    """
    Calcula la diferencia absoluta píxel a píxel entre la interpolación clásica
    (dato estimado matemáticamente sin inventar estructuras) y la difusión.
    Las zonas brillantes muestran dónde la IA agregó o modificó más información.
    """
    diff = ImageChops.difference(img_clasica, img_ia)
    # Realzamos el contraste x3 para que los cambios sutiles sean visibles al ojo humano
    return ImageEnhance.Contrast(diff).enhance(3.0)


def ejecutar_pipeline(
    imagen_entrada: Image.Image,
    prompt_guia: str,
    pasos_inferencia: int,
    semilla: int,
    escala_guia: float,
):
    if imagen_entrada is None:
        raise gr.Error("Por favor, subí una imagen antes de ejecutar el análisis.")

    # 1. Preprocesamiento: aseguramos dimensiones seguras para espacio latente
    img_base = preparar_imagen_entrada(imagen_entrada, tamano_objetivo=256)
    ancho_final, alto_final = img_base.width * 2, img_base.height * 2

    # 2. Línea base clásica (Interpolación Bicúbica):
    # Calcula nuevos píxeles promediando vecinos. No inventa detalles nuevos,
    # por eso conserva bordes suaves/borrosos. Representa la evidencia visual directa.
    img_bicubica = img_base.resize((ancho_final, alto_final), Image.Resampling.BICUBIC)

    # 3. Reconstrucción por Difusión Latente:
    # Usamos un generador con semilla fija para garantizar reproducibilidad.
    generador = torch.Generator(device=DEVICE).manual_seed(int(semilla))
    prompt_efectivo = prompt_guia.strip() if prompt_guia.strip() else "high resolution photograph, sharp details"

    resultado_difusion = upscaler(
        prompt=prompt_efectivo,
        image=img_base,
        num_inference_steps=int(pasos_inferencia),
        guidance_scale=float(escala_guia),
        generator=generador,
    ).images[0]

    # 4. Evidencia visual de cambios (Mapa de diferencias)
    mapa_diferencias = generar_mapa_diferencias(img_bicubica, resultado_difusion)

    # 5. Análisis Semántico con CLIP (Antes vs. Después)
    scores_antes = analizar_con_clip(img_bicubica)
    scores_despues = analizar_con_clip(resultado_difusion)

    # Armamos el informe explicativo para la persona usuaria
    lineas_reporte = [
        f"### Informe de Procesamiento ({img_base.width}×{img_base.height} px → {ancho_final}×{alto_final} px)",
        f"- **Parámetros utilizados:** Semilla `{int(semilla)}` | Pasos de denoising: `{int(pasos_inferencia)}` | Guía semántica: `{escala_guia}`",
        "",
        "#### Comparación Semántica con CLIP (Similitud relativa)",
        "| Descripción evaluada | Base Clásica (Bicúbica) | Reconstrucción (Difusión) | Variación |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for etiqueta in ETIQUETAS_CLIP:
        v_antes = scores_antes[etiqueta] * 100
        v_despues = scores_despues[etiqueta] * 100
        delta = v_despues - v_antes
        signo = "+" if delta >= 0 else ""
        lineas_reporte.append(
            f"| {etiqueta} | {v_antes:.1f}% | {v_despues:.1f}% | {signo}{delta:.1f}% |"
        )

    lineas_reporte.extend([
        "",
        "#### Interpretación y límites del análisis",
        "- **¿Qué mide CLIP?** Evalúa qué tan coherente parece la imagen frente a conceptos textuales aprendidos. Un aumento en *'detalles definidos'* indica mayor **plausibilidad visual**, pero **no certifica autenticidad**.",
        "- **¿Qué información fue inventada?** Todo detalle de alta frecuencia (letras pequeñas, rasgos faciales, texturas de materiales) que no era legible en la imagen base de entrada fue estimado estadísticamente por el modelo de difusión a partir de su entrenamiento previo.",
    ])

    return (
        (img_bicubica, resultado_difusion),
        mapa_diferencias,
        "\n".join(lineas_reporte),
    )


# ==============================================================================
# 3. INTERFAZ WEB EN GRADIO (ENFOCADA EN TRANSPARENCIA Y CONTROL)
# ==============================================================================
ADVERTENCIA_TRANSPARENCIA = """
> ⚠️ **ADVERTENCIA DE RECONSTRUCCIÓN POR INTELIGENCIA ARTIFICIAL:**
> Este sistema aplica **super-resolución mediante difusión latente**. Los detalles finos (caracteres tipográficos, números, rostros o texturas) que no eran visibles en la miniatura original **son estimaciones generativas plausibles, NO información recuperada del documento real**. No utilice el resultado como prueba documental, pericial o histórica sin verificación independiente.
"""

with gr.Blocks(title="MVP Imagen Digital — Super-Resolución Crítica") as demo:
    gr.Markdown("# Ampliación Crítica de Miniaturas de Archivo (Super-Resolución x2)")
    gr.Markdown(
        "Herramienta de apoyo para catalogación digital: compara una ampliación matemática clásica "
        "(**interpolación bicúbica**) contra una reconstrucción generativa (**difusión latente**) "
        "haciendo explícito qué es evidencia y qué es síntesis."
    )
    gr.Markdown(ADVERTENCIA_TRANSPARENCIA)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="1. Cargar miniatura o recorte (se ajustará a máx. 256px)")
            prompt_txt = gr.Textbox(
                label="2. Orientación semántica opcional (Prompt)",
                value="archival photograph, clear street sign text, sharp details",
                info="Orienta el espacio latente hacia el tipo de escena esperada.",
            )
            with gr.Accordion("3. Parámetros del proceso de difusión (Experimentación)", open=True):
                steps_slider = gr.Slider(
                    minimum=4, maximum=50, value=20, step=1,
                    label="Pasos de inferencia (Denoising steps)",
                    info="Más pasos refinan patrones, pero también pueden consolidar detalles inventados.",
                )
                seed_number = gr.Number(
                    value=42, precision=0,
                    label="Semilla aleatoria (Seed)",
                    info="Cambiar la semilla genera trayectorias distintas de reconstrucción sobre la misma imagen.",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=12.0, value=5.0, step=0.5,
                    label="Escala de guía textual (Guidance Scale)",
                    info="Controla cuánto peso tiene el texto frente a la imagen original.",
                )

            btn_procesar = gr.Button("Ejecutar Reconstrucción y Análisis", variant="primary")

        with gr.Column(scale=2):
            slider_comparador = gr.ImageSlider(
                label="Comparador: Izquierda = Interpolación Bicúbica (Dato clásico) | Derecha = Difusión (IA)",
                type="pil",
            )
            with gr.Row():
                img_diferencias = gr.Image(
                    label="Mapa de diferencias absolutas (x3 contraste): zonas brillantes = mayor intervención de la IA",
                    type="pil",
                )
                reporte_md = gr.Markdown("### El informe de análisis aparecerá aquí tras procesar una imagen.")

    btn_procesar.click(
        fn=ejecutar_pipeline,
        inputs=[input_img, prompt_txt, steps_slider, seed_number, guidance_slider],
        outputs=[slider_comparador, img_diferencias, reporte_md],
    )

if __name__ == "__main__":
    # share=True genera un enlace público temporal gratuito indispensable para Google Colab
    demo.launch(share=True)