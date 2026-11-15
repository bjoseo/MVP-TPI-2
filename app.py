import os
import gradio as gr
from PIL import Image

from src.diffusion import preparar_imagen_entrada, ejecutar_difusion
from src.analysis import comparar_analisis_visual
from src.honesty import (
    ADVERTENCIA_TRANSPARENCIA,
    generar_linea_base_bicubica,
    generar_mapa_diferencias,
    construir_informe_markdown,
)


def procesar_imagen(
    imagen_entrada: Image.Image,
    prompt_guia: str,
    pasos_inferencia: int,
    semilla: int,
    escala_guia: float,
    progress=gr.Progress(),
):
    if imagen_entrada is None:
        raise gr.Error("Por favor, subí una imagen o elegí uno de los ejemplos antes de procesar.")

    try:
        progress(0.1, desc="1/4 Preparando imagen y ajustando dimensiones (256x256)...")
        img_base = preparar_imagen_entrada(imagen_entrada, tamano_objetivo=256)
        img_bicubica = generar_linea_base_bicubica(img_base, factor=2)

        progress(0.3, desc="2/4 Ejecutando super-resolución en espacio latente (Difusión)...")
        img_difusion = ejecutar_difusion(
            img_base=img_base,
            prompt_guia=prompt_guia,
            pasos_inferencia=pasos_inferencia,
            semilla=semilla,
            escala_guia=escala_guia,
        )

        progress(0.75, desc="3/4 Calculando mapa de diferencias absolutas...")
        mapa_diferencias = generar_mapa_diferencias(img_bicubica, img_difusion, realce=3.0)

        progress(0.85, desc="4/4 Evaluando plausibilidad semántica con CLIP...")
        resultados_clip = comparar_analisis_visual(img_bicubica, img_difusion)
        informe_md = construir_informe_markdown(
            dim_entrada=img_base.size,
            dim_salida=img_difusion.size,
            semilla=int(semilla),
            pasos=int(pasos_inferencia),
            escala_guia=float(escala_guia),
            resultados_clip=resultados_clip,
        )

        progress(1.0, desc="¡Procesamiento completado!")
        return (img_bicubica, img_difusion), mapa_diferencias, informe_md

    except Exception as error:
        raise gr.Error(f"Ocurrió un error al procesar la imagen: {str(error)}") from error


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
            input_img = gr.Image(
                type="pil",
                label="1. Cargar miniatura o recorte (se ajustará a 256×256 px)",
            )
            prompt_txt = gr.Textbox(
                label="2. Orientación semántica opcional (en inglés para mayor precisión)",
                value="archival photograph, clear street sign text, sharp details",
                info="Orienta el espacio latente hacia el tipo de escena esperada.",
            )
            with gr.Accordion("3. Parámetros del proceso de difusión (Experimentación)", open=True):
                steps_slider = gr.Slider(
                    minimum=4, maximum=50, value=20, step=1,
                    label="Pasos de inferencia (Denoising steps)",
                    info="Más pasos refinan patrones, pero pueden consolidar detalles inventados.",
                )
                seed_number = gr.Number(
                    value=42, precision=0,
                    label="Semilla aleatoria (Seed)",
                    info="Cambiar la semilla genera trayectorias distintas sobre la misma imagen.",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=12.0, value=5.0, step=0.5,
                    label="Escala de guía textual (Guidance Scale)",
                    info="Controla cuánto peso tiene el texto frente a la imagen original.",
                )

            btn_procesar = gr.Button("Ejecutar Reconstrucción y Análisis", variant="primary")

            # Carga automática de ejemplos si existen en la carpeta examples/
            ejemplos_disponibles = [
                [ruta, "archival photograph, clear text and sharp details", 20, 42, 5.0]
                for ruta in sorted(
                    os.path.join("examples", f)
                    for f in (os.listdir("examples") if os.path.isdir("examples") else [])
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                )
            ]
            if ejemplos_disponibles:
                gr.Examples(
                    examples=ejemplos_disponibles,
                    inputs=[input_img, prompt_txt, steps_slider, seed_number, guidance_slider],
                    label="O elegí una imagen de ejemplo del repositorio:",
                )

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
        fn=procesar_imagen,
        inputs=[input_img, prompt_txt, steps_slider, seed_number, guidance_slider],
        outputs=[slider_comparador, img_diferencias, reporte_md],
    )

if __name__ == "__main__":
    demo.launch(share=True)