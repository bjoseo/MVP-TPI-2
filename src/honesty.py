# ==============================================================================
# Implementa la separacion del dato matematico clasico de la estimacion generativa, calcular el mapa de diferencias absolutas y redactar el informe comprensible para la el usuario.
# ==============================================================================


from PIL import Image, ImageChops, ImageEnhance

ADVERTENCIA_TRANSPARENCIA = """
> ⚠️ **ADVERTENCIA DE RECONSTRUCCIÓN POR INTELIGENCIA ARTIFICIAL:**
> Este sistema aplica **super-resolución mediante difusión latente**. Los detalles finos (caracteres tipográficos, números, códigos QR, rostros o texturas) que no eran legibles en la miniatura original **son estimaciones generativas plausibles, NO información recuperada del documento real**. No utilice el resultado como prueba documental, pericial o histórica sin verificación independiente.
"""


def generar_linea_base_bicubica(img_base: Image.Image, factor: int = 2) -> Image.Image:
    """
    Aplica interpolación bicúbica clásica para duplicar la resolución.
    No inventa estructuras nuevas: representa la evidencia visual directa sin alucinaciones.
    """
    ancho_final = img_base.width * factor
    alto_final = img_base.height * factor
    return img_base.resize((ancho_final, alto_final), Image.Resampling.BICUBIC)


def generar_mapa_diferencias(img_clasica: Image.Image, img_ia: Image.Image, realce: float = 3.0) -> Image.Image:
    """
    Resta píxel a píxel la línea base clásica contra la reconstrucción por IA
    y realza el contraste para hacer visibles las zonas donde el modelo intervino más.
    """
    diff = ImageChops.difference(img_clasica, img_ia)
    return ImageEnhance.Contrast(diff).enhance(realce)


def construir_informe_markdown(
    dim_entrada: tuple[int, int],
    dim_salida: tuple[int, int],
    semilla: int,
    pasos: int,
    escala_guia: float,
    resultados_clip: list[dict],
) -> str:
    """
    Formatea el reporte técnico y de transparencia en lenguaje claro para el usuario.
    """
    lineas = [
        f"### Informe de Procesamiento ({dim_entrada[0]}×{dim_entrada[1]} px → {dim_salida[0]}×{dim_salida[1]} px)",
        f"- **Parámetros aplicados:** Semilla `{semilla}` | Pasos de denoising: `{pasos}` | Guía semántica: `{escala_guia}`",
        "",
        "#### Comparación Semántica con CLIP (Plausibilidad relativa)",
        "| Descripción evaluada | Base Clásica (Bicúbica) | Reconstrucción (Difusión) | Variación |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for fila in resultados_clip:
        signo = "+" if fila["delta"] >= 0 else ""
        lineas.append(
            f"| {fila['criterio']} | {fila['antes']:.1f}% | {fila['despues']:.1f}% | {signo}{fila['delta']:.1f}% |"
        )

    lineas.extend([
        "",
        "#### Interpretación y límites del análisis",
        "- **¿Qué mide CLIP?** Evalúa qué tan coherente resulta la imagen frente a conceptos visuales aprendidos. Un aumento en *'Imagen nítida'* indica mayor **plausibilidad visual**, pero **no certifica autenticidad**.",
        "- **¿Qué información fue inventada?** Todo detalle de alta frecuencia (letras pequeñas, números, códigos o rasgos faciales) que no era legible en la entrada fue estimado estadísticamente por difusión.",
    ])

    return "\n".join(lineas)