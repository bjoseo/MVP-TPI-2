# Contexto y Reglas del Proyecto (AGENTS.md)

## Descripción General

MVP para el Trabajo Integrador N°2 de Procesamiento Digital de Imágenes (IFTS 24, 2026).
Aplicación web en Gradio para super-resolución crítica (x2) de miniaturas de archivo mediante difusión latente, comparada contra interpolación bicúbica clásica y evaluada con CLIP.

## Stack y Entorno

- **Lenguaje:** Python 3.12
- **Gestor de dependencias:** Astral `uv` (`pyproject.toml` y `uv.lock`).
- **Hardware objetivo:** Google Colab con GPU T4 (15 GB VRAM) usando `torch.float16` y `enable_attention_slicing()`.
- **Modelos (100% abiertos, sin tokens):**
  - Difusión: `stabilityai/sd-x2-latent-upscaler`
  - Análisis visual: `openai/clip-vit-base-patch32`

## Reglas de Arquitectura y Código

1. Mantener separada la lógica en `src/` (`diffusion.py`, `analysis.py`, `honesty.py`) y la UI en `app.py`.
2. Cargar los modelos una sola vez a nivel de módulo para evitar recargas en cada inferencia.
3. Toda imagen de entrada debe preprocesarse con relleno neutro (zero-padding) a `256x256` px para garantizar divisibilidad exacta por 128 (`VAE x8 * U-Net x16`).
4. Conservar siempre visible la advertencia de transparencia sobre reconstrucción generativa frente a evidencia documental.
