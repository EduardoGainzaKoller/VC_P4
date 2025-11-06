# Práctica 4. Detección de vehículos y matrículas

## Contenidos
- [Descripción](#descripción)
- [Configuración del entorno](#configuración-del-entorno)
- [Entrenamiento del modelo YOLO para matrículas](#entrenamiento-del-modelo-yolo-para-matrículas)
- [Combinación de modelos y detección conjunta](#combinación-de-modelos-y-detección-conjunta)
- [Detección alternativa mediante contornos](#detección-alternativa-mediante-contornos)
- [Detección de texto mediante OCR](#deteccion-OCR)
- [Resultados y análisis](#resultados-y-análisis)
- [Autoría](#autoría)
- [Fuentes y referencias](#fuentes-y-referencias)

# Descripción

En este proyecto se desarrolla un modelo para la detección de vehículos, personas y matrículas. Además, se emplean técnicas de OCR para la detección del texto de la matrícula. A continuación, se detallan los componentes principales del trabajo:

  1. **Detección de vehículos y personas** utilizando un modelo preentrenado **YOLOv11**.  
  2. **Entrenamiento de un modelo YOLO personalizado** para la **detección de matrículas**.
  3. **SMOLLVM** y **Tesseract** para la detección de texto.

Al final se hace una comparativa entre los dos métodos de detección de texto.

## Configuración del entorno

Se crea un entorno virtual con **Conda** para asegurar la correcta instalación de dependencias.

### Crear el entorno

```bash
conda create --name VC_P4 python=3.9.5
conda activate VC_P4
pip install ultralytics opencv-python pillow torch torchvision torchaudio numpy transformers pytesseract matplotlib
```

## Entrenamiento del modelo YOLO para matrículas
La librerias de ultralytics ya proporciona modelos para la detección de vehículos y personas. Sim embargo, para el desarrollo de esta práctica también era necesario un modelo que detectara matrículas de los vehículos, para ello se entreno un modelo desde cero para este propósito:

```python
non_trained_model = YOLO("yolo11n.yaml")
print(torch.cuda.is_available())

results = non_trained_model.train(
    data='LICENSE_PLATE_DETECTION.yaml',
    epochs=100,                          # Número de épocas
    imgsz=640,                           # Tamaño de la imagen de entrada
    batch=16,                            # Tamaño del lote
    patience=10,                        # Paciencia para la detención temprana
    device=0                            # Usar GPU 0 para el entrenamiento
)
```
