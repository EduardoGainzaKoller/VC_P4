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

Para entrenar el modelo se utilizo un dataset obtenido de kaggle llamado ["License Plate Detection Dataset"](https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset). Este dataset proporciona hasta diez mil imágenes de entrenamiento lo que favorece a la precisión del modelo. Sin embargo, por limitaciones técnicas se tuvo que utilizar el modelo nano de YOLO que reduce significativamente el rendimiento en la detección.


## Combinación de modelos y detección conjunta
Unza vez entrenado el modelo para la detección de matriculas se utilizo la combinación de este y el modelo predefinido de YOLO para detectar el conjutno de interés en el video de test porporcionado en el moddle de la asignatura. La idea se basa en tratar frame a frame con ambos modelos y si se detecta un objeto de las clases de interés envolverlo con una caja y mostrar la confianza de la predicción. Se muestra a continuación una imagen de ejemplo:

<img width="1093" height="443" alt="image" src="https://github.com/user-attachments/assets/52f4ba8d-72c1-4459-bd33-2c8cc3c33e98" />

## Detección alternativa mediante contornos
En esta sección se propone una **alternativa al modelo YOLO** para detectar matrículas utilizando **procesamiento clásico de imágenes** mediante **OpenCV**.  
El objetivo es comparar la robustez y precisión del aprendizaje profundo frente a un enfoque tradicional basado en operaciones sobre píxeles.

---

### Descripción general del proceso

El flujo de trabajo implementado puede dividirse en dos fases:

  1. **Detección de vehículos y personas** con el modelo YOLO preentrenado.  
  2. **Detección de matrículas** aplicando técnicas de contornos y filtrado geométrico.

El vídeo se procesa frame a frame desde `LicenseDetection.mp4`.  
Para cada fotograma:
  - Se detectan coches, camiones y personas con YOLO.   
  - Se buscan regiones candidatas a matrículas a través de análisis de contornos.  
  - Se **delimitan visualmente** los vehículos en azul y las matrículas detectadas en verde.

Aunque el método de detección mediante contornos es una alternativa válida y útil desde el punto de vista didáctico, presenta **desventajas notables** en comparación con los modelos de aprendizaje profundo como **YOLO**:

  1. **Falta de robustez ante condiciones reales**  
     - El algoritmo depende fuertemente de factores como la **iluminación**, el **contraste**, la **orientación de la cámara** o el **color del vehículo**.  
     - Un ligero cambio en la luz ambiental o en la posición del coche puede hacer que los contornos de la matrícula se pierdan o generen falsos positivos.
  
  2. **Ausencia de comprensión semántica**  
     - El método de contornos solo analiza **características geométricas básicas** (bordes, formas, tamaños), sin entender qué representa cada región.  
     - En cambio, YOLO aprende **patrones visuales complejos** y puede identificar correctamente una matrícula incluso si está parcialmente oculta, sucia o girada.
  
  3. **Dificultad para generalizar**  
     - Si el vídeo cambia de resolución, ángulo o tipo de matrícula, el método clásico requiere **ajustar manualmente parámetros** como el umbral, la relación de aspecto o el área del contorno.  
     - YOLO, al estar entrenado sobre grandes datasets, mantiene su rendimiento de forma **consistente** ante variaciones.
  
  4. **Mayor tasa de falsos positivos y negativos**  
     - Los contornos pueden confundir faros, rejillas o reflejos con una matrícula, generando **detecciones erróneas**.  
     - Los modelos de detección profunda, en cambio, aprenden a **distinguir patrones específicos**, minimizando estos errores.

