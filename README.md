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

Como detalle se añadio a ambas formas de detección un difuminado para las personas y matrículas para mantener la privacidad.

## Detección de texto mediante OCR
Una vez localizadas las matrículas, el siguiente paso consiste en **extraer el texto alfanumérico** que contienen.  
En este cuaderno se implementan **dos estrategias diferentes** para la detección y lectura del texto:

  1. Una solución moderna basada en **SmolVLM**, un modelo multimodal de visión y lenguaje.  
  2. Un enfoque clásico mediante **OCR tradicional con Tesseract**.

---

### Detección mediante SmolVLM (modelo multimodal)

SmolVLM es un modelo **multimodal** capaz de interpretar imágenes y texto de forma conjunta.  
En este caso, se utiliza para **describir o leer el contenido visual** de las matrículas detectadas.  
El modelo no requiere entrenamiento adicional: basta con enviarle una imagen recortada (la matrícula) y una instrucción textual.

### Detección mediante Tesseract (OCR Clásico)

En este caso se utiliza **Tesseract**, un motor OCR de código abierto desarrollado por Google, ampliamente utilizado en tareas de reconocimiento óptico de texto.  
A diferencia de SmolVLM, este método no realiza una interpretación semántica del contenido visual, sino que se centra exclusivamente en **detectar patrones de caracteres** dentro de una imagen binarizada.

El flujo de trabajo se basa en una serie de **transformaciones clásicas de procesamiento de imagen** antes de aplicar el OCR, con el objetivo de mejorar la legibilidad de los caracteres y reducir el ruido.

---

#### Proceso de detección paso a paso

  1. **Recorte de la matrícula** a partir del frame detectado por YOLO.  
  2. **Conversión a escala de grises** para simplificar la información visual.  
  3. **Suavizado mediante filtro Gaussiano** para eliminar pequeñas imperfecciones.  
  4. **Binarización adaptativa u Otsu**, que resalta los contornos del texto sobre el fondo.  
  5. **Aplicación del motor OCR de Tesseract**, configurado para priorizar caracteres alfanuméricos.

Para probar el rendimiento de ambos métodos se proceso el dataset presente en el proyecto llamado [OCR Detection](./OCR%20Detection/). Este dataset se compone de 100 imagenes de prueba con sus respectivos .json para confirmar la predicción de los métodos ya mencionados. El flujo de ejecución es el siguiente:

  1. Se extraen las imágenes del dataset
  2. Se Utiliza el modelo entrenado para detectar la matrícula
  3. Se pasa la región de interés a los métodos y estos devuelven el resultado


## Resultados y análisis
Los métodos comentados en el apartado anterior dan como resultado una serie de csv con los datos de las predicciones ([resultados_smol.csv](./resultados_smol.csv) y [resultados_tesseract.csv](./resultados_tesseract.csv)).
Se resaltan los siguientes puntos:

El modelo SmolVLM logra una mayor precisión en el reconocimiento de caracteres, incluso en condiciones de iluminación desfavorable o con matrículas parcialmente deterioradas.

Por otro lado, Tesseract (OCR clásico) presenta limitaciones significativas ante distorsiones, reflejos o variaciones en el ángulo de visión, reduciendo la tasa de aciertos en casos no ideales.

El tiempo de inferencia de SmolVLM es superior, ya que implica una red neuronal más compleja, mientras que Tesseract ofrece una ejecución más rápida y ligera. El modelo de SmolLVM tardó aproximádamente 45 minutos en ejecutarse (en cpu) mientras que tesseract tardo apenas 2 minutos.

<img width="540" height="374" alt="image" src="https://github.com/user-attachments/assets/e869b092-e22c-428b-bb2a-01e56f3e1251" />

Podemos extraer como conclusión que el modelo LVM es mucho más robusto y preciso, pero como consecuencia encontramos largos tiempos de ejecución y consumición de recursos. Por otro lado tesseract es muy susceptible a cambios de iluminación, ángulos, colores, etc, pero sus tiempos de ejecución son mucho menores.

## Autoría
Este trabajo ha sido realizado por Eduardo Gainza Koller.

## Fuentes y referencias

Durante el desarrollo de la práctica se consultaron o utilizaron las siguientes fuentes:

- Consultas realizadas a [ChatGPT](https://chatgpt.com/) sobre:
  - Datasets útiles para el desempeño del trabajo
  - Técnicas de detección de matrículas mediante análisis de contornos
  - Diseño de prompts efectivos para modelos de visión-lenguaje en tareas de OCR
  - Instalación y compatibilidad de dependencias
  - Cómo mantener el mejor resultado de OCR cuando un vehículo aparece en múltiples frames
- Artículo sobre SmolVLM en Hugging Face: [https://huggingface.co/spaces/ashkamath/smol-vlm](https://huggingface.co/spaces/ashkamath/smol-vlm)
- Matplotlib para generación de gráficas: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
