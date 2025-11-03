# Clasificador de Gestos Estáticos LSE (Lengua de Signos Española)

Proyecto de Computer Vision para clasificar gestos estáticos de la mano (letras A, B, C, D, E del LSE) usando MediaPipe y Machine Learning.

## 📋 Características

- **Captura de datos**: Interfaz intuitiva para recolectar imágenes de gestos
- **Extracción de características**: Uso de MediaPipe para detectar 21 landmarks de la mano
- **Múltiples modelos**: Comparación automática de Random Forest, SVM, KNN y Redes Neuronales
- **Reconocimiento en tiempo real**: Clasificación de gestos en vivo con webcam
- **Visualizaciones**: Gráficos de rendimiento y matriz de confusión

## 🎯 Gestos Soportados

- **A**: Puño cerrado
- **B**: Mano plana, dedos juntos
- **C**: Mano en forma de C
- **D**: Índice levantado, otros dedos doblados
- **E**: Todos los dedos doblados hacia la palma

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- Webcam funcional
- Sistema operativo: Windows, macOS o Linux

### Pasos de instalación

1. **Clonar o descargar el proyecto**

```bash
# Si tienes el código en un repositorio
git clone <tu-repositorio>
cd clasificador-gestos-lse
```

2. **Crear entorno virtual (recomendado)**

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

## 📖 Uso

### Paso 1: Capturar Datos

Ejecuta el script de captura para recolectar imágenes de tus gestos:

```bash
python capture_gestures.py
```

**Instrucciones:**
- Muestra el gesto indicado en pantalla
- Presiona **ESPACIO** para capturar una muestra
- El programa capturará 100 muestras por gesto automáticamente
- Presiona **'n'** para saltar al siguiente gesto manualmente
- Presiona **'q'** para salir

**Consejos para mejores resultados:**
- Mantén la mano en el centro del encuadre
- Varía ligeramente la posición y ángulo de la mano
- Asegúrate de tener buena iluminación
- Usa un fondo simple y sin distracciones

### Paso 2: Entrenar el Modelo

Una vez capturados los datos, entrena el clasificador:

```bash
python train_model.py
```

El script:
1. Carga los datos capturados
2. Entrena 4 modelos diferentes (Random Forest, SVM, KNN, MLP)
3. Evalúa cada modelo con validación cruzada
4. Selecciona automáticamente el mejor modelo
5. Genera visualizaciones y reportes
6. Guarda el modelo entrenado

**Archivos generados:**
- `data/gesture_model_latest.pkl` - Modelo entrenado (el más importante)
- `data/model_comparison.png` - Comparación de modelos
- `data/confusion_matrix.png` - Matriz de confusión
- `data/classification_report.txt` - Reporte detallado

### Paso 3: Reconocimiento en Tiempo Real

Usa el modelo entrenado para reconocer gestos en vivo:

```bash
python real_time_recognition.py
```

**Controles:**
- **'q'** - Salir de la aplicación
- **'r'** - Reiniciar el buffer de predicciones (útil si hay errores)
- **'h'** - Mostrar/ocultar los landmarks de la mano

**Información en pantalla:**
- Gesto predicho con color distintivo
- Nivel de confianza de la predicción
- FPS del sistema
- Probabilidades de cada clase (panel derecho)

## 🧠 Arquitectura del Sistema

### 1. Captura de Datos (`capture_gestures.py`)

```
Webcam → MediaPipe → Landmarks (21 puntos, x/y/z) → Normalización → Dataset
```

- Detecta 21 puntos clave de la mano (landmarks)
- Normaliza las coordenadas respecto a la muñeca
- Escala por la distancia muñeca-dedo medio
- Genera vectores de 63 características (21 puntos × 3 coordenadas)

### 2. Entrenamiento (`train_model.py`)

```
Dataset → Split (80/20) → Modelos ML → Validación → Mejor Modelo → .pkl
```

**Modelos evaluados:**
- **Random Forest**: Ensemble de árboles de decisión
- **SVM**: Support Vector Machine con kernel RBF
- **KNN**: K-Nearest Neighbors (k=5)
- **MLP**: Red neuronal (128-64-32 neuronas)

**Métricas:**
- Accuracy en conjunto de test
- Validación cruzada (5-fold)
- Precision, Recall, F1-Score por clase

### 3. Reconocimiento (`real_time_recognition.py`)

```
Webcam → MediaPipe → Normalización → Modelo → Suavizado → Predicción Final
```

- Procesamiento en tiempo real (~30 FPS)
- Buffer de predicciones para suavizado (reduce falsos positivos)
- Visualización de confianza y probabilidades

## 📊 Estructura de Archivos

```
clasificador-gestos-lse/
│
├── capture_gestures.py          # Script de captura de datos
├── train_model.py                # Script de entrenamiento
├── real_time_recognition.py     # Script de reconocimiento en vivo
├── requirements.txt              # Dependencias
├── README.md                     # Esta documentación
│
└── data/                         # Carpeta generada automáticamente
    ├── gestures_data_*.npy      # Datos de entrenamiento
    ├── gestures_labels_*.npy    # Etiquetas
    ├── metadata_*.json          # Información de captura
    ├── gesture_model_latest.pkl # Modelo entrenado (importante!)
    ├── model_comparison.png     # Visualización de modelos
    ├── confusion_matrix.png     # Matriz de confusión
    └── classification_report.txt # Reporte detallado
```

## 🔧 Solución de Problemas

### La webcam no funciona

```python
# En real_time_recognition.py, cambia el índice de la cámara:
cap = cv2.VideoCapture(0)  # Prueba con 1, 2, etc.
```

### Baja precisión del modelo

1. Captura más datos (aumenta `samples_per_gesture` en `capture_gestures.py`)
2. Mejora la calidad de las capturas (iluminación, fondo limpio)
3. Asegúrate de hacer los gestos de forma consistente

### El reconocimiento es muy sensible

En `real_time_recognition.py`, aumenta el tamaño del buffer:

```python
self.prediction_buffer = deque(maxlen=15)  # Por defecto es 10
```

### Errores de dependencias

```bash
# Reinstalar todas las dependencias
pip install --upgrade -r requirements.txt
```

## 🎨 Personalización

### Añadir más gestos

1. Modifica la lista en `capture_gestures.py`:
```python
self.gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # Añade más letras
```

2. Añade colores en `real_time_recognition.py`:
```python
self.colors = {
    'A': (255, 100, 100),
    'F': (100, 255, 255),  # Añade color para F
    'G': (255, 150, 100),  # Añade color para G
}
```

### Cambiar el número de muestras

En `capture_gestures.py`:
```python
self.samples_per_gesture = 150  # Por defecto es 100
```

### Ajustar confianza de detección

En `capture_gestures.py` o `real_time_recognition.py`:
```python
self.hands = self.mp_hands.Hands(
    min_detection_confidence=0.8,  # Aumenta para más precisión
    min_tracking_confidence=0.7
)
```

## 📈 Resultados Esperados

Con 100 muestras por gesto:
- **Accuracy esperada**: 90-98%
- **FPS**: 25-35 en hardware moderno
- **Tiempo de entrenamiento**: 10-30 segundos

## 🤝 Contribuciones

¿Mejoras o sugerencias? ¡Son bienvenidas!

## 📝 Notas Adicionales

- Los modelos se guardan automáticamente después del entrenamiento
- Puedes reentrenar en cualquier momento ejecutando `train_model.py`
- Los datos antiguos no se sobrescriben, se crean nuevos archivos con timestamp
- Para mejores resultados, captura datos con diferentes condiciones de iluminación

## 🔍 Referencias

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Lengua de Signos Española](https://www.cnse.es/)

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

**¡Disfruta clasificando gestos! 🤟**