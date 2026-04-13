# Clasificación de Residuos con Transfer Learning (VGG16)

##  Descripción del Proyecto

Clasificador de imágenes para distinguir entre **residuos orgánicos (O)** y **reciclables (R)** usando **Transfer Learning** con el modelo pre-entrenado **VGG16** en ImageNet.

El proyecto implementa dos enfoques de aprendizaje:
- **Extracción de características**: Todas las capas de VGG16 congeladas (feature extraction)
- **Fine-tuning**: Última capa convolucional descongelada para mejorar el rendimiento

---

##  Objetivo

Crear un modelo de deep learning que pueda clasificar automáticamente imágenes de residuos como:
- **Orgánicos (O)**: Desechos biodegradables
- **Reciclables (R)**: Materiales reciclables (plástico, vidrio, metal, papel)

Esto facilita la **clasificación automática de residuos** en sistemas de gestión ambiental.

---

##  Dataset

**Origen**: IBM Cloud Object Storage

**Nombre**: `o-vs-r-split-reduced-1200.zip`

**Características**:
- Imágenes de residuos orgánicos y reciclables
- **Resolución**: 150x150 píxeles
- **Clases**: 2 (Orgánico, Reciclable)
- **División**: 
  - 80% Entrenamiento
  - 20% Validación
  - Conjunto de prueba separado

**URL de descarga**:
```
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip
```

---

##  Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|-----------|---------|----------|
| **TensorFlow** | 2.17.0 | Framework de deep learning |
| **Keras** | Integrado en TF | API de alto nivel para modelos |
| **NumPy** | - | Operaciones numéricas |
| **Scikit-learn** | 1.5.1 | Métricas de evaluación |
| **Matplotlib** | 3.9.2 | Visualización de resultados |
| **tqdm** | - | Barras de progreso |

---

##  Requisitos Previos

### Instalación de Dependencias

```bash
pip install tensorflow==2.17.0
pip install numpy
pip install scikit-learn==1.5.1
pip install matplotlib==3.9.2
pip install tqdm
```

### Requisitos del Sistema

- **Python**: 3.8 o superior
- **Memoria RAM**: Mínimo 4GB (recomendado 8GB+)
- **GPU** (opcional): CUDA para aceleración (NVIDIA GPU)
- **Almacenamiento**: ~2GB para el dataset

---

##  Estructura del Proyecto

```
Clasificacion_Residuos_Transfer_Learning/
├── README.md                           # Este archivo
├── Clasificacion_Residuos_Transfer_Learning.ipynb  # Notebook principal
├── data/
│   ├── o-vs-r-split/
│   │   ├── train/                     # Datos de entrenamiento
│   │   │   ├── O/                     # Imágenes de residuos orgánicos
│   │   │   └── R/                     # Imágenes de residuos reciclables
│   │   └── test/                      # Datos de prueba
│   │       ├── O/
│   │       └── R/
│   └── o-vs-r-split-reduced-1200.zip  # Dataset comprimido
└── results/                            # Resultados y modelos entrenados
    ├── model_feature_extraction.h5    # Modelo con feature extraction
    ├── model_fine_tuning.h5           # Modelo con fine-tuning
    ├── training_history.png           # Gráficos de entrenamiento
    └── confusion_matrix.png           # Matriz de confusión
```

---

##  Uso

### 1. **Ejecutar el Notebook Completo**

```bash
jupyter notebook Clasificacion_Residuos_Transfer_Learning.ipynb
```

### 2. **Pasos Principales del Notebook**

#### Step 1: Verificar TensorFlow
```python
print('Versión de TensorFlow:', tf.__version__)
```

#### Step 2: Descargar el Dataset
El notebook descarga automáticamente el dataset desde IBM Cloud Object Storage y lo extrae.

#### Step 3: Configurar Parámetros
```python
img_rows, img_cols = 150, 150   # Dimensiones de las imágenes
batch_size = 32                  # Imágenes por lote
n_epochs = 10                    # Épocas máximas
n_classes = 2                    # Clases (O y R)
val_split = 0.2                  # 20% validación
```

#### Step 4: Crear Generadores de Datos
- **ImageDataGenerator**: Aplica data augmentation al conjunto de entrenamiento
- Normaliza imágenes al rango [0, 1]
- Aplica transformaciones aleatorias:
  - Desplazamiento horizontal (±10%)
  - Desplazamiento vertical (±10%)
  - Zoom (±10%)
  - Rotación (±10°)

#### Step 5: Cargar VGG16 Pre-entrenado
```python
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)
```

#### Step 6: Entrenar Dos Versiones del Modelo

**Versión 1: Feature Extraction**
- Todas las capas de VGG16 congeladas
- Solo entrena las capas finales (Dense)
- Entrenamiento más rápido

**Versión 2: Fine-tuning**
- Última capa convolucional descongelada
- Ajusta pesos de VGG16
- Mejor rendimiento (más lento)

#### Step 7: Evaluar en Conjunto de Prueba
- Calcula precisión, recall, F1-score
- Genera matriz de confusión
- Visualiza resultados

---

##  Arquitectura del Modelo

### VGG16 (Base)
```
Input (150, 150, 3)
    ↓
Bloques convolucionales 1-5 (VGG16)
    ↓
Flatten
    ↓
Dense (1024, relu) + Dropout (0.5)
    ↓
Dense (512, relu) + Dropout (0.5)
    ↓
Dense (1, sigmoid)  ← Salida (Orgánico vs Reciclable)
```

### Capas Agregadas
```python
Dense(1024, activation='relu')
Dropout(0.5)
Dense(512, activation='relu')
Dropout(0.5)
Dense(1, activation='sigmoid')  # Clasificación binaria
```

---

##  Conceptos Implementados

### Transfer Learning
Utiliza pesos pre-entrenados de VGG16 en ImageNet, ahorrando tiempo y mejorando precisión con pocos datos.

### Data Augmentation
Aplica transformaciones aleatorias a las imágenes de entrenamiento para:
- Aumentar tamaño efectivo del dataset
- Mejorar generalización del modelo
- Reducir overfitting

### Fine-tuning
Descongela la última capa convolucional de VGG16 para:
- Adaptar características a la tarea específica
- Mejorar rendimiento del modelo
- Usar learning rate bajo para evitar perder conocimiento pre-entrenado

### Early Stopping
Detiene el entrenamiento cuando la pérdida de validación no mejora:
- Evita overfitting
- Ahorra tiempo de entrenamiento
- Usa el mejor modelo encontrado

---

##  Métricas de Evaluación

El modelo es evaluado usando:

| Métrica | Descripción |
|---------|------------|
| **Accuracy** | Proporción de predicciones correctas |
| **Precision** | De las predicciones positivas, cuántas fueron correctas |
| **Recall** | De los casos positivos reales, cuántos fueron detectados |
| **F1-Score** | Media armónica de precisión y recall |
| **Matriz de Confusión** | Visualización de verdaderos/falsos positivos y negativos |

---

##  Salidas del Proyecto

### Modelos Entrenados
- `model_feature_extraction.h5` - Modelo con todas las capas congeladas
- `model_fine_tuning.h5` - Modelo con fine-tuning

### Visualizaciones
- **Gráficos de entrenamiento**: Precisión y pérdida por época
- **Matriz de confusión**: Distribución de predicciones
- **Curvas ROC**: Rendimiento a diferentes umbrales

### Reportes
- Métricas de precisión, recall y F1-score
- Comparación entre ambas versiones del modelo
- Recomendaciones para mejora

---

##  Resultados Esperados

### Versión Feature Extraction
- **Tiempo de entrenamiento**: ~5-10 minutos
- **Precisión esperada**: 85-92%
- **Ventaja**: Rápido, suficiente para la mayoría de casos

### Versión Fine-tuning
- **Tiempo de entrenamiento**: ~15-30 minutos
- **Precisión esperada**: 90-95%
- **Ventaja**: Mejor rendimiento, generaliza mejor

---

##  Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
pip install tensorflow==2.17.0
```

### Error: "CUDA/GPU no disponible"
El modelo funcionará en CPU (más lento pero funcional)

### Bajo rendimiento en validación
- Aumentar data augmentation
- Entrenar más épocas
- Usar fine-tuning en lugar de feature extraction
- Ajustar hyperparámetros (learning rate, batch size)

### Memoria insuficiente
- Reducir `batch_size` (ej: 16 en lugar de 32)
- Reducir resolución de imágenes
- Usar Google Colab con GPU gratuita

---

##  Referencias y Recursos

### Papers Relevantes
- **VGG16**: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- **Transfer Learning**: Pan, S. J., & Yang, Q. (2010). "A Survey on Transfer Learning"
- **ImageNet**: Deng, J., et al. (2009). "ImageNet: A large-scale hierarchical image database"

### Documentación Oficial
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [VGG16 Pre-entrenado](https://keras.io/api/applications/vgg/#vgg16-function)
- [ImageDataGenerator](https://keras.io/api/preprocessing/image/)

### Tutoriales Relacionados
- Transfer Learning con Keras
- Data Augmentation para Computer Vision
- Fine-tuning de modelos pre-entrenados

---

##  Autor y Contexto

**Proyecto**: Clasificación de Residuos con Transfer Learning

**Tecnología Base**: IBM Cloud, TensorFlow, Keras

**Objetivo Educativo**: Demostrar transfer learning en clasificación de imágenes

**Aplicación Práctica**: Sistemas de gestión ambiental y clasificación automática de residuos

---

## Licencia

Este proyecto es de código abierto y disponible para propósitos educativos y comerciales.

---

## Contribuciones

Para mejoras o reportar errores:
1. Ejecuta el notebook y documenta cualquier problema
2. Sugiere cambios en hyperparámetros
3. Propone nuevas arquitecturas o técnicas de augmentation

---

## Contacto y Soporte

Para preguntas o soporte técnico, consulta:
- Documentación de TensorFlow
- Stack Overflow (tag: tensorflow, keras)
- GitHub Issues (si aplica)

---

**Última actualización**: Abril 2026

**Estado**: Funcional y probado 
