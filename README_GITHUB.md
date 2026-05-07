# 🎭 Eigenfaces Face Recognition

Reconocimiento de rostros usando PCA y SVM. Modelo entrenado con 970+ fotos de 14 personas.

**Precisión: 93.93%** | **Velocidad: <100ms/rostro** | **Eficiente: 50 dimensiones**

---

## 📊 Rendimiento

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 93.93% |
| **Validación Cruzada** | 96.38% |
| **Personas Reconocidas** | 14 |
| **Total Imágenes Entrenamiento** | 970+ |
| **Dimensiones PCA** | 50 |

---

## 🚀 Inicio Rápido (Google Colab)

### 1. **Abre el Notebook en Colab**
```
Click aquí: https://colab.research.google.com/github/[TU_USUARIO]/[TU_REPO]/blob/main/eigenfaces_recognition.ipynb
```

### 2. **Autoriza Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. **Sube Fotos a Clasificar**
Crea esta estructura en tu Google Drive:
```
Mi unidad
└── Fotos_para_clasificar/
    ├── foto1.jpg
    ├── foto2.jpg
    ├── foto3.png
    └── ... más fotos
```

### 4. **Ejecuta la Celda de Predicción**
```python
# Cargar modelo
import joblib
modelo = joblib.load('modelo_eigenfaces.pkl')
nombres = joblib.load('nombres_unicos.pkl')

# Clasificar fotos nuevas
ruta_fotos = '/content/drive/MyDrive/Fotos_para_clasificar'

for foto in os.listdir(ruta_fotos):
    resultado = identificar_cara(f'{ruta_fotos}/{foto}')
    print(f"{foto}: {resultado['nombre']} ({resultado['confianza']:.1%})")
```

---

## 📁 Estructura del Repositorio

```
eigenfaces-recognition/
├── README.md                          ← Este archivo
├── INSTRUCCIONES_COLAB.md             ← Guía detallada
├── eigenfaces_recognition.ipynb       ← Notebook principal
├── modelos/
│   ├── modelo_eigenfaces.pkl          ← Modelo entrenado
│   └── nombres_unicos.pkl             ← Lista de personas
├── datos_entrenamiento/
│   ├── agustina_sebben/
│   ├── belen_/
│   ├── fede_/
│   └── ... (12 personas más)
├── scripts/
│   ├── predict.py                     ← Predicción en Python puro
│   └── batch_predict.py               ← Predicción por lotes
└── ejemplos/
    ├── foto_ejemplo1.jpg
    └── resultados_ejemplo.json
```

---

## 💻 Opción A: Colab (Recomendado - Sin instalaciones)

### Ventajas:
- ✅ No requiere instalar nada
- ✅ GPU gratuita
- ✅ Fácil de compartir

### Pasos:
1. Abre: `eigenfaces_recognition.ipynb`
2. Crea carpeta `/Fotos_para_clasificar` en Drive
3. Sube fotos
4. Ejecuta última celda para clasificar

---

## 🖥️ Opción B: Localmente (Python)

### Requisitos:
```bash
python >= 3.8
pip install numpy opencv-python scikit-learn pillow joblib mtcnn
```

### Instalación:
```bash
git clone https://github.com/[TU_USUARIO]/eigenfaces-recognition.git
cd eigenfaces-recognition
pip install -r requirements.txt
```

### Uso:
```bash
# Clasificar una foto
python scripts/predict.py --ruta foto_nueva.jpg

# Clasificar carpeta completa
python scripts/batch_predict.py --carpeta ./Fotos_para_clasificar
```

---

## 📸 Cómo Usar

### Paso 1: Prepara tus Fotos
- **Formato:** JPG, PNG, HEIC
- **Resolución:** 400x400 mínimo
- **Contenido:** Rostro frontal o 3/4
- **Luz:** Bien iluminado

### Paso 2: Sube a Google Drive
```
Fotos_para_clasificar/
├── persona1_foto1.jpg
├── persona1_foto2.jpg
├── evento_grupal.jpg
└── ...
```

### Paso 3: Ejecuta Predicción
```python
# En Colab
resultado = identificar_cara('/content/drive/MyDrive/Fotos_para_clasificar/foto.jpg')
print(f"Es: {resultado['nombre']}")
print(f"Confianza: {resultado['confianza']:.1%}")
```

### Paso 4: Interpreta Resultados
```
Confianza > 90%  → Muy confiable ✅
Confianza 80-90% → Probable ✅
Confianza < 80%  → Revisar manualmente ⚠️
```

---

## 👥 Personas Reconocidas (14)

```
1. agustina_sebben      10. martin
2. belen_               11. mati_villanueva
3. fede_                12. migue
4. guille_              13. millie
5. ignacio_paberolis    14. tomas_delbo
6. juan_cacchione
7. judi_
8. lucia_
9. mariangeles_
```

---

## 🔍 Interpretación de Resultados

### Salida Típica:
```python
{
    'nombre': 'agustina_sebben',           # Persona identificada
    'confianza': 0.9842,                   # 0.0-1.0 (98.42%)
    'probabilidades': {                    # Todas las probabilidades
        'agustina_sebben': 0.9842,
        'belen_': 0.0102,
        'fede_': 0.0034,
        ...
    },
    'face_img': numpy_array                # Rostro detectado
}
```

### Errores Posibles:
```python
{
    'error': 'No se detectó cara en la imagen'
    # → Foto sin rostro claro, de lado, o muy pequeña
}

{
    'error': 'Imagen muy borrosa o pequeña'
    # → Rostro < 400px, imagen borrosa
}
```

---

## ⚙️ Configuración Avanzada

### Ajustar Confianza Mínima:
```python
def identificar_cara(ruta_foto, confianza_minima=0.80):
    resultado = ...
    if resultado['confianza'] < confianza_minima:
        return {'error': 'Confianza muy baja'}
    return resultado
```

### Filtrar por Persona:
```python
def identificar_cara(ruta_foto, personas_permitidas=None):
    resultado = ...
    if personas_permitidas and resultado['nombre'] not in personas_permitidas:
        return {'error': f"Persona no permitida"}
    return resultado
```

### Procesar Batch:
```python
import os
from pathlib import Path

carpeta = './Fotos_para_clasificar'
resultados = {}

for foto in os.listdir(carpeta):
    ruta = os.path.join(carpeta, foto)
    resultado = identificar_cara(ruta, mostrar=False)
    resultados[foto] = resultado

# Guardar resultados
import json
with open('resultados.json', 'w') as f:
    json.dump(resultados, f, indent=2)
```

---

## 🔧 Solución de Problemas

### "No se detectó cara"
**Causas:**
- Foto de lado o rotada
- Rostro muy pequeño (< 400px)
- Luz muy baja/alta
- Sin rostro en la imagen

**Solución:**
- Usa fotos frontales claras
- Rostro debe ocupar 30-80% de la imagen
- Buena iluminación

### "Error: No module named 'mtcnn'"
**Solución:**
```bash
pip install mtcnn
# O en Colab:
!pip install mtcnn
```

### "Confianza muy baja"
**Causas:**
- Persona no en el dataset
- Iluminación diferente al entrenamiento
- Expresión facial muy diferente
- Accesorios (gafas, sombrero)

**Solución:**
- Revisar que la persona esté en la lista
- Mejorar iluminación
- Sin accesorios que cambien rostro

---

## 📊 Entrenar con Nuevos Datos

Si quieres reentrenar el modelo con más fotos:

### 1. Prepara estructura:
```
datos_entrenamiento/
├── persona_nueva/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ...
└── ... (14 personas existentes)
```

### 2. Ejecuta en Colab:
```python
# Celda 7
procesar_dataset_mejorado(
    '/content/drive/MyDrive/datos_entrenamiento',
    '/content/drive/MyDrive/Caras_procesadas'
)

# Celda 13
grid_search.fit(X_train, y_train)

# Celda 19
joblib.dump(best_model, 'modelo_eigenfaces_nuevo.pkl')
```

---

## 📚 Artículos y Referencias

- Turk & Pentland (1991): "Eigenfaces for Recognition" - Paper original
- PCA: Principal Component Analysis
- SVM: Support Vector Machine

---

## 📄 Licencia

MIT License - Libre para usar y modificar

---

## 👤 Autor

Data Mining Avanzado - Maestría en Ciencia de Datos
Universidadad Austral

---

## 🤝 Contribuciones

¿Quieres mejorar? 
1. Fork el repositorio
2. Crea rama: `git checkout -b feature/mejora`
3. Commit: `git commit -m "Agrego mejora"`
4. Push: `git push origin feature/mejora`
5. Pull Request

---

## 📞 Soporte

**¿Problemas?**
- 📖 Lee `INSTRUCCIONES_COLAB.md`
- 🔍 Revisa la sección "Solución de Problemas"
- 💬 Abre un Issue en GitHub

---

**Hecho con ❤️ usando Eigenfaces y Machine Learning**

Last updated: 2026-05-07
