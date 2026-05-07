# 🎓 INSTRUCCIONES PASO-A-PASO: Google Colab

## 📋 Pre-requisitos

- ✅ Cuenta Google (Gmail)
- ✅ Acceso a Google Colab (gratis)
- ✅ Fotos JPG/PNG/HEIC para clasificar

---

## 🚀 PASO 1: Abre el Notebook en Colab

### Opción A: Desde GitHub
```
1. Ve a: https://github.com/[TU_USUARIO]/eigenfaces-recognition
2. Busca archivo: eigenfaces_recognition.ipynb
3. Click derecho → Abrir con → Google Colaboratory
   (O haz click en botón "Open in Colab")
```

### Opción B: URL Directa
```
https://colab.research.google.com/github/[TU_USUARIO]/eigenfaces-recognition/blob/main/eigenfaces_recognition.ipynb
```

### Opción C: Manual
```
1. Abre https://colab.research.google.com
2. File → Open notebook → GitHub
3. Pega URL del repo: https://github.com/[TU_USUARIO]/eigenfaces-recognition
4. Selecciona eigenfaces_recognition.ipynb
5. Click Open
```

---

## 🔐 PASO 2: Autoriza Google Drive

**Primera celda del notebook:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**¿Qué pasa?**
1. Aparece link azul: "Click here to authenticate"
2. Click en el link
3. Selecciona tu cuenta Google
4. Click "Allow"
5. Copia el código de autorización
6. Pégalo en el cuadro
7. Enter

**Resultado esperado:**
```
Mounted at /content/drive
```

---

## 📁 PASO 3: Prepara Estructura en Google Drive

### 3A: Crea Carpeta Base
```
Google Drive
└── eigenfaces-project/
    ├── Fotos_para_clasificar/
    ├── datos_entrenamiento/     (opcional, para reentrenar)
    └── resultados/
```

### 3B: Sube Fotos para Clasificar
```
Mi unidad (Google Drive)
└── eigenfaces-project/
    └── Fotos_para_clasificar/
        ├── foto_evento1.jpg
        ├── foto_reunion.jpg
        ├── prueba.png
        └── ... más fotos
```

**Cómo subir:**
1. Abre Google Drive
2. Crea carpeta: `eigenfaces-project`
3. Dentro, crea: `Fotos_para_clasificar`
4. Abre la carpeta
5. Click "Nuevo" → Cargar archivos
6. Selecciona tus fotos
7. Espera a que suban

---

## ⚙️ PASO 4: Ejecuta Celdas de Setup

### Celda 1: Autorización Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
✅ Click Run (▶️)

### Celda 2: Instalar Dependencias
```python
!pip install mtcnn dlib pillow_heif scikit-optimize -q
```
✅ Click Run (▶️)
⏳ Espera 2-3 minutos

### Celda 3: Importar Librerías
```python
import os
import cv2
import numpy as np
# ... etc
```
✅ Click Run (▶️)

---

## 📊 PASO 5: Cargar Modelo Entrenado

### Celda 4: Cargar Archivos .pkl
```python
import joblib

# Cargar modelo entrenado
modelo = joblib.load('/content/drive/MyDrive/[TU_CARPETA]/modelo_eigenfaces.pkl')
nombres_unicos = joblib.load('/content/drive/MyDrive/[TU_CARPETA]/nombres_unicos.pkl')

print("✅ Modelo cargado")
print(f"Personas reconocidas: {nombres_unicos}")
```

✅ Click Run (▶️)

**¿Dónde están los .pkl?**
```
Opción A: Los bajaste de GitHub
  → Sube a: /eigenfaces-project/

Opción B: Los creaste en tu notebook anterior
  → Están en: /content/drive/MyDrive/[RUTA_DONDE_GUARDASTE]/

Opción C: No tienes los archivos
  → Sigue instrucciones al final para entrenar
```

---

## 🎯 PASO 6: Clasificar Fotos Individuales

### Celda 5: Función de Identificación
```python
def identificar_cara(ruta_foto, mostrar_imagen=True):
    """
    Identifica una cara en una foto nueva.
    
    Args:
        ruta_foto: ruta a la imagen
        mostrar_imagen: si True, visualiza la cara detectada
    
    Returns:
        dict con predicción y confianza
    """
    try:
        # Cargar imagen
        img_bgr = cv2.imread(ruta_foto)
        if img_bgr is None:
            return {'error': 'No se pudo cargar la imagen'}
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Detectar cara
        from mtcnn import MTCNN
        detector = MTCNN()
        res = detector.detect_faces(img_rgb)
        if not res:
            return {'error': 'No se detectó cara en la imagen'}
        
        # Recortar
        x, y, w, h = res[0]['box']
        padding = 0.2
        x = max(0, int(x - w * padding))
        y = max(0, int(y - h * padding))
        w = int(w * (1 + 2 * padding))
        h = int(h * (1 + 2 * padding))
        
        face = img_rgb[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        face_resized = cv2.equalizeHist(face_resized)
        
        # Normalizar
        face_vector = face_resized.flatten() / 255.0
        
        # Predicción
        nombre = modelo.predict(face_vector.reshape(1, -1))[0]
        probabilidades = modelo.predict_proba(face_vector.reshape(1, -1))[0]
        confianza = np.max(probabilidades)
        
        # Visualizar
        if mostrar_imagen:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(face_gray, cmap='gray')
            ax.set_title(f"Identificado: {nombre.upper()}\nConfianza: {confianza:.2%}", 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.show()
        
        return {
            'nombre': nombre,
            'confianza': confianza,
            'probabilidades': dict(zip(nombres_unicos, probabilidades))
        }
    
    except Exception as e:
        return {'error': str(e)}
```

✅ Click Run (▶️)

### Celda 6: Clasificar Una Foto
```python
# Cambiar la ruta a tu foto
ruta_foto = '/content/drive/MyDrive/eigenfaces-project/Fotos_para_clasificar/foto1.jpg'

resultado = identificar_cara(ruta_foto)

if 'error' not in resultado:
    print(f"\n✅ RESULTADO:")
    print(f"   Persona: {resultado['nombre'].upper()}")
    print(f"   Confianza: {resultado['confianza']:.1%}")
    
    print(f"\n📊 Probabilidades detalladas:")
    for nombre, prob in sorted(resultado['probabilidades'].items(), 
                               key=lambda x: x[1], reverse=True)[:3]:
        print(f"   {nombre:<25} {prob:.1%}")
else:
    print(f"❌ Error: {resultado['error']}")
```

✅ Click Run (▶️)

**Resultado esperado:**
```
✅ RESULTADO:
   Persona: AGUSTINA_SEBBEN
   Confianza: 98.4%

📊 Probabilidades detalladas:
   agustina_sebben          98.4%
   belen_                   1.2%
   fede_                    0.3%
```

---

## 📂 PASO 7: Clasificar Carpeta Completa

### Celda 7: Procesar Lote
```python
import os
import json

ruta_carpeta = '/content/drive/MyDrive/eigenfaces-project/Fotos_para_clasificar'
resultados = {}

print("🔄 Clasificando fotos...\n")

for foto in sorted(os.listdir(ruta_carpeta)):
    if not foto.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
        continue
    
    ruta_completa = os.path.join(ruta_carpeta, foto)
    resultado = identificar_cara(ruta_completa, mostrar_imagen=False)
    
    if 'error' not in resultado:
        resultados[foto] = {
            'nombre': resultado['nombre'],
            'confianza': float(resultado['confianza']),
            'probabilidades': {k: float(v) for k, v in resultado['probabilidades'].items()}
        }
        print(f"✅ {foto:<40} → {resultado['nombre']:<25} ({resultado['confianza']:.1%})")
    else:
        print(f"❌ {foto:<40} → ERROR: {resultado['error'][:40]}")

# Guardar resultados en JSON
ruta_salida = '/content/drive/MyDrive/eigenfaces-project/resultados/clasificaciones.json'
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

with open(ruta_salida, 'w') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"\n✅ Resultados guardados en: {ruta_salida}")
print(f"Total clasificadas: {len(resultados)}")
```

✅ Click Run (▶️)

---

## 📊 PASO 8: Ver Resultados

### Celda 8: Estadísticas
```python
import json
import pandas as pd

with open('/content/drive/MyDrive/eigenfaces-project/resultados/clasificaciones.json', 'r') as f:
    resultados = json.load(f)

# Crear tabla
datos = []
for foto, info in resultados.items():
    datos.append({
        'Foto': foto,
        'Persona': info['nombre'],
        'Confianza': f"{info['confianza']:.1%}"
    })

df = pd.DataFrame(datos)
print(df.to_string(index=False))

# Estadísticas por persona
print("\n📊 RESUMEN POR PERSONA:")
personas = {}
for info in resultados.values():
    nombre = info['nombre']
    if nombre not in personas:
        personas[nombre] = 0
    personas[nombre] += 1

for nombre, count in sorted(personas.items(), key=lambda x: x[1], reverse=True):
    print(f"   {nombre:<25} {count:>3} fotos")
```

✅ Click Run (▶️)

---

## ⚠️ TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'mtcnn'"
**Solución:**
1. Ejecuta Celda 2 nuevamente (instalaciones)
2. Espera a que termine
3. Click Restart runtime (arriba a la derecha)
4. Intenta de nuevo

### "FileNotFoundError: modelo_eigenfaces.pkl"
**Solución:**
1. Verifica que el archivo está en Google Drive
2. Copia la ruta exacta
3. Comprueba: `/content/drive/MyDrive/[TU_CARPETA]/`

### "No se detectó cara en la imagen"
**Causas:**
- Foto de lado (no frontal)
- Rostro muy pequeño
- Luz muy baja/alta
- Sin rostro en la imagen

**Solución:**
- Usa fotos frontales
- Rostro debe ocupar 30-80% de la imagen
- Buena iluminación

### "CUDA out of memory" / "Kernel died"
**Solución:**
1. Runtime → Restart all runtimes
2. Runtime → Change runtime type → GPU (si no está)
3. Intenta de nuevo

---

## 🎓 TIPS

### Tip 1: Acelera con GPU
```
Runtime → Change runtime type
Accelerator: GPU (T4 o mejor)
Click Save
```

### Tip 2: Guarda Resultados
```python
# Al final, descarga el JSON
from google.colab import files
files.download('/content/drive/MyDrive/eigenfaces-project/resultados/clasificaciones.json')
```

### Tip 3: Usar Sin Mostrar Imágenes
```python
# Más rápido (no muestra visualización)
resultado = identificar_cara(ruta_foto, mostrar_imagen=False)
```

### Tip 4: Filtrar por Confianza
```python
if resultado['confianza'] < 0.85:
    print(f"⚠️  Confianza baja, revisar manualmente")
```

---

## 📋 CHECKLIST FINAL

- [ ] Autorizaste Google Drive
- [ ] Subiste fotos a `Fotos_para_clasificar`
- [ ] Instalaste dependencias (Celda 2)
- [ ] Cargaste modelo (Celda 4)
- [ ] Ejecutaste función `identificar_cara` (Celda 5)
- [ ] Clasificaste una foto (Celda 6)
- [ ] Procesaste carpeta completa (Celda 7)
- [ ] Viste resultados (Celda 8)

---

## 🎉 ¡LISTO!

Tu modelo está funcionando y clasificando caras. Ahora puedes:

1. ✅ Subir nuevas fotos a `Fotos_para_clasificar`
2. ✅ Ejecutar Celda 7 nuevamente
3. ✅ Ver resultados en JSON

**¡Sin necesidad de reentrenar!**

---

**¿Dudas? Revisa README.md o abre un Issue en GitHub** 🚀
