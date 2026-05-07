# 🎭 Eigenfaces Face Recognition

Reconocimiento de rostros pre-entrenado. **Solo predice. Sin entrenar.**

**Precisión: 93.93% | Reconoce: 14 personas | Listo para usar**

---

## ⚡ Inicio Rápido (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[TU_USUARIO]/eigenfaces-recognition/blob/main/eigenfaces_prediction_only.ipynb)

1. Click en el botón "Open in Colab" arriba ☝️
2. Sube fotos a `Fotos_para_clasificar` en tu Google Drive
3. Ejecuta el notebook
4. Obtén predicciones

---

## 📊 Modelo

| Métrica | Valor |
|---------|-------|
| Precisión | 93.93% |
| Personas | 14 |
| Fotos Entrenamiento | 970+ |
| Componentes PCA | 50 |
| Algoritmo | SVM + PCA |

---

## 👥 Personas Reconocidas

1. agustina_sebben
2. belen_
3. fede_
4. guille_
5. ignacio_paberolis
6. juan_cacchione
7. judi_
8. lucia_
9. mariangeles_
10. martin
11. mati_villanueva
12. migue
13. millie
14. tomas_delbo

---

## 🚀 Cómo Usar

### En Colab (Recomendado)
```
1. Click botón "Open in Colab" arriba
2. Edita las rutas en PASO 4 (donde están tus .pkl)
3. Sube fotos a Google Drive
4. Ejecuta el notebook
5. Listo
```

### Localmente (Python)
```bash
pip install -r requirements.txt
python predict.py --ruta foto.jpg
python predict.py --carpeta ./fotos/
```

---

## 📁 Estructura

```
eigenfaces-recognition/
├── README.md                          ← Este archivo
├── eigenfaces_prediction_only.ipynb  ← Notebook para Colab
├── predict.py                        ← Script Python
├── requirements.txt                  ← Dependencias
└── modelos/
    ├── modelo_eigenfaces.pkl         ← Modelo pre-entrenado
    └── nombres_unicos.pkl            ← Lista de personas
```

---

## 🎯 Resultado Típico

**Entrada:**
```
foto.jpg (imagen con un rostro)
```

**Salida:**
```python
{
    'nombre': 'agustina_sebben',
    'confianza': 0.98,  # 98%
    'probabilidades': {
        'agustina_sebben': 0.98,
        'belen_': 0.01,
        'fede_': 0.005,
        ...
    }
}
```

---

## ⚙️ Requisitos

- Python 3.8+
- Google Colab (sin instalaciones)
- O instalar: `pip install -r requirements.txt`

---

## ❓ Preguntas Frecuentes

### "¿Necesito entrenar el modelo?"
**No.** El modelo ya está entrenado y guardado. Solo predicción.

### "¿Qué pasa si no detecta cara?"
La foto está de lado, muy borrosa, o el rostro es muy pequeño. Intenta con mejor iluminación.

### "¿Puedo agregar nuevas personas?"
No con este modelo. Necesitarías reentrenar (ver rama `training`).

### "¿Funciona sin Google Drive?"
Sí, con el script `predict.py` localmente.

---

## 📚 Referencia

- **Paper Original:** Turk & Pentland (1991) "Eigenfaces for Recognition"
- **Técnica:** PCA + SVM
- **Dataset:** 970+ fotos de 14 personas

---

## 📄 Licencia

MIT License - Libre para usar y modificar

---

## 🤝 Contribuciones

Fork → Branch → Commit → Pull Request

---

**Hecho con ❤️ usando Eigenfaces**
