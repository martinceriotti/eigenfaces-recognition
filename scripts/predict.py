#!/usr/bin/env python3
"""
Clasificador de Rostros - Eigenfaces
Uso: python predict.py --ruta foto.jpg
     python predict.py --carpeta ./fotos/
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings('ignore')

try:
    import joblib
    from mtcnn import MTCNN
    from sklearn.decomposition import PCA
except ImportError:
    print("❌ Error: Falta instalar dependencias")
    print("\nInstala con:")
    print("  pip install joblib mtcnn opencv-python pillow numpy scikit-learn")
    sys.exit(1)


class EigenfacesClassifier:
    """Clasificador de rostros usando Eigenfaces"""
    
    def __init__(self, ruta_modelo, ruta_nombres):
        """
        Inicializa el clasificador.
        
        Args:
            ruta_modelo: Ruta a modelo_eigenfaces.pkl
            ruta_nombres: Ruta a nombres_unicos.pkl
        """
        try:
            self.modelo = joblib.load(ruta_modelo)
            self.nombres = joblib.load(ruta_nombres)
            self.detector = MTCNN()
            self.img_size = 64
            print(f"✅ Modelo cargado")
            print(f"✅ Personas: {len(self.nombres)}")
        except FileNotFoundError as e:
            print(f"❌ Error: Archivo no encontrado: {e}")
            sys.exit(1)
    
    def identificar_cara(self, ruta_foto):
        """
        Identifica una cara en una foto.
        
        Args:
            ruta_foto: Ruta a la imagen
        
        Returns:
            dict con resultados
        """
        try:
            # Cargar imagen
            img_bgr = cv2.imread(ruta_foto)
            if img_bgr is None:
                # Intentar con PIL (para HEIC)
                img_pil = Image.open(ruta_foto).convert('RGB')
                img_array = np.array(img_pil)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Redimensionar si es muy grande
            max_dim = 1200
            if img_rgb.shape[0] > max_dim or img_rgb.shape[1] > max_dim:
                ratio = max_dim / max(img_rgb.shape[:2])
                new_size = (int(img_rgb.shape[1] * ratio), int(img_rgb.shape[0] * ratio))
                img_rgb = cv2.resize(img_rgb, new_size)
            
            # Detectar cara
            res = self.detector.detect_faces(img_rgb)
            if not res:
                return {'error': 'No se detectó cara en la imagen'}
            
            # Usar detección con mayor confianza
            det = max(res, key=lambda x: x['confidence'])
            if det['confidence'] < 0.95:
                return {'error': f"Confianza baja ({det['confidence']:.1%})"}
            
            x, y, w, h = det['box']
            
            # Expandir ROI
            padding = 0.2
            x = max(0, int(x - w * padding))
            y = max(0, int(y - h * padding))
            w_exp = int(w * (1 + 2 * padding))
            h_exp = int(h * (1 + 2 * padding))
            
            x_fin = min(x + w_exp, img_rgb.shape[1])
            y_fin = min(y + h_exp, img_rgb.shape[0])
            
            face = img_rgb[y:y_fin, x:x_fin]
            
            # Procesar
            face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (self.img_size, self.img_size))
            face_resized = cv2.equalizeHist(face_resized)
            
            # Normalizar
            face_vector = face_resized.flatten() / 255.0
            
            # Predicción
            nombre = self.modelo.predict(face_vector.reshape(1, -1))[0]
            probabilidades = self.modelo.predict_proba(face_vector.reshape(1, -1))[0]
            confianza = np.max(probabilidades)
            
            return {
                'nombre': nombre,
                'confianza': float(confianza),
                'probabilidades': {
                    n: float(p) for n, p in zip(self.nombres, probabilidades)
                }
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def procesar_carpeta(self, ruta_carpeta):
        """
        Procesa todas las imágenes en una carpeta.
        
        Args:
            ruta_carpeta: Ruta a carpeta con imágenes
        
        Returns:
            dict con resultados
        """
        resultados = {}
        
        for archivo in sorted(os.listdir(ruta_carpeta)):
            if not archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                continue
            
            ruta_completa = os.path.join(ruta_carpeta, archivo)
            print(f"Procesando: {archivo}...", end=' ', flush=True)
            
            resultado = self.identificar_cara(ruta_completa)
            
            if 'error' not in resultado:
                resultados[archivo] = resultado
                print(f"✅ {resultado['nombre']} ({resultado['confianza']:.1%})")
            else:
                print(f"❌ {resultado['error']}")
        
        return resultados


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Clasificador de rostros usando Eigenfaces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  python predict.py --ruta foto.jpg
  python predict.py --carpeta ./fotos/
  python predict.py --ruta foto.jpg --modelo modelo.pkl --nombres nombres.pkl
        '''
    )
    
    parser.add_argument('--ruta', type=str, help='Ruta a una imagen')
    parser.add_argument('--carpeta', type=str, help='Ruta a carpeta con imágenes')
    parser.add_argument('--modelo', type=str, default='modelo_eigenfaces.pkl',
                       help='Ruta al modelo (default: modelo_eigenfaces.pkl)')
    parser.add_argument('--nombres', type=str, default='nombres_unicos.pkl',
                       help='Ruta a nombres únicos (default: nombres_unicos.pkl)')
    parser.add_argument('--salida', type=str, help='Guardar resultados en JSON')
    parser.add_argument('--verbose', action='store_true', help='Modo verbose')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.ruta and not args.carpeta:
        parser.print_help()
        sys.exit(1)
    
    # Validar archivos del modelo
    if not os.path.exists(args.modelo):
        print(f"❌ Error: Archivo no encontrado: {args.modelo}")
        sys.exit(1)
    
    if not os.path.exists(args.nombres):
        print(f"❌ Error: Archivo no encontrado: {args.nombres}")
        sys.exit(1)
    
    # Crear clasificador
    print("🎭 Eigenfaces Face Recognition\n")
    clf = EigenfacesClassifier(args.modelo, args.nombres)
    print()
    
    # Procesar
    if args.ruta:
        if not os.path.exists(args.ruta):
            print(f"❌ Error: Archivo no encontrado: {args.ruta}")
            sys.exit(1)
        
        print(f"🔍 Procesando: {args.ruta}\n")
        resultado = clf.identificar_cara(args.ruta)
        
        if 'error' not in resultado:
            print(f"\n✅ RESULTADO:")
            print(f"   Persona: {resultado['nombre'].upper()}")
            print(f"   Confianza: {resultado['confianza']:.1%}")
            
            if args.verbose:
                print(f"\n   Probabilidades:")
                for nombre, prob in sorted(resultado['probabilidades'].items(),
                                         key=lambda x: x[1], reverse=True)[:5]:
                    print(f"      {nombre:<30} {prob:>6.1%}")
            
            # Guardar si se solicita
            if args.salida:
                os.makedirs(os.path.dirname(args.salida) or '.', exist_ok=True)
                with open(args.salida, 'w') as f:
                    json.dump({Path(args.ruta).name: resultado}, f, indent=2)
                print(f"\n✅ Resultados guardados: {args.salida}")
        else:
            print(f"❌ Error: {resultado['error']}")
    
    elif args.carpeta:
        if not os.path.isdir(args.carpeta):
            print(f"❌ Error: Directorio no encontrado: {args.carpeta}")
            sys.exit(1)
        
        print(f"📂 Procesando carpeta: {args.carpeta}\n")
        resultados = clf.procesar_carpeta(args.carpeta)
        
        # Resumen
        print(f"\n📊 RESUMEN:")
        print(f"   Total procesadas: {len(resultados)}")
        
        # Por persona
        personas = {}
        for info in resultados.values():
            nombre = info['nombre']
            personas[nombre] = personas.get(nombre, 0) + 1
        
        print(f"\n   Por persona:")
        for nombre, count in sorted(personas.items(), key=lambda x: x[1], reverse=True):
            print(f"      {nombre:<30} {count:>3} fotos")
        
        # Guardar si se solicita
        if args.salida:
            os.makedirs(os.path.dirname(args.salida) or '.', exist_ok=True)
            with open(args.salida, 'w') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Resultados guardados: {args.salida}")


if __name__ == '__main__':
    main()
