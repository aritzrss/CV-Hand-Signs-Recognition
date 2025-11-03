"""
Script para probar el modelo con imágenes estáticas
Útil para debugging y pruebas sin webcam
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys
import os

class StaticGestureTest:
    def __init__(self, model_path='data/gesture_model_latest.pkl'):
        # Cargar modelo
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.classes = model_data['classes']
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image):
        """Extrae landmarks de la imagen"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks), hand_landmarks
        
        return None, None
    
    def normalize_landmarks(self, landmarks):
        """Normaliza landmarks"""
        landmarks = landmarks.reshape(-1, 3)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        middle_finger_mcp = landmarks[9]
        scale = np.linalg.norm(middle_finger_mcp - wrist)
        
        if scale > 0:
            normalized = normalized / scale
        
        return normalized.flatten()
    
    def predict_image(self, image_path):
        """Predice el gesto en una imagen"""
        if not os.path.exists(image_path):
            print(f"❌ Error: No se encuentra el archivo '{image_path}'")
            return
        
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: No se pudo cargar la imagen '{image_path}'")
            return
        
        # Extraer landmarks
        landmarks_array, hand_landmarks = self.extract_landmarks(image)
        
        if landmarks_array is None:
            print("❌ No se detectó ninguna mano en la imagen")
            return
        
        # Dibujar landmarks
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )
        
        # Predecir
        normalized_landmarks = self.normalize_landmarks(landmarks_array)
        normalized_landmarks = normalized_landmarks.reshape(1, -1)
        
        prediction = self.model.predict(normalized_landmarks)[0]
        gesture = self.label_encoder.inverse_transform([prediction])[0]
        
        # Obtener probabilidades
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(normalized_landmarks)[0]
            confidence = probs[prediction]
        else:
            probs = None
            confidence = 1.0
        
        # Mostrar resultados
        print(f"\n{'='*50}")
        print(f"📸 Imagen: {image_path}")
        print(f"{'='*50}")
        print(f"✅ Gesto detectado: {gesture}")
        print(f"🎯 Confianza: {confidence:.2%}")
        
        if probs is not None:
            print(f"\n📊 Probabilidades:")
            for cls, prob in zip(self.classes, probs):
                bar = '█' * int(prob * 30)
                print(f"   {cls}: {prob:6.2%} {bar}")
        
        # Añadir texto a la imagen
        h, w = image.shape[:2]
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 100), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        cv2.putText(image, f"Gesto: {gesture}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(image, f"Confianza: {confidence:.2%}", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mostrar imagen
        cv2.imshow('Predicción', image)
        print(f"\n💡 Presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def test_from_webcam(self):
        """Captura una imagen de la webcam y la analiza"""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*50)
        print("CAPTURA DE IMAGEN PARA PRUEBA")
        print("="*50)
        print("Presiona ESPACIO para capturar una foto")
        print("Presiona 'q' para salir")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.flip(image, 1)
            
            # Mostrar instrucciones
            cv2.putText(image, "Presiona ESPACIO para capturar", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Captura desde Webcam', image)
            
            key = cv2.waitKey(5) & 0xFF
            
            if key == ord(' '):
                # Guardar imagen temporal
                temp_path = 'temp_test_image.jpg'
                cv2.imwrite(temp_path, image)
                cap.release()
                cv2.destroyAllWindows()
                
                # Analizar la imagen
                self.predict_image(temp_path)
                
                # Eliminar temporal
                os.remove(temp_path)
                break
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("\n" + "="*50)
    print("PRUEBA DE MODELO CON IMÁGENES ESTÁTICAS")
    print("="*50)
    
    try:
        tester = StaticGestureTest()
        
        if len(sys.argv) > 1:
            # Probar con imagen específica
            image_path = sys.argv[1]
            tester.predict_image(image_path)
        else:
            # Capturar desde webcam
            tester.test_from_webcam()
    
    except FileNotFoundError:
        print("\n❌ Error: No se encontró el modelo entrenado.")
        print("Por favor, ejecuta primero:")
        print("  1. python capture_gestures.py")
        print("  2. python train_model.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    print("\nModos de uso:")
    print("  1. python test_static.py                    # Captura desde webcam")
    print("  2. python test_static.py imagen.jpg         # Analiza imagen específica")
    print()
    
    main()