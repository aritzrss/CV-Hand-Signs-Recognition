"""
Script para capturar gestos estáticos de LSE (A, B, C, D, E)
Extrae landmarks de la mano usando MediaPipe y guarda las características
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class GestureCapture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gestos a capturar
        self.gestures = ['A', 'B', 'C', 'D', 'E']
        self.current_gesture_idx = 0
        self.samples_per_gesture = 100
        self.current_samples = 0
        
        # Almacenamiento de datos
        self.data = []
        self.labels = []
        
    def extract_landmarks(self, image):
        """Extrae los landmarks de la mano de una imagen"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extraer coordenadas x, y, z de los 21 landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks), hand_landmarks
        
        return None, None
    
    def normalize_landmarks(self, landmarks):
        """Normaliza los landmarks relativos a la muñeca"""
        landmarks = landmarks.reshape(-1, 3)
        
        # Usar la muñeca (landmark 0) como referencia
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        # Escalar por la distancia entre muñeca y dedo medio
        middle_finger_mcp = landmarks[9]
        scale = np.linalg.norm(middle_finger_mcp - wrist)
        
        if scale > 0:
            normalized = normalized / scale
        
        return normalized.flatten()
    
    def find_camera(self):
        """Encuentra una cámara disponible"""
        print("\n🔍 Buscando cámaras disponibles...")
        
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Usar DirectShow en Windows
                if cap.isOpened():
                    # Intentar leer un frame de prueba
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Verificar que el frame tiene dimensiones válidas
                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            print(f"✅ Cámara encontrada en índice {i}")
                            print(f"   Resolución: {width}x{height}")
                            return cap, i
                    cap.release()
            except Exception as e:
                if cap:
                    cap.release()
                continue
        
        # Si DirectShow falla, intentar sin especificar backend
        print("\n⚠️  DirectShow falló, intentando con backend automático...")
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            print(f"✅ Cámara encontrada en índice {i}")
                            return cap, i
                    cap.release()
            except:
                if cap:
                    cap.release()
                continue
        
        return None, -1
    
    def run(self):
        """Ejecuta la captura de gestos"""
        cap, cam_index = self.find_camera()
        
        if cap is None:
            print("\n❌ ERROR: No se pudo acceder a ninguna cámara")
            print("\nPosibles soluciones:")
            print("1. Verifica que tu cámara esté conectada")
            print("2. Cierra otras aplicaciones que usen la cámara (Zoom, Skype, etc.)")
            print("3. Ejecuta: python camera_detector.py")
            print("4. En Linux, verifica permisos: sudo chmod 666 /dev/video*")
            return
        
        # Configurar resolución con manejo de errores
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            pass  # Si falla, usar resolución por defecto
        
        # Verificar que la cámara puede leer frames
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("\n❌ ERROR: No se puede leer de la cámara")
            print("Posibles soluciones:")
            print("1. Cierra otras aplicaciones que usen la cámara")
            print("2. Verifica permisos de cámara en Windows")
            print("3. Reinicia el ordenador")
            cap.release()
            return
        
        print("=" * 50)
        print("CAPTURA DE GESTOS LSE")
        print("=" * 50)
        print("\nInstrucciones:")
        print("- Presiona ESPACIO para capturar una muestra")
        print("- Presiona 'n' para pasar al siguiente gesto")
        print("- Presiona 'q' para salir")
        print("\n" + "=" * 50)
        
        capturing = False
        frame_errors = 0
        max_frame_errors = 10
        
        while cap.isOpened():
            success, image = cap.read()
            if not success or image is None:
                frame_errors += 1
                if frame_errors > max_frame_errors:
                    print(f"\n❌ ERROR: Demasiados errores de lectura ({frame_errors})")
                    print("La cámara se desconectó o dejó de responder")
                    break
                continue
            
            frame_errors = 0  # Resetear contador si la lectura fue exitosa
            
            image = cv2.flip(image, 1)
            landmarks_array, hand_landmarks = self.extract_landmarks(image)
            
            # Dibujar landmarks si se detecta una mano
            if hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
            
            # Información en pantalla
            current_gesture = self.gestures[self.current_gesture_idx]
            
            # Fondo para el texto
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (600, 150), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            
            # Texto
            cv2.putText(image, f"Gesto actual: {current_gesture}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(image, f"Muestras: {self.current_samples}/{self.samples_per_gesture}", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Progreso: {self.current_gesture_idx + 1}/{len(self.gestures)}", 
                       (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Estado de detección
            if hand_landmarks:
                cv2.putText(image, "MANO DETECTADA", 
                           (20, image.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(image, "SIN MANO", 
                           (20, image.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Captura de Gestos LSE', image)
            
            key = cv2.waitKey(5) & 0xFF
            
            # Capturar muestra
            if key == ord(' ') and landmarks_array is not None:
                normalized_landmarks = self.normalize_landmarks(landmarks_array)
                self.data.append(normalized_landmarks)
                self.labels.append(current_gesture)
                self.current_samples += 1
                
                print(f"✓ Capturada muestra {self.current_samples} de '{current_gesture}'")
                
                # Pasar al siguiente gesto automáticamente
                if self.current_samples >= self.samples_per_gesture:
                    self.current_samples = 0
                    self.current_gesture_idx += 1
                    
                    if self.current_gesture_idx >= len(self.gestures):
                        print("\n¡Captura completada!")
                        break
                    
                    print(f"\n→ Ahora muestra el gesto: {self.gestures[self.current_gesture_idx]}")
            
            # Siguiente gesto manualmente
            elif key == ord('n'):
                self.current_samples = 0
                self.current_gesture_idx = (self.current_gesture_idx + 1) % len(self.gestures)
                print(f"\n→ Cambiando a gesto: {self.gestures[self.current_gesture_idx]}")
            
            # Salir
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Guardar datos
        if len(self.data) > 0:
            self.save_data()
    
    def save_data(self):
        """Guarda los datos capturados"""
        os.makedirs('data', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar como numpy arrays
        np.save(f'data/gestures_data_{timestamp}.npy', np.array(self.data))
        np.save(f'data/gestures_labels_{timestamp}.npy', np.array(self.labels))
        
        # Contar muestras por gesto
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        samples_per_gesture = {str(label): int(count) for label, count in zip(unique_labels, counts)}
        
        # Guardar metadata (convertir a tipos nativos de Python)
        metadata = {
            'timestamp': timestamp,
            'num_samples': int(len(self.data)),
            'gestures': self.gestures,
            'samples_per_gesture': samples_per_gesture
        }
        
        with open(f'data/metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ Datos guardados en data/gestures_data_{timestamp}.npy")
        print(f"✓ Total de muestras: {len(self.data)}")
        print(f"✓ Distribución: {samples_per_gesture}")

if __name__ == "__main__":
    capturer = GestureCapture()
    capturer.run()