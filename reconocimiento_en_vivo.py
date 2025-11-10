"""
Script para reconocimiento de gestos en tiempo real
Usa el modelo entrenado para clasificar gestos en vivo
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time

class RealTimeGestureRecognizer:
    def __init__(self, model_path='data/gesture_model_latest.pkl', confidence_threshold=0.7):
        # Cargar modelo
        print("Cargando modelo...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.classes = model_data['classes']
        self.model_name = model_data['model_name']
        
        # Umbral de confianza (0.0 - 1.0)
        self.confidence_threshold = confidence_threshold
        
        print(f"✓ Modelo cargado: {self.model_name}")
        print(f"✓ Clases disponibles: {self.classes}")
        print(f"✓ Umbral de confianza: {confidence_threshold:.1%}")
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Para suavizado de predicciones
        self.prediction_buffer = deque(maxlen=10)
        
        # Métricas
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Colores para cada clase
        self.colors = {
            'A': (255, 100, 100),  # Azul claro
            'B': (100, 255, 100),  # Verde claro
            'C': (100, 100, 255),  # Rojo claro
            'D': (255, 255, 100),  # Cyan
            'E': (255, 100, 255),  # Magenta
            'UNKNOWN': (128, 128, 128),  # Gris
        }
    
    def extract_landmarks(self, image):
        """Extrae landmarks de la mano"""
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
        """Normaliza los landmarks (igual que en entrenamiento)"""
        landmarks = landmarks.reshape(-1, 3)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        middle_finger_mcp = landmarks[9]
        scale = np.linalg.norm(middle_finger_mcp - wrist)
        
        if scale > 0:
            normalized = normalized / scale
        
        return normalized.flatten()
    
    def get_smoothed_prediction(self, prediction):
        """Suaviza las predicciones usando un buffer"""
        self.prediction_buffer.append(prediction)
        
        if len(self.prediction_buffer) < 5:
            return prediction
        
        # Votar por la predicción más común
        predictions_array = np.array(list(self.prediction_buffer))
        unique, counts = np.unique(predictions_array, return_counts=True)
        return unique[np.argmax(counts)]
    
    def calculate_fps(self):
        """Calcula FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        return self.fps
    
    def draw_info_panel(self, image, gesture, confidence, probs=None, is_recognized=True):
        """Dibuja panel de información"""
        h, w = image.shape[:2]
        
        # Panel superior
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Color según reconocimiento
        if gesture == "DESCONOCIDO":
            color = self.colors['UNKNOWN']
            status = "⚠️ Gesto no reconocido"
        elif gesture == "---":
            color = (255, 255, 255)
            status = "Esperando mano..."
        else:
            color = self.colors.get(gesture, (255, 255, 255))
            status = "✓ Reconocido"
        
        # Gesto predicho
        cv2.putText(image, f"Gesto: {gesture}", 
                   (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Estado
        status_color = (100, 255, 100) if is_recognized else (100, 100, 255)
        cv2.putText(image, status, 
                   (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Confianza
        conf_text = f"Confianza: {confidence:.2%}"
        if confidence < self.confidence_threshold and gesture != "---":
            conf_text += f" (min: {self.confidence_threshold:.0%})"
        cv2.putText(image, conf_text, 
                   (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        fps = self.calculate_fps()
        cv2.putText(image, f"FPS: {fps:.1f}", 
                   (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Panel de probabilidades (derecha)
        if probs is not None:
            panel_x = w - 280
            overlay = image.copy()
            cv2.rectangle(overlay, (panel_x, 10), (w - 10, 220), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
            
            cv2.putText(image, "Probabilidades:", 
                       (panel_x + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Línea del umbral
            cv2.line(image, (panel_x + 10, 50), (w - 20, 50), (255, 255, 0), 1)
            cv2.putText(image, f"Umbral: {self.confidence_threshold:.0%}", 
                       (panel_x + 10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            for i, (cls, prob) in enumerate(zip(self.classes, probs)):
                y_pos = 75 + i * 30
                color = self.colors.get(cls, (255, 255, 255))
                
                # Marcar si está por debajo del umbral
                if prob < self.confidence_threshold:
                    color = tuple([c // 2 for c in color])  # Oscurecer
                
                # Texto
                cv2.putText(image, f"{cls}: {prob:.1%}", 
                           (panel_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Barra de progreso
                bar_length = int(220 * prob)
                cv2.rectangle(image, 
                            (panel_x + 10, y_pos + 5), 
                            (panel_x + 10 + bar_length, y_pos + 15), 
                            color, -1)
                
                # Línea del umbral en cada barra
                threshold_x = panel_x + 10 + int(220 * self.confidence_threshold)
                cv2.line(image, (threshold_x, y_pos + 5), (threshold_x, y_pos + 15), (255, 255, 0), 1)
        
        return image
    
    def find_camera(self):
        """Encuentra una cámara disponible"""
        print("\n🔍 Buscando cámaras disponibles...")
        
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Usar DirectShow en Windows
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
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
        """Ejecuta el reconocimiento en tiempo real"""
        cap, cam_index = self.find_camera()
        
        if cap is None:
            print("\n❌ ERROR: No se pudo acceder a ninguna cámara")
            print("\nPosibles soluciones:")
            print("1. Verifica que tu cámara esté conectada")
            print("2. Cierra otras aplicaciones que usen la cámara (Zoom, Skype, etc.)")
            print("3. Ejecuta: python camera_detector.py")
            print("4. En Linux, verifica permisos: sudo chmod 666 /dev/video*")
            return
        
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            pass
        
        # Verificar que puede leer frames
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("\n❌ ERROR: No se puede leer de la cámara")
            cap.release()
            return
        
        print("\n" + "="*60)
        print("RECONOCIMIENTO DE GESTOS EN TIEMPO REAL")
        print("="*60)
        print("\nControles:")
        print("  'q' - Salir")
        print("  'r' - Reiniciar buffer de predicciones")
        print("  'h' - Mostrar/ocultar landmarks")
        print("  '+' - Aumentar umbral de confianza (+5%)")
        print("  '-' - Disminuir umbral de confianza (-5%)")
        print("\n" + "="*60 + "\n")
        
        show_landmarks = True
        frame_errors = 0
        max_frame_errors = 10
        
        while cap.isOpened():
            success, image = cap.read()
            if not success or image is None:
                frame_errors += 1
                if frame_errors > max_frame_errors:
                    print(f"\n❌ ERROR: Demasiados errores de lectura")
                    break
                continue
            
            frame_errors = 0
            
            image = cv2.flip(image, 1)
            landmarks_array, hand_landmarks = self.extract_landmarks(image)
            
            gesture = "---"
            confidence = 0.0
            probs = None
            is_recognized = False
            
            if landmarks_array is not None:
                # Dibujar landmarks
                if show_landmarks and hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                
                # Predecir
                normalized_landmarks = self.normalize_landmarks(landmarks_array)
                normalized_landmarks = normalized_landmarks.reshape(1, -1)
                
                # Obtener predicción
                prediction = self.model.predict(normalized_landmarks)[0]
                
                # Obtener probabilidades si están disponibles
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(normalized_landmarks)[0]
                    confidence = probs[prediction]
                    
                    # Verificar umbral de confianza
                    if confidence >= self.confidence_threshold:
                        is_recognized = True
                        # Suavizar predicción
                        smoothed_prediction = self.get_smoothed_prediction(prediction)
                        gesture = self.label_encoder.inverse_transform([smoothed_prediction])[0]
                    else:
                        # Confianza demasiado baja
                        gesture = "DESCONOCIDO"
                        is_recognized = False
                else:
                    confidence = 1.0
                    is_recognized = True
                    smoothed_prediction = self.get_smoothed_prediction(prediction)
                    gesture = self.label_encoder.inverse_transform([smoothed_prediction])[0]
            
            # Dibujar información
            image = self.draw_info_panel(image, gesture, confidence, probs, is_recognized)
            
            # Instrucciones en la parte inferior
            h, w = image.shape[:2]
            cv2.putText(image, "q=salir | r=reiniciar | h=landmarks | +/- ajustar umbral", 
                       (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Reconocimiento de Gestos LSE', image)
            
            key = cv2.waitKey(5) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.prediction_buffer.clear()
                print("✓ Buffer de predicciones reiniciado")
            elif key == ord('h'):
                show_landmarks = not show_landmarks
                status = "visibles" if show_landmarks else "ocultos"
                print(f"✓ Landmarks {status}")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                print(f"✓ Umbral aumentado a {self.confidence_threshold:.1%}")
            elif key == ord('-') or key == ord('_'):
                self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
                print(f"✓ Umbral reducido a {self.confidence_threshold:.1%}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n✓ Reconocimiento finalizado")

if __name__ == "__main__":
    try:
        recognizer = RealTimeGestureRecognizer()
        recognizer.run()
    except FileNotFoundError:
        print("\n❌ Error: No se encontró el modelo entrenado.")
        print("Por favor, ejecuta primero:")
        print("  1. python capture_gestures.py")
        print("  2. python train_model.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")