"""
Script para detectar cámaras disponibles en el sistema
"""

import cv2
import platform

def detect_cameras(max_tested=10):
    """Detecta todas las cámaras disponibles"""
    print("\n" + "="*60)
    print("DETECCIÓN DE CÁMARAS")
    print("="*60)
    print(f"Sistema operativo: {platform.system()}")
    print(f"Probando índices del 0 al {max_tested-1}...\n")
    
    available_cameras = []
    
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                print(f"✅ Cámara {i}: Disponible")
                print(f"   Resolución: {width}x{height}")
                print(f"   FPS: {fps}")
                print()
            cap.release()
    
    print("="*60)
    
    if available_cameras:
        print(f"\n✅ Se encontraron {len(available_cameras)} cámara(s) disponible(s)")
        print("\nÍndices de cámaras disponibles:", [cam['index'] for cam in available_cameras])
        return available_cameras
    else:
        print("\n❌ No se encontraron cámaras disponibles")
        print("\nPosibles soluciones:")
        print("1. Verifica que tu cámara esté conectada correctamente")
        print("2. Verifica que no esté siendo usada por otra aplicación")
        print("3. En Linux, verifica permisos: sudo usermod -a -G video $USER")
        print("4. Reinicia tu computadora")
        print("5. Prueba con una cámara USB externa")
        return []

def test_camera(camera_index=0):
    """Prueba una cámara específica"""
    print(f"\n{'='*60}")
    print(f"PROBANDO CÁMARA {camera_index}")
    print("="*60)
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara {camera_index}")
        return False
    
    print(f"✅ Cámara {camera_index} abierta correctamente")
    print("\nPresiona 'q' para salir")
    print("Presiona 's' para capturar una imagen de prueba")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error al leer frame de la cámara")
            break
        
        frame_count += 1
        
        # Información en pantalla
        cv2.putText(frame, f"Camara {camera_index} - Frame {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Presiona 'q' para salir | 's' para capturar", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(f'Test Camara {camera_index}', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"\n✅ Prueba completada. Se procesaron {frame_count} frames")
            break
        elif key == ord('s'):
            filename = f'test_capture_cam{camera_index}.jpg'
            cv2.imwrite(filename, frame)
            print(f"✅ Imagen guardada: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def interactive_menu():
    """Menú interactivo para probar cámaras"""
    while True:
        print("\n" + "="*60)
        print("DIAGNÓSTICO DE CÁMARAS")
        print("="*60)
        print("\nOpciones:")
        print("1. Detectar todas las cámaras")
        print("2. Probar cámara específica")
        print("3. Probar cámara 0 (predeterminada)")
        print("4. Probar cámara 1")
        print("5. Probar cámara 2")
        print("6. Salir")
        print()
        
        choice = input("Selecciona una opción (1-6): ").strip()
        
        if choice == '1':
            cameras = detect_cameras()
            if cameras:
                print("\nRecomendación: Usa el primer índice disponible en tus scripts")
        
        elif choice == '2':
            try:
                cam_index = int(input("Ingresa el índice de la cámara a probar: "))
                test_camera(cam_index)
            except ValueError:
                print("❌ Por favor ingresa un número válido")
        
        elif choice == '3':
            test_camera(0)
        
        elif choice == '4':
            test_camera(1)
        
        elif choice == '5':
            test_camera(2)
        
        elif choice == '6':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║           DIAGNÓSTICO DE CÁMARAS - OpenCV                  ║
╚════════════════════════════════════════════════════════════╝

Este script te ayudará a:
- Detectar qué cámaras están disponibles en tu sistema
- Probar cada cámara individualmente
- Identificar el índice correcto para usar en tus proyectos
    """)
    
    interactive_menu()