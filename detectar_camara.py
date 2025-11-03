"""
Script avanzado de diagnóstico de cámaras
Incluye verificación del sistema y soluciones específicas
"""

import cv2
import platform
import subprocess
import os
import sys

def print_section(title):
    """Imprime una sección con formato"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_system_info():
    """Muestra información del sistema"""
    print_section("INFORMACIÓN DEL SISTEMA")
    
    print(f"Sistema Operativo: {platform.system()}")
    print(f"Versión: {platform.release()}")
    print(f"Arquitectura: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")

def check_linux_cameras():
    """Diagnóstico específico para Linux"""
    print_section("DIAGNÓSTICO LINUX")
    
    # 1. Verificar dispositivos de video
    print("\n1️⃣ Dispositivos de video detectados:")
    try:
        video_devices = subprocess.run(['ls', '-la', '/dev/video*'], 
                                      capture_output=True, text=True)
        if video_devices.returncode == 0:
            print(video_devices.stdout)
        else:
            print("❌ No se encontraron dispositivos /dev/video*")
            print("\nPosibles causas:")
            print("  • La cámara no está siendo detectada por el sistema")
            print("  • Los controladores no están instalados")
            print("  • La cámara está deshabilitada en BIOS/UEFI")
    except Exception as e:
        print(f"Error al buscar dispositivos: {e}")
    
    # 2. Verificar permisos
    print("\n2️⃣ Verificando permisos de usuario:")
    user = os.environ.get('USER', 'desconocido')
    print(f"Usuario actual: {user}")
    
    try:
        groups = subprocess.run(['groups'], capture_output=True, text=True)
        print(f"Grupos: {groups.stdout.strip()}")
        
        if 'video' in groups.stdout:
            print("✅ Usuario está en el grupo 'video'")
        else:
            print("⚠️  Usuario NO está en el grupo 'video'")
            print("\n💡 SOLUCIÓN:")
            print(f"   sudo usermod -a -G video {user}")
            print("   Luego cierra sesión y vuelve a entrar")
    except Exception as e:
        print(f"Error al verificar grupos: {e}")
    
    # 3. Verificar módulo uvcvideo
    print("\n3️⃣ Verificando módulo de kernel uvcvideo:")
    try:
        lsmod = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'uvcvideo' in lsmod.stdout:
            print("✅ Módulo uvcvideo cargado")
            # Mostrar información del módulo
            modinfo = subprocess.run(['modinfo', 'uvcvideo'], 
                                   capture_output=True, text=True)
            lines = modinfo.stdout.split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Módulo uvcvideo NO cargado")
            print("\n💡 SOLUCIÓN:")
            print("   sudo modprobe uvcvideo")
    except Exception as e:
        print(f"Error al verificar módulo: {e}")
    
    # 4. v4l2-ctl (si está disponible)
    print("\n4️⃣ Información detallada de cámaras (v4l2):")
    try:
        v4l2 = subprocess.run(['v4l2-ctl', '--list-devices'], 
                             capture_output=True, text=True)
        if v4l2.returncode == 0:
            print(v4l2.stdout)
        else:
            print("⚠️  v4l2-ctl no instalado")
            print("\n💡 INSTALAR:")
            print("   sudo apt-get install v4l-utils")
    except FileNotFoundError:
        print("⚠️  v4l2-ctl no instalado")
        print("\n💡 INSTALAR:")
        print("   sudo apt-get install v4l-utils")
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Buscar en dmesg
    print("\n5️⃣ Mensajes del kernel sobre cámaras:")
    try:
        dmesg = subprocess.run(['dmesg', '|', 'grep', '-i', 'video'], 
                              capture_output=True, text=True, shell=True)
        if dmesg.stdout:
            lines = dmesg.stdout.split('\n')[-10:]  # Últimas 10 líneas
            for line in lines:
                if line.strip():
                    print(f"   {line}")
        else:
            print("No se encontraron mensajes relevantes")
    except Exception as e:
        print(f"Error al leer dmesg: {e}")

def check_windows_cameras():
    """Diagnóstico específico para Windows"""
    print_section("DIAGNÓSTICO WINDOWS")
    
    print("\n1️⃣ Verificando dispositivos con PowerShell:")
    try:
        ps_cmd = "Get-PnpDevice -Class Camera,Image | Format-Table -AutoSize"
        result = subprocess.run(['powershell', '-Command', ps_cmd], 
                               capture_output=True, text=True, timeout=10)
        if result.stdout:
            print(result.stdout)
        else:
            print("❌ No se encontraron dispositivos de cámara")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n💡 VERIFICACIONES MANUALES NECESARIAS:")
    print("   1. Configuración → Privacidad → Cámara")
    print("      • 'Acceso a la cámara' debe estar ACTIVADO")
    print("      • 'Aplicaciones de escritorio' debe estar ACTIVADO")
    print()
    print("   2. Administrador de dispositivos (devmgmt.msc)")
    print("      • Busca 'Cámaras' o 'Dispositivos de imagen'")
    print("      • Si aparece con ⚠️ o ❌, actualiza controlador")
    print()
    print("   3. Cierra estas aplicaciones si están abiertas:")
    print("      • Teams, Zoom, Skype, Discord, OBS")

def check_macos_cameras():
    """Diagnóstico específico para macOS"""
    print_section("DIAGNÓSTICO macOS")
    
    print("\n1️⃣ Verificando dispositivos:")
    try:
        system_profiler = subprocess.run(
            ['system_profiler', 'SPCameraDataType'], 
            capture_output=True, text=True, timeout=10
        )
        if system_profiler.stdout:
            print(system_profiler.stdout)
        else:
            print("❌ No se encontraron cámaras")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n💡 VERIFICACIONES NECESARIAS:")
    print("   1. Preferencias del Sistema → Seguridad y Privacidad → Cámara")
    print("      • Asegúrate de que Terminal/Python tengan permiso")
    print()
    print("   2. Prueba reiniciar servicios de cámara:")
    print("      sudo killall VDCAssistant")
    print("      sudo killall AppleCameraAssistant")
    print()
    print("   3. Prueba con Photo Booth o FaceTime")
    print("      Si funciona ahí, es problema de permisos")

def test_opencv_backends():
    """Prueba diferentes backends de OpenCV"""
    print_section("PROBANDO BACKENDS DE OPENCV")
    
    backends = [
        (cv2.CAP_ANY, "CAP_ANY (Auto)"),
        (cv2.CAP_V4L2, "CAP_V4L2 (Linux Video4Linux2)"),
        (cv2.CAP_DSHOW, "CAP_DSHOW (Windows DirectShow)"),
        (cv2.CAP_AVFOUNDATION, "CAP_AVFOUNDATION (macOS)"),
        (cv2.CAP_GSTREAMER, "CAP_GSTREAMER"),
    ]
    
    for backend_id, backend_name in backends:
        try:
            print(f"\nProbando {backend_name}...")
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"  ✅ {backend_name} FUNCIONA!")
                    cap.release()
                    return backend_id, backend_name
                else:
                    print(f"  ⚠️  {backend_name} abrió pero no puede leer frames")
            else:
                print(f"  ❌ {backend_name} no pudo abrir la cámara")
            cap.release()
        except Exception as e:
            print(f"  ❌ Error con {backend_name}: {e}")
    
    return None, None

def scan_camera_indices():
    """Escanea índices de cámara con más detalle"""
    print_section("ESCANEANDO ÍNDICES DE CÁMARA (0-20)")
    
    found_cameras = []
    
    for i in range(20):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend = cap.getBackendName()
                    
                    found_cameras.append({
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': backend
                    })
                    
                    print(f"\n✅ Cámara encontrada en índice {i}")
                    print(f"   Backend: {backend}")
                    print(f"   Resolución: {width}x{height}")
                    print(f"   FPS: {fps}")
                cap.release()
        except Exception as e:
            pass
    
    if not found_cameras:
        print("\n❌ No se encontró ninguna cámara funcional")
    
    return found_cameras

def interactive_test(camera_index):
    """Test interactivo de una cámara"""
    print_section(f"TEST INTERACTIVO - CÁMARA {camera_index}")
    print("\nPresiona 'q' para salir")
    print("Presiona 's' para capturar una imagen de prueba\n")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara {camera_index}")
        return False
    
    print(f"✅ Cámara {camera_index} abierta")
    
    frame_count = 0
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error al leer frame")
            break
        
        frame_count += 1
        
        # Calcular FPS real
        current_time = cv2.getTickCount()
        elapsed = (current_time - start_time) / cv2.getTickFrequency()
        if elapsed > 0:
            real_fps = frame_count / elapsed
        else:
            real_fps = 0
        
        # Añadir información al frame
        cv2.putText(frame, f"Camara {camera_index} - Frames: {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {real_fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "'q'=salir | 's'=capturar", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(f'Test Camara {camera_index}', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"\n✅ Test completado - {frame_count} frames procesados")
            print(f"   FPS promedio: {real_fps:.1f}")
            break
        elif key == ord('s'):
            filename = f'test_camera_{camera_index}.jpg'
            cv2.imwrite(filename, frame)
            print(f"✅ Imagen guardada: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    """Función principal"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║          DIAGNÓSTICO AVANZADO DE CÁMARAS - OpenCV                ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Información del sistema
    check_system_info()
    
    # Diagnóstico según OS
    system = platform.system()
    
    if system == "Linux":
        check_linux_cameras()
    elif system == "Windows":
        check_windows_cameras()
    elif system == "Darwin":  # macOS
        check_macos_cameras()
    
    # Probar backends
    working_backend, backend_name = test_opencv_backends()
    
    if working_backend:
        print(f"\n✅ Backend funcional encontrado: {backend_name}")
    
    # Escanear cámaras
    cameras = scan_camera_indices()
    
    # Resumen final
    print_section("RESUMEN Y RECOMENDACIONES")
    
    if cameras:
        print(f"\n✅ Se encontraron {len(cameras)} cámara(s) funcional(es)")
        print("\nCámaras disponibles:")
        for cam in cameras:
            print(f"  • Índice {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.0f}fps ({cam['backend']})")
        
        print("\n💡 CÓMO USARLAS EN TUS SCRIPTS:")
        print(f"   cap = cv2.VideoCapture({cameras[0]['index']})")
        if working_backend and working_backend != cv2.CAP_ANY:
            print(f"   # O especificando backend:")
            print(f"   cap = cv2.VideoCapture({cameras[0]['index']}, {working_backend})")
        
        # Ofrecer test interactivo
        print("\n" + "="*70)
        response = input("\n¿Quieres probar una cámara interactivamente? (s/n): ").lower()
        if response == 's':
            if len(cameras) == 1:
                interactive_test(cameras[0]['index'])
            else:
                print("\nCámaras disponibles:")
                for i, cam in enumerate(cameras):
                    print(f"  {i+1}. Índice {cam['index']}")
                choice = input("Selecciona (1-{}): ".format(len(cameras)))
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(cameras):
                        interactive_test(cameras[idx]['index'])
                except:
                    print("Selección inválida")
    else:
        print("\n❌ NO SE ENCONTRARON CÁMARAS FUNCIONALES")
        print("\n🔧 PASOS RECOMENDADOS:")
        
        if system == "Linux":
            print("\n1. Verificar que la cámara esté detectada por el sistema:")
            print("   ls -la /dev/video*")
            print("\n2. Dar permisos:")
            print("   sudo chmod 666 /dev/video*")
            print("\n3. Añadir usuario al grupo video:")
            print(f"   sudo usermod -a -G video {os.environ.get('USER', '$USER')}")
            print("   Luego cierra sesión y vuelve a entrar")
            print("\n4. Cargar módulo uvcvideo:")
            print("   sudo modprobe uvcvideo")
            print("\n5. Instalar v4l-utils para más diagnóstico:")
            print("   sudo apt-get install v4l-utils")
            print("   v4l2-ctl --list-devices")
            print("\n6. Verificar en BIOS/UEFI:")
            print("   La cámara podría estar deshabilitada en BIOS")
            
        elif system == "Windows":
            print("\n1. Configuración → Privacidad → Cámara")
            print("   Activar todos los permisos")
            print("\n2. Administrador de dispositivos")
            print("   Buscar y actualizar controlador de cámara")
            print("\n3. Cerrar aplicaciones que usen la cámara")
            print("\n4. Reiniciar el PC")
            
        elif system == "Darwin":
            print("\n1. Preferencias → Seguridad → Privacidad → Cámara")
            print("   Dar permiso a Terminal/Python")
            print("\n2. Reiniciar servicios:")
            print("   sudo killall VDCAssistant")
            print("\n3. Probar con FaceTime primero")
            print("\n4. Reiniciar Mac")
        
        print("\n7. Como última opción:")
        print("   • Usa una webcam USB externa")
        print("   • Trabaja con imágenes estáticas (test_static.py)")

if __name__ == "__main__":
    main()