import cv2
import os

#Configuración
RUTA_BASE=r"C:\Users\Usuario\Documents\Proyectos senati 4to semestre\Capitulo 8 - Baches"
RUTA_VIDEOS=os.path.join(RUTA_BASE,"videos")
RUTA_DATA=os.path.join(RUTA_BASE,"data")
CLASES=['bache','deterioro']
EXT_VIDEO=('.mp4','.avi','.mov','.mkv')

#Guardar 1 frame cada N frames (los frames seguidos son casi iguales)
FRAME_CADA=30

for clase in CLASES:
    carpeta_videos=os.path.join(RUTA_VIDEOS,clase)
    carpeta_destino=os.path.join(RUTA_DATA,clase)
    os.makedirs(carpeta_videos,exist_ok=True)
    os.makedirs(carpeta_destino,exist_ok=True)

    for archivo in sorted(os.listdir(carpeta_videos)):
        if not archivo.lower().endswith(EXT_VIDEO):
            continue

        ruta_video=os.path.join(carpeta_videos,archivo)
        nombre_video=os.path.splitext(archivo)[0]

        cap=cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            print(f"No se pudo abrir: {archivo}")
            continue

        n_frame=0
        guardados=0
        while True:
            ret,frame=cap.read()
            if not ret:
                break

            if n_frame%FRAME_CADA==0:
                ruta_guardado=os.path.join(carpeta_destino,f"{nombre_video}_f{n_frame:05d}.jpg")
                #No sobreescribe si ya existe (así no pierdes etiquetas al repetir)
                if not os.path.exists(ruta_guardado):
                    cv2.imwrite(ruta_guardado,frame)
                    guardados+=1
            n_frame+=1

        cap.release()
        print(f"{clase} | {archivo}: {guardados} frames guardados")

print("\nExtracción de frames finalizada")