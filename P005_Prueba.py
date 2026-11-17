from ultralytics import YOLO
import cv2
import numpy as np
import os

#Configuración
RUTA_BASE=r"C:\Users\Usuario\Documents\Proyectos senati 4to semestre\Capitulo 8 - Baches"
RUTA_MODELO=os.path.join(RUTA_BASE,"runs","segment","pavimento_seg","weights","best.pt")

#Fuente: 0 = camara en vivo, o la ruta de un video, por ejemplo:
#FUENTE=r"D:\Andy - Python1\Capitulo 8 - Baches\videos\prueba.mp4"
FUENTE=0

#Guardar el video procesado (solo si la fuente es un archivo de video)
GUARDAR_VIDEO=True
RUTA_SALIDA=os.path.join(RUTA_BASE,"resultados")

CONFIANZA=0.3
NOMBRE_DETERIORO="deterioro"   #Debe coincidir con el nombre de la clase en P003

#Colores por tamaño (BGR): azul = pequeño, amarillo = mediano, rojo = grande
AZUL=(255,0,0)
AMARILLO=(0,255,255)
ROJO=(0,0,255)

#Umbrales para BACHES (porcentaje del area del frame que ocupa el cuadro)
BACHE_MEDIO=0.02     #desde 2% del frame -> amarillo
BACHE_GRANDE=0.08    #desde 8% del frame -> rojo

#Umbrales para DETERIORO (porcentaje del area del frame que ocupa el contorno)
DET_MEDIO=0.05       #desde 5% del frame -> amarillo
DET_GRANDE=0.15      #desde 15% del frame -> rojo

#Transparencia del relleno del deterioro (0 = invisible, 1 = solido)
TRANSPARENCIA=0.45

#Ancho maximo de la ventana en pantalla
MAX_ANCHO_VENTANA=1000


def nivel_por_tamano(ratio,umbral_medio,umbral_grande):
    if ratio>=umbral_grande:
        return "GRANDE",ROJO
    elif ratio>=umbral_medio:
        return "MEDIO",AMARILLO
    else:
        return "PEQUENO",AZUL


if not os.path.exists(RUTA_MODELO):
    print("No se encontro el archivo best.pt.")
    exit()

print("Cargando el modelo entrenado...")
modelo=YOLO(RUTA_MODELO)

#Iniciar Camara / Video
cap=cv2.VideoCapture(FUENTE)
if not cap.isOpened():
    print("No se pudo acceder a la cámara o al video")
    exit()

es_video=isinstance(FUENTE,str)
writer=None

if es_video and GUARDAR_VIDEO:
    os.makedirs(RUTA_SALIDA,exist_ok=True)
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:
        fps=30
    ancho=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nombre_salida=os.path.splitext(os.path.basename(FUENTE))[0]+"_resultado.mp4"
    ruta_salida=os.path.join(RUTA_SALIDA,nombre_salida)
    writer=cv2.VideoWriter(ruta_salida,cv2.VideoWriter_fourcc(*'mp4v'),fps,(ancho,alto))

print("Detección y seguimiento iniciado")
print("Azul = pequeño | Amarillo = mediano | Rojo = grande")
print("Presiona 'q' para salir")

while True:
    ret,frame=cap.read()
    if not ret:
        print("Fin del video o error al capturar el frame")
        break

    alto_f,ancho_f=frame.shape[:2]
    area_frame=ancho_f*alto_f
    grosor=max(2,ancho_f//400)
    escala_texto=max(0.5,ancho_f/1400)

    #Detectamos y SEGUIMOS (cada objeto recibe un ID)
    resultado=modelo.track(source=frame,persist=True,imgsz=640,conf=CONFIANZA,device='cpu',verbose=False)
    r=resultado[0]

    conteo={"PEQUENO":0,"MEDIO":0,"GRANDE":0}
    dibujos=[]   #(tipo, datos, color, etiqueta, x, y)

    for i,caja in enumerate(r.boxes):
        x1,y1,x2,y2=map(int,caja.xyxy[0])
        clase=int(caja.cls[0])
        conf=float(caja.conf[0])
        id_obj=int(caja.id[0]) if caja.id is not None else None

        etiqueta=f"{modelo.names[clase]} {conf:.2f}"
        if id_obj is not None:
            etiqueta=f"#{id_obj} "+etiqueta

        es_deterioro=(modelo.names[clase]==NOMBRE_DETERIORO and r.masks is not None)

        if es_deterioro:
            #Contorno real del deterioro (coordenadas en pixeles del frame)
            poligono=r.masks.xy[i].astype(np.int32)
            if len(poligono)<3:
                continue
            ratio=cv2.contourArea(poligono)/area_frame
            nivel,color=nivel_por_tamano(ratio,DET_MEDIO,DET_GRANDE)
            dibujos.append(("poly",poligono,color,etiqueta,x1,y1))
        else:
            #Bache: cuadro, color segun el area del cuadro
            ratio=((x2-x1)*(y2-y1))/area_frame
            nivel,color=nivel_por_tamano(ratio,BACHE_MEDIO,BACHE_GRANDE)
            dibujos.append(("caja",(x1,y1,x2,y2),color,etiqueta,x1,y1))

        conteo[nivel]+=1

    #1) Relleno semitransparente de los contornos de deterioro
    relleno=frame.copy()
    for tipo,datos,color,etiqueta,tx,ty in dibujos:
        if tipo=="poly":
            cv2.fillPoly(relleno,[datos],color)
    frame=cv2.addWeighted(relleno,TRANSPARENCIA,frame,1-TRANSPARENCIA,0)

    #2) Bordes, cuadros y textos encima
    for tipo,datos,color,etiqueta,tx,ty in dibujos:
        if tipo=="poly":
            cv2.polylines(frame,[datos],True,color,grosor)
        else:
            x1,y1,x2,y2=datos
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,grosor)

        #Fondo para que el texto se lea bien
        (tw,th),_=cv2.getTextSize(etiqueta,cv2.FONT_HERSHEY_SIMPLEX,escala_texto,1)
        y_texto=max(ty,th+8)
        cv2.rectangle(frame,(tx,y_texto-th-8),(tx+tw+6,y_texto),color,-1)
        cv2.putText(frame,etiqueta,(tx+3,y_texto-4),cv2.FONT_HERSHEY_SIMPLEX,
                    escala_texto,(0,0,0),1)

    #Leyenda con conteo por nivel
    y_leyenda=int(35*escala_texto*1.6)
    paso=int(35*escala_texto*1.6)
    for nombre,color in (("GRANDE",ROJO),("MEDIO",AMARILLO),("PEQUENO",AZUL)):
        cv2.putText(frame,f"{nombre}: {conteo[nombre]}",(10,y_leyenda),
                    cv2.FONT_HERSHEY_SIMPLEX,escala_texto*1.1,color,grosor)
        y_leyenda+=paso

    if writer is not None:
        writer.write(frame)

    #Reducir solo para mostrar en pantalla
    if ancho_f>MAX_ANCHO_VENTANA:
        factor=MAX_ANCHO_VENTANA/ancho_f
        mostrar=cv2.resize(frame,(MAX_ANCHO_VENTANA,int(alto_f*factor)))
    else:
        mostrar=frame

    cv2.imshow("Yolo - Baches y deterioro",mostrar)

    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cap.release()
if writer is not None:
    writer.release()
    print(f"Video guardado en: {ruta_salida}")
cv2.destroyAllWindows()
print("Detección finalizada")