from ultralytics import YOLO
import os
#Configuración
RUTA_BASE=r"C:\Users\Usuario\Documents\Proyectos senati 4to semestre\Capitulo 8 - Baches"
RUTA_YAML=os.path.join(RUTA_BASE,"data","dataset","objetos.yaml")

#Cambiar de directorio
os.chdir(RUTA_BASE) #Va crea una carpeta "runs" en Capitulo 8...

#Cargar modelo de SEGMENTACIÓN (la primera vez se descarga solo)
modelo=YOLO("yolov8n-seg.pt")

#Entrenar el modelo
resultado=modelo.train(
    data=RUTA_YAML,
    epochs=100,
    imgsz=640,
    batch=8,
    name="pavimento_seg",   #Carpeta nueva: no pisa los modelos anteriores
    exist_ok=True,
    device='cpu'
)
print("\nEntrenamiento completado correctamente")
print("Los resultados se guardaron en: ")
print(os.path.join(RUTA_BASE,"runs","segment","pavimento_seg"))
print("Dentro encontraras el archivo 'best.pt' (modelo entrenado)")