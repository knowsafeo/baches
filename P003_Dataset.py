#pip install pyyaml
#pip install ultralytics   --YOLO
import os  #Gestionar archivos y carpetas
import yaml #Archivo de configuración para YOLO
import shutil #Gestionar las imagenes y etiquetas
import random #Mezclar las imagenes aleatoriamente

#Configuración
RUTA_BASE=r"C:\Users\Usuario\Documents\Proyectos senati 4to semestre\Capitulo 8 - Baches\data"
RUTA_DATASET=os.path.join(RUTA_BASE,"dataset")
CLASES=['bache','deterioro']   #El orden define los ids: 0=bache, 1=deterioro
DIV_ENTRENAMIENTO=0.8  #80% - Entrenamiento, 20% - validación
EXTENSIONES=('.jpg','.jpeg','.png')


def convertir_etiquetas(ruta_origen,ruta_destino):
    """Deja todas las etiquetas en formato segmentación (clase x1 y1 x2 y2 ...).
    - Líneas con 5 valores (caja de bache) -> se convierten a polígono de 4 esquinas.
    - Líneas con 7 o más valores (polígono) -> se copian igual."""
    lineas_salida=[]
    with open(ruta_origen) as f:
        for linea in f:
            partes=linea.split()
            if len(partes)==5:
                clase=partes[0]
                xc,yc,w,h=map(float,partes[1:])
                xa=max(xc-w/2,0); xb=min(xc+w/2,1)
                ya=max(yc-h/2,0); yb=min(yc+h/2,1)
                lineas_salida.append(
                    f"{clase} {xa:.6f} {ya:.6f} {xb:.6f} {ya:.6f} {xb:.6f} {yb:.6f} {xa:.6f} {yb:.6f}"
                )
            elif len(partes)>=7:
                lineas_salida.append(linea.strip())
    with open(ruta_destino,'w') as f:
        f.write("\n".join(lineas_salida)+("\n" if lineas_salida else ""))


#Borrar el dataset anterior (evita archivos duplicados al volver a ejecutar)
if os.path.exists(RUTA_DATASET):
    shutil.rmtree(RUTA_DATASET)

#Crear la estructura de carpetas
for subset in['train','val']:
    for tipo in['images','labels']:
        os.makedirs(os.path.join(RUTA_DATASET,subset,tipo),exist_ok=True)

print("Carpetas creadas correctamente")

#Dividir automaticamente las imagenes entre TRAIN y VAL
for clase_id,clase in enumerate (CLASES):
    carpeta_clase=os.path.join(RUTA_BASE,clase)

    if not os.path.exists(carpeta_clase):
        print(f"No existe la carpeta de la clase '{clase}': {carpeta_clase}")
        continue

    #Obtener todos los nombres de las imagenes
    archivos=[f for f in os.listdir(carpeta_clase) if f.lower().endswith(EXTENSIONES)]

    random.shuffle(archivos)
    split_idx=int(len(archivos)*DIV_ENTRENAMIENTO)

    print(f"Clase '{clase}': {len(archivos)} imagenes ({split_idx} train / {len(archivos)-split_idx} val)")

    for i, archivo_img in enumerate(archivos):
        conjunto='train' if i<split_idx else 'val'

        nombre_base=os.path.splitext(archivo_img)[0]

        ruta_img_origen=os.path.join(carpeta_clase,archivo_img)
        ruta_txt_origen=os.path.join(carpeta_clase,nombre_base+'.txt')

        ruta_img_destino=os.path.join(RUTA_DATASET,conjunto,'images',archivo_img)
        ruta_txt_destino=os.path.join(RUTA_DATASET,conjunto,'labels',nombre_base+'.txt')

        shutil.copy(ruta_img_origen,ruta_img_destino)
        if os.path.exists(ruta_txt_origen):
            convertir_etiquetas(ruta_txt_origen,ruta_txt_destino)
        else:
            print(f"No se encontró etiqueta para: {archivo_img}")

print("Imagenes y etiquetas divididas en Train/Val correctamente.")

#Crear archivo YAML
yaml_data={
    'train':os.path.abspath(os.path.join(RUTA_DATASET,'train','images')).replace('\\','/'),
    'val':os.path.abspath(os.path.join(RUTA_DATASET,'val','images')).replace('\\','/'),
    'nc':len(CLASES),
    'names':CLASES
}

ruta_yaml=os.path.join(RUTA_DATASET,'objetos.yaml')
with open(ruta_yaml,'w') as f:
    yaml.dump(yaml_data,f,sort_keys=False)

print("\n Archivo YAML creado correctamente: ")
print(ruta_yaml)
print(yaml.dump(yaml_data,sort_keys=False))
