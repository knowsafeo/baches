import cv2
import os
import numpy as np

RUTA = r"C:\Users\Usuario\Documents\Proyectos senati 4to semestre\Capitulo 8 - Baches\data"
CLASES = {
    0: "bache",
    1: "deterioro"
}

# Color de cada clase en el etiquetador (BGR)
COLORES = {
    0: (0, 255, 0),     # bache -> verde
    1: (255, 0, 255)    # deterioro -> magenta
}

# Tamaño máximo de la ventana (las fotos de celular son muy grandes)
MAX_ANCHO = 1000
MAX_ALTO = 700

dibujando = False
x1 = y1 = x2 = y2 = 0
mx = my = 0
# Cada objeto es ("caja", (xa, ya, xb, yb), clase) o ("poly", [(x, y), ...], clase)
objetos = []
puntos = []          # puntos del polígono que se está dibujando
clase_actual = 0


def cerrar_poligono():
    if len(puntos) >= 3:
        objetos.append(("poly", list(puntos), clase_actual))
        puntos.clear()
        print("Polígono cerrado.")
    else:
        print("Un polígono necesita al menos 3 puntos.")


def mouse(event, x, y, flags, param):
    global dibujando, x1, y1, x2, y2, mx, my

    mx, my = x, y

    # Clase 0 (bache): caja arrastrando el mouse
    if clase_actual == 0:

        if event == cv2.EVENT_LBUTTONDOWN:
            dibujando = True
            x1, y1 = x, y
            x2, y2 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and dibujando:
            x2, y2 = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            dibujando = False
            x2, y2 = x, y

            xa, xb = sorted([x1, x2])
            ya, yb = sorted([y1, y2])

            if xb - xa > 5 and yb - ya > 5:
                objetos.append(("caja", (xa, ya, xb, yb), 0))

    # Clase 1 (deterioro): polígono punto por punto
    else:

        if event == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            cerrar_poligono()


def etiquetar(ruta_imagen, clase_carpeta):
    global clase_actual, dibujando

    img = cv2.imread(ruta_imagen)

    if img is None:
        return

    nombre_imagen = os.path.basename(ruta_imagen)
    nombre_txt = os.path.splitext(ruta_imagen)[0] + ".txt"

    # Si ya fue etiquetada, se salta
    if os.path.exists(nombre_txt):
        print(f"Ya etiquetada, se salta: {nombre_imagen}")
        return

    # Reducir la imagen solo para mostrarla/etiquetarla.
    # Las coordenadas YOLO son normalizadas (0-1), así que sirven igual
    # para la imagen original.
    h0, w0 = img.shape[:2]
    escala = min(1.0, MAX_ANCHO / w0, MAX_ALTO / h0)
    if escala < 1.0:
        img = cv2.resize(img, (int(w0 * escala), int(h0 * escala)))

    h, w = img.shape[:2]
    objetos.clear()
    puntos.clear()
    dibujando = False

    # La clase inicial es la de la carpeta donde está la imagen
    for id_clase, nombre in CLASES.items():
        if nombre == clase_carpeta:
            clase_actual = id_clase

    print(f"\nImagen: {nombre_imagen}")
    print(f"Carpeta de clase: {clase_carpeta}")
    print("'0' = bache (caja arrastrando el mouse)")
    print("'1' = deterioro (clic izquierdo por cada punto, clic derecho o 'c' para cerrar)")
    print("Presiona 'u' para deshacer (último punto o último objeto).")
    print("Presiona 's' para guardar etiquetas o 'q' para saltar.\n")

    cv2.namedWindow("Etiquetador")
    cv2.setMouseCallback("Etiquetador", mouse)

    while True:

        copia = img.copy()
        relleno = img.copy()

        # Relleno semitransparente de los polígonos terminados
        for tipo, datos, clase in objetos:
            if tipo == "poly":
                cv2.fillPoly(relleno, [np.array(datos, dtype=np.int32)], COLORES[clase])
        copia = cv2.addWeighted(relleno, 0.3, copia, 0.7, 0)

        # Dibujar objetos guardados
        for tipo, datos, clase in objetos:

            if tipo == "caja":
                xa, ya, xb, yb = datos
                cv2.rectangle(copia, (xa, ya), (xb, yb), COLORES[clase], 2)
                cv2.putText(copia, CLASES[clase], (xa, max(ya - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORES[clase], 2)

            else:
                pts = np.array(datos, dtype=np.int32)
                cv2.polylines(copia, [pts], True, COLORES[clase], 2)
                cv2.putText(copia, CLASES[clase], (datos[0][0], max(datos[0][1] - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORES[clase], 2)

            if tipo == "caja":
                xa, ya, xb, yb = datos
                cv2.rectangle(copia, (xa, ya), (xb, yb), COLORES[clase], 2)
                cv2.putText(copia, CLASES[clase], (xa, max(ya - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORES[clase], 2)

            else:
                pts = __import__("numpy").array(datos, dtype="int32")
                cv2.polylines(copia, [pts], True, COLORES[clase], 2)
                cv2.putText(copia, CLASES[clase], (datos[0][0], max(datos[0][1] - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORES[clase], 2)

        # Dibujar caja en tiempo real (bache)
        if dibujando:
            cv2.rectangle(copia, (x1, y1), (x2, y2), COLORES[0], 2)

        # Dibujar polígono en curso (deterioro)
        if puntos:
            for p in puntos:
                cv2.circle(copia, p, 3, COLORES[1], -1)
            for i in range(len(puntos) - 1):
                cv2.line(copia, puntos[i], puntos[i + 1], COLORES[1], 2)
            # línea guía hasta el mouse
            cv2.line(copia, puntos[-1], (mx, my), COLORES[1], 1)

        # Mostrar clase y cantidad de objetos
        cv2.putText(
            copia,
            f"Clase: {CLASES[clase_actual]} | Objetos: {len(objetos)} | Puntos: {len(puntos)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Etiquetador", copia)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("0"):
            clase_actual = 0
            puntos.clear()
            print("Clase seleccionada: bache")

        elif tecla == ord("1"):
            clase_actual = 1
            dibujando = False
            print("Clase seleccionada: deterioro")

        elif tecla in (ord("c"), 13):
            cerrar_poligono()

        elif tecla == ord("u"):
            if puntos:
                puntos.pop()
            elif objetos:
                objetos.pop()
                print("Último objeto eliminado.")

        elif tecla == ord("q"):
            print("Imagen saltada.")
            break

        elif tecla == ord("s"):

            if puntos:
                print("Hay un polígono sin cerrar: ciérralo con clic derecho o 'c' antes de guardar.")
                continue

            with open(nombre_txt, "w") as f:

                for tipo, datos, clase in objetos:

                    if tipo == "caja":
                        xa, ya, xb, yb = datos
                        xc = ((xa + xb) / 2) / w
                        yc = ((ya + yb) / 2) / h
                        ancho = (xb - xa) / w
                        alto = (yb - ya) / h
                        f.write(f"{clase} {xc:.6f} {yc:.6f} {ancho:.6f} {alto:.6f}\n")

                    else:
                        coords = " ".join(
                            f"{min(max(px / w, 0), 1):.6f} {min(max(py / h, 0), 1):.6f}"
                            for px, py in datos
                        )
                        f.write(f"{clase} {coords}\n")

            print(f"Etiquetas guardadas en: {nombre_txt}")
            break

    cv2.destroyAllWindows()


# Recorrer las carpetas
for carpeta in CLASES.values():

    ruta_carpeta = os.path.join(RUTA, carpeta)

    if not os.path.exists(ruta_carpeta):
        print(f"No existe la carpeta: {ruta_carpeta}")
        continue

    for archivo in sorted(os.listdir(ruta_carpeta)):

        if archivo.lower().endswith((".jpg", ".jpeg", ".png")):

            ruta_img = os.path.join(
                ruta_carpeta,
                archivo
            )

            etiquetar(
                ruta_img,
                carpeta
            )


print("\nProceso de etiquetado finalizado correctamente.")
