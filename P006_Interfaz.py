# -*- coding: utf-8 -*-
"""
P006_Interfaz.py — TacnaVial: interfaz para detección de baches y deterioro
──────────────────────────────────────────────────────────────────
Usa 2 modelos YOLOv8 (baches = cajas, deterioro = polígonos) con una
interfaz hecha en CustomTkinter que funciona como un REPRODUCTOR DE VIDEO:
    - Reproducir / Pausar, retroceder y adelantar 5 segundos
    - Barra para moverse a cualquier parte del video
    - Velocidad 0.5x, 1x y 2x
    - Atajos de teclado: Espacio = play/pausa, ← y → = mover 5 segundos

Requisitos:
    pip install customtkinter pillow ultralytics opencv-python numpy pandas
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
import os
import threading
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE RUTAS
# ═══════════════════════════════════════════════════════════════════════════════
RUTA_BASE = r"D:\Proyectos senati 4to semestre\Capitulo 8 - Baches"
# Dos modelos entrenados en Google Colab:
#   - best_baches.pt    → detección de objetos (cajas), clase "bache"
#   - best_deterioro.pt → segmentación (polígonos), clase "deterioro"
RUTA_MODELO_BACHES = os.path.join(RUTA_BASE, "best_baches.pt")
RUTA_MODELO_DETERIORO = os.path.join(RUTA_BASE, "best_deterioro.pt")
RUTA_SALIDA = os.path.join(RUTA_BASE, "resultados")
RUTA_CAPTURAS = os.path.join(RUTA_BASE, "capturas")
RUTA_CSV = os.path.join(RUTA_SALIDA, "detecciones.csv")   # Registro para P007

# ═══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS FIJOS DE DETECCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
# Confianza mínima: 40 % para los 2 modelos (punto de equilibrio entre
# falsos positivos y baches perdidos)
CONFIANZA = 0.40
CONFIANZA_DET = 0.40

# Los videos del celular son 4K: cada frame se achica a este ancho antes de
# usarlo para que la interfaz no se congele
ANCHO_MAX = 1280
IMGSZ = 640          # Tamaño de imagen para el modelo de baches
IMGSZ_DET = 480      # Deterioro con imagen más chica: va más rápido

# El modelo de deterioro solo se pasa 1 de cada 3 frames
CADA_DETERIORO = 3
# Pero el morado de antes solo se reusa si es RECIENTE (máximo 6 frames de
# video ≈ 0.2 s). En una PC lenta se saltan frames y, si se reusa uno viejo,
# el morado se queda "pegado" mientras la cámara avanza.
MAX_EDAD_DET = 6

# ZONA DE MEDICIÓN: el tamaño solo se mide cuando el centro del objeto está
# entre el 50 % y el 75 % de la altura (todos a la misma distancia)
ZONA_INICIO = 0.50
ZONA_FIN = 0.75
MOSTRAR_ZONA = False   # True = dibuja las 2 líneas blancas de la zona

# Si más de la mitad de un deterioro cae dentro de un bache, se queda el bache
SOLAPE_MAXIMO = 0.50

# Segundos que se mueve el video con los botones ⏪ ⏩ y las flechas
SALTO_SEGUNDOS = 5

# ═══════════════════════════════════════════════════════════════════════════════
#  PALETA DE COLORES DE LA INTERFAZ (tema oscuro)
# ═══════════════════════════════════════════════════════════════════════════════
C_FONDO     = "#15171C"   # Fondo general
C_PANEL     = "#1D2027"   # Panel lateral
C_TARJETA   = "#242833"   # Tarjetas y botones secundarios
C_TARJETA_H = "#2E3340"   # Hover de botones secundarios
C_BORDE     = "#2F3440"   # Bordes y separadores
C_TEXTO     = "#E8EAED"   # Texto principal
C_TENUE     = "#9AA3B2"   # Texto secundario
C_VIDEO     = "#0B0C0F"   # Fondo del área de video
C_ACENTO    = "#3B82F6"   # Azul principal
C_ACENTO_H  = "#2F6FD8"
C_VERDE     = "#22C55E"
C_ROJO      = "#EF4444"
C_ROJO_H    = "#D63B3B"
C_AMARILLO  = "#EAB308"
C_AZUL      = "#60A5FA"
C_MORADO    = "#AA46BE"   # Mismo morado que el deterioro en el video

# ═══════════════════════════════════════════════════════════════════════════════
#  COLORES DE DETECCIÓN EN EL VIDEO (BGR para OpenCV)
# ═══════════════════════════════════════════════════════════════════════════════
CV_AZUL      = (237, 149, 100)   # Pequeño
CV_AMARILLO  = (50, 190, 245)    # Mediano
CV_ROJO      = (80, 80, 220)     # Grande
CV_DETERIORO = (190, 70, 170)    # Morado: color fijo del deterioro
CV_ZONA      = (255, 255, 255)   # Líneas de la zona de medición

# ═══════════════════════════════════════════════════════════════════════════════
#  UMBRALES DE TAMAÑO
# ═══════════════════════════════════════════════════════════════════════════════
# Bache: porcentaje del frame que ocupa la caja
BACHE_MEDIO  = 0.02   # ≥ 2 % → mediano
BACHE_GRANDE = 0.08   # ≥ 8 % → grande

# Deterioro: porcentaje del frame que ocupa el contorno (solo para el CSV)
DET_MEDIO  = 0.05
DET_GRANDE = 0.15

# Nombre que se escribe sobre cada bache según su tamaño
NOMBRE_BACHE = {"PEQUENO": "bache_pequeno", "MEDIO": "bache_mediano",
                "GRANDE": "bache_grande"}

# Tamaño en palabras (para el registro)
TAMANO_TEXTO = {"PEQUENO": "pequeño", "MEDIO": "mediano", "GRANDE": "grande"}

# Orden de tamaños (para saber si uno es "más grande" que otro)
ORDEN_NIVEL = {"PEQUENO": 0, "MEDIO": 1, "GRANDE": 2}

# Transparencia del relleno de los polígonos
TRANSPARENCIA = 0.45


# ═══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class DetectorApp(ctk.CTk):
    """Ventana principal del detector de baches y deterioros."""

    def __init__(self):
        super().__init__()

        # ── Configuración de ventana ─────────────────────────────────
        ctk.set_appearance_mode("dark")
        self.title("TacnaVial — Detección de baches y deterioro en vías de Tacna")
        self.geometry("1280x760")
        self.minsize(1000, 640)
        # Abrir maximizada: así entra bien en laptops con pantalla pequeña
        self.after(0, self._maximizar)
        self.configure(fg_color=C_FONDO)

        # ── Estado interno ───────────────────────────────────────────
        self.modelo = None              # Modelo de baches (cajas)
        self.modelo_det = None          # Modelo de deterioro (polígonos)
        self.dispositivo = "cpu"        # "cpu" o 0 (tarjeta NVIDIA)
        self.det_previo = []            # Último resultado de deterioro
        self.det_frame = -999           # En qué frame se calculó ese resultado
        self.cuenta_proc = 0            # Cuántos frames se procesaron
        self.det_recientes = []         # Deterioros vistos hace poco
        self.det_contador = 0           # Para numerar cada deterioro
        self.cap = None
        self.hilo = None
        self.corriendo = False
        self.pausado = False
        self.fuente_es_video = False
        self.ruta_video_actual = None

        # Reproductor
        self.fps_video = 30.0           # FPS del video abierto
        self.velocidad = 1.0            # 0.5, 1 o 2
        self.pedido_salto = None        # Frame al que se quiere ir (o None)
        self.fin_video = False          # True cuando el video llegó al final
        self.frame_max = 0              # Hasta qué frame ya se analizó
        self.arrastrando = False        # True mientras se mueve la barra
        self.resumen_mostrado = False

        self.frame_actual = None        # Último frame procesado (BGR)
        self._ultimo_mostrado = None    # Último frame ya dibujado en la UI
        self.foto_tk = None             # PhotoImage para el label

        # Guardado del video procesado
        self.ruta_guardar = None        # Ruta elegida por el usuario
        self.writer = None
        self.frames_escritos = 0
        self.parar_grabacion = False    # Pedido para cerrar la grabación

        self.frame_idx = 0
        self.total_frames = 0

        self.ids_vistos = set()
        self.conteo_global = self._conteo_vacio()
        self.lock = threading.Lock()

        # Registro de detecciones (se guarda en CSV para el análisis)
        self.sector = "Sin sector"
        # Semáforo de la vía (para el Arduino): 0 bueno, 1 regular, 2 malo
        self.estado_via = 0
        self.t_estado = 0.0
        self.registros = []
        # Las claves son (tipo, id) porque cada modelo numera sus IDs aparte
        self.nivel_id = {}           # Tamaño más grande que alcanzó cada ID
        self.fila_id = {}            # En qué fila del CSV está cada ID

        # ── Construir interfaz ───────────────────────────────────────
        self._crear_sidebar()
        self._crear_area_principal()
        self._mostrar_bienvenida()

        # ── Atajos de teclado ────────────────────────────────────────
        self.bind("<space>", lambda e: self._tecla("play"))
        self.bind("<Left>", lambda e: self._tecla("atras"))
        self.bind("<Right>", lambda e: self._tecla("adelante"))

        # ── Cargar modelos con un pequeño retraso para que la UI se dibuje ──
        self.after(600, self._cargar_modelo)

        # ── Bucle de actualización de la UI ──────────────────────────
        self._tick_ui()

        # ── Limpieza al cerrar ───────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_cerrar)

    # ───────────────────────────────────────────────────────────────────
    #  Utilidades
    # ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _f(tamano, negrita=False):
        """Fuente de la interfaz (Segoe UI es la letra de Windows)."""
        return ctk.CTkFont(family="Segoe UI", size=tamano,
                           weight="bold" if negrita else "normal")

    def _maximizar(self):
        try:
            self.state("zoomed")      # Solo existe en Windows
        except Exception:
            pass

    def _desde_hilo(self, funcion, *args):
        """Pide a la ventana que ejecute una función (se usa desde el hilo
        de detección). Si la ventana ya se cerró, no hace nada."""
        try:
            self.after(0, funcion, *args)
        except Exception:
            pass

    @staticmethod
    def _conteo_vacio():
        return {
            "bache":     {"PEQUENO": 0, "MEDIO": 0, "GRANDE": 0},
            "deterioro": {"PEQUENO": 0, "MEDIO": 0, "GRANDE": 0},
        }

    @staticmethod
    def _formato_tiempo(segundos):
        """Convierte segundos a texto mm:ss (ej: 75 → 01:15)."""
        segundos = max(0, int(segundos))
        return f"{segundos // 60:02d}:{segundos % 60:02d}"

    def _id_deterioro(self, x1, y1, x2, y2):
        """Da un número a cada deterioro. Si se superpone con uno visto hace
        poco, es el mismo y se reusa su número; si no, es nuevo."""
        mejor, mejor_iou = None, 0.15
        for d in self.det_recientes:
            if self.frame_idx - d["frame"] > 20:
                continue
            bx1, by1, bx2, by2 = d["caja"]
            ancho = min(x2, bx2) - max(x1, bx1)
            alto = min(y2, by2) - max(y1, by1)
            if ancho <= 0 or alto <= 0:
                continue
            inter = ancho * alto
            union = ((x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - inter)
            iou = inter / max(1, union)
            if iou > mejor_iou:
                mejor, mejor_iou = d, iou
        if mejor is None:
            self.det_contador += 1
            mejor = {"id": self.det_contador}
            self.det_recientes.append(mejor)
        mejor["caja"] = (x1, y1, x2, y2)
        mejor["frame"] = self.frame_idx
        # Olvidar los que no se ven hace rato
        self.det_recientes = [d for d in self.det_recientes
                              if self.frame_idx - d["frame"] <= 20]
        return mejor["id"]

    @staticmethod
    def _tapado_por_bache(x1, y1, x2, y2, objetos):
        """True si el deterioro está casi todo dentro de la caja de un bache."""
        area_det = max(1, (x2 - x1) * (y2 - y1))
        for o in objetos:
            if o[0] != "bache":
                continue
            bx1, by1, bx2, by2 = o[3], o[4], o[5], o[6]
            ancho = min(x2, bx2) - max(x1, bx1)
            alto = min(y2, by2) - max(y1, by1)
            if ancho > 0 and alto > 0 and ancho * alto / area_det > SOLAPE_MAXIMO:
                return True
        return False

    @staticmethod
    def _nivel_por_tamano(ratio, umbral_medio, umbral_grande):
        """Clasifica un objeto por su tamaño relativo al frame."""
        if ratio >= umbral_grande:
            return "GRANDE", CV_ROJO
        elif ratio >= umbral_medio:
            return "MEDIO", CV_AMARILLO
        return "PEQUENO", CV_AZUL

    # ═══════════════════════════════════════════════════════════════════
    #  PANEL LATERAL (izquierda)
    # ═══════════════════════════════════════════════════════════════════
    def _crear_sidebar(self):
        sb = ctk.CTkFrame(self, width=290, corner_radius=0, fg_color=C_PANEL)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # ── Título con logo ──────────────────────────────────────────
        # Logo + nombre, alineados con el resto del panel
        cab = ctk.CTkFrame(sb, fg_color="transparent")
        cab.pack(fill="x", padx=22, pady=(18, 0))
        ctk.CTkLabel(cab, text="", image=self._crear_logo()).pack(
            side="left", padx=(0, 10))
        ctk.CTkLabel(cab, text="TacnaVial", font=self._f(22, True),
                     text_color=C_TEXTO, height=40).pack(side="left")

        self._sep(sb)

        # ── 1. Fuente ────────────────────────────────────────────────
        self._titulo_seccion(sb, "1.  FUENTE")
        self.btn_cargar = self._boton(sb, "Abrir video", self.cargar_video,
                                      principal=True)
        self.btn_camara = self._boton(sb, "Cámara en vivo",
                                      self.iniciar_camara)

        # ── 2. Sector ────────────────────────────────────────────────
        self._titulo_seccion(sb, "2.  SECTOR / CALLE")
        self.entry_sector = ctk.CTkEntry(
            sb, height=34, placeholder_text="Ej: Av. Bolognesi",
            font=self._f(13), fg_color=C_TARJETA, border_color=C_BORDE,
            text_color=C_TEXTO, corner_radius=8)
        self.entry_sector.pack(fill="x", padx=22, pady=(0, 4))
        self.entry_sector.bind("<Return>", self._cambiar_sector)
        ctk.CTkLabel(sb, text="Enter = cambiarlo con el video abierto",
                     font=self._f(11), text_color=C_TENUE, anchor="w",
                     wraplength=240, justify="left").pack(fill="x", padx=24)

        # ── 3. Exportar ──────────────────────────────────────────────
        self._titulo_seccion(sb, "3.  EXPORTAR")
        self.btn_captura = self._boton(sb, "Captura de pantalla",
                                       self.capturar_pantalla)
        self.btn_captura.configure(state="disabled")
        self.btn_grabar = self._boton(sb, "Grabar video procesado",
                                      self.alternar_grabacion)
        self._boton(sb, "Abrir carpeta de resultados", self.abrir_resultados)

        # ── Estado de la vía (semáforo) ──────────────────────────────
        # El mismo semáforo que se usará para los LEDs del Arduino
        est = ctk.CTkFrame(sb, fg_color=C_TARJETA, corner_radius=10)
        est.pack(fill="x", padx=18, pady=(12, 0))
        ctk.CTkLabel(est, text="ESTADO DE LA VÍA", font=self._f(11, True),
                     text_color=C_TENUE, anchor="w", height=18).pack(
                         fill="x", padx=14, pady=(8, 2))
        fila_e = ctk.CTkFrame(est, fg_color="transparent")
        fila_e.pack(fill="x", padx=14)
        self.punto_estado = ctk.CTkFrame(fila_e, width=22, height=22,
                                         corner_radius=11, fg_color=C_BORDE)
        self.punto_estado.pack(side="left", padx=(0, 10))
        self.lbl_estado_via = ctk.CTkLabel(
            fila_e, text="SIN ANÁLISIS", font=self._f(18, True),
            text_color=C_TENUE, anchor="w", height=26)
        self.lbl_estado_via.pack(side="left")
        self.lbl_estado_detalle = ctk.CTkLabel(
            est, text="Abre un video para empezar", font=self._f(11),
            text_color=C_TENUE, anchor="w", height=16)
        self.lbl_estado_detalle.pack(fill="x", padx=14, pady=(0, 8))
        self._estado_mostrado = None

        # ── Ficha del sistema (al fondo) ─────────────────────────────
        # Tabla: nombre a la izquierda, valor a la derecha y un punto de
        # color (verde = cargado, amarillo = cargando, rojo = error)
        info = ctk.CTkFrame(sb, fg_color=C_TARJETA, corner_radius=10)
        info.pack(fill="x", padx=18, pady=10, side="bottom")
        ctk.CTkLabel(info, text="SISTEMA", font=self._f(11, True),
                     text_color=C_TENUE, anchor="w", height=18).pack(
                         fill="x", padx=14, pady=(8, 4))
        tk.Frame(info, height=1, bg="#3A4150").pack(fill="x", padx=14)
        tabla = ctk.CTkFrame(info, fg_color="transparent")
        tabla.pack(fill="x", padx=14, pady=(4, 8))
        tabla.columnconfigure(1, weight=1)

        self._info_valor = {}     # clave → label con el valor
        self._info_punto = {}     # clave → puntito de color
        filas = [("baches", "Mod. baches", True),
                 ("deterioro", "Mod. deterioro", True),
                 ("confianza", "Confianza", False),
                 ("equipo", "Equipo", False)]
        for i, (clave, nombre, con_punto) in enumerate(filas):
            ctk.CTkLabel(tabla, text=nombre, font=self._f(12),
                         text_color=C_TENUE, anchor="w", height=22).grid(
                             row=i, column=0, sticky="w", pady=0)
            valor = ctk.CTkLabel(tabla, text="…", font=self._f(12, True),
                                 text_color=C_TEXTO, anchor="e", height=22)
            valor.grid(row=i, column=1, sticky="e", pady=0)
            self._info_valor[clave] = valor
            if con_punto:
                punto = ctk.CTkFrame(tabla, width=8, height=8,
                                     corner_radius=4, fg_color=C_AMARILLO)
                punto.grid(row=i, column=2, padx=(8, 0))
                self._info_punto[clave] = punto

    def _crear_logo(self):
        """Logo dibujado con PIL: una pista con sus líneas amarillas."""
        t = 160                                   # Se dibuja grande y se achica
        img = Image.new("RGBA", (t, t), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, t - 1, t - 1], radius=36, fill=C_ACENTO)
        d.polygon([(46, 30), (114, 30), (140, 132), (20, 132)],
                  fill=(30, 34, 44))
        for y in range(40, 125, 26):
            d.rectangle([75, y, 85, y + 14], fill=(234, 179, 8))
        return ctk.CTkImage(light_image=img, dark_image=img, size=(40, 40))

    # Textos del semáforo: (nombre, color, explicación)
    SEMAFORO = [("BUENO", C_VERDE, "Sin daños a la vista"),
                ("REGULAR", C_AMARILLO, "Bache mediano o deterioro"),
                ("MALO", C_ROJO, "Bache grande: requiere arreglo")]

    def _mostrar_estado_via(self):
        """Actualiza el cuadro ESTADO DE LA VÍA (solo si cambió)."""
        nuevo = self.estado_via if self.corriendo else None
        if nuevo == self._estado_mostrado:
            return
        self._estado_mostrado = nuevo
        if nuevo is None:
            self.punto_estado.configure(fg_color=C_BORDE)
            self.lbl_estado_via.configure(text="SIN ANÁLISIS",
                                          text_color=C_TENUE)
            self.lbl_estado_detalle.configure(text="Abre un video para empezar")
        else:
            nombre, color, detalle = self.SEMAFORO[nuevo]
            self.punto_estado.configure(fg_color=color)
            self.lbl_estado_via.configure(text=nombre, text_color=color)
            self.lbl_estado_detalle.configure(text=detalle)

    def _info(self, clave, valor, color=None):
        """Cambia un valor de la ficha SISTEMA (y su punto de color)."""
        self._info_valor[clave].configure(text=valor)
        if color is not None and clave in self._info_punto:
            self._info_punto[clave].configure(fg_color=color)

    # ── Ayudantes del panel lateral ───────────────────────────────────
    def _sep(self, padre):
        # Línea de 1 píxel (con tk.Frame: el de CTk no dibuja algo tan fino)
        tk.Frame(padre, height=1, bg="#3A4150").pack(
            fill="x", padx=22, pady=(12, 6))

    def _titulo_seccion(self, padre, texto):
        ctk.CTkLabel(padre, text=texto, font=self._f(11, True),
                     text_color=C_TENUE, anchor="w", height=20).pack(
                         fill="x", padx=22, pady=(6, 2))

    def _boton(self, padre, texto, comando, principal=False):
        """Botón del panel lateral. principal=True → azul."""
        b = ctk.CTkButton(
            padre, text=texto, height=34, font=self._f(13),
            fg_color=C_ACENTO if principal else C_TARJETA,
            hover_color=C_ACENTO_H if principal else C_TARJETA_H,
            text_color=C_TEXTO, text_color_disabled="#5C6370",
            border_width=0 if principal else 1, border_color=C_BORDE,
            corner_radius=8, command=comando)
        b.pack(fill="x", padx=22, pady=3)
        return b

    # ═══════════════════════════════════════════════════════════════════
    #  ÁREA PRINCIPAL — Video + Reproductor + Tarjetas + Registro
    # ═══════════════════════════════════════════════════════════════════
    def _crear_area_principal(self):
        main = ctk.CTkFrame(self, fg_color=C_FONDO, corner_radius=0)
        main.pack(side="right", fill="both", expand=True)

        # ── Encabezado: nombre del video + estado ────────────────────
        top = ctk.CTkFrame(main, fg_color="transparent")
        top.pack(fill="x", padx=18, pady=(16, 8))

        self.lbl_titulo_video = ctk.CTkLabel(
            top, text="Ningún video abierto", font=self._f(16, True),
            text_color=C_TEXTO, anchor="w")
        self.lbl_titulo_video.pack(side="left")

        self.lbl_estado = ctk.CTkLabel(
            top, text="DETENIDO", font=self._f(14, True), width=150,
            text_color=C_TEXTO, fg_color=C_TARJETA_H, corner_radius=8,
            height=36)
        self.lbl_estado.pack(side="right")

        # ── Video ────────────────────────────────────────────────────
        # (se coloca al final, para que use solo el espacio que sobra y no
        #  empuje el reproductor ni el registro fuera de la ventana)
        vf = ctk.CTkFrame(main, fg_color=C_VIDEO, corner_radius=12)

        # Label de tkinter normal: es más rápido para mostrar video
        self.lbl_video = tk.Label(vf, bg=C_VIDEO, bd=0, highlightthickness=0)
        self.lbl_video.pack(fill="both", expand=True, padx=4, pady=4)
        # Clic en el video = play/pausa (como en cualquier reproductor)
        self.lbl_video.bind("<Button-1>", self._clic_video)

        # ── Registro de detecciones ──────────────────────────────────
        lh = ctk.CTkFrame(main, fg_color="transparent")
        ctk.CTkLabel(lh, text="REGISTRO", font=self._f(11, True),
                     text_color=C_TENUE).pack(side="left")
        ctk.CTkButton(
            lh, text="Limpiar", width=70, height=24, font=self._f(11),
            fg_color=C_TARJETA, hover_color=C_TARJETA_H, text_color=C_TENUE,
            corner_radius=6, command=self._limpiar_log).pack(side="right")

        self.log_box = ctk.CTkTextbox(
            main, height=110, fg_color=C_PANEL, text_color=C_TEXTO,
            font=ctk.CTkFont(family="Consolas", size=12), corner_radius=10,
            border_width=1, border_color=C_BORDE, state="disabled")
        self.log_box.tag_config("tenue", foreground=C_TENUE)
        for tipo, (_, color) in self.TIPOS_LOG.items():
            self.log_box.tag_config(tipo, foreground=color)
        self.log_box.pack(fill="x", padx=18, pady=(0, 16), side="bottom")
        lh.pack(fill="x", padx=18, pady=(10, 4), side="bottom")

        # ── Reproductor ──────────────────────────────────────────────
        rep = ctk.CTkFrame(main, fg_color=C_PANEL, corner_radius=12)
        rep.pack(fill="x", padx=18, pady=(8, 0), side="bottom")

        # Barra de tiempo (se puede arrastrar o hacer clic para moverse)
        self.slider = ctk.CTkSlider(
            rep, from_=0, to=1, height=16, command=self._al_mover_barra,
            button_color=C_ACENTO, button_hover_color=C_ACENTO_H,
            progress_color=C_ACENTO, fg_color=C_BORDE, state="disabled")
        self.slider.set(0)
        self.slider.pack(fill="x", padx=16, pady=(12, 4))
        self.slider.bind("<ButtonRelease-1>", self._al_soltar_barra)

        fila = ctk.CTkFrame(rep, fg_color="transparent")
        fila.pack(fill="x", padx=12, pady=(0, 10))

        self.btn_atras = self._boton_rep(
            fila, f"⏪ {SALTO_SEGUNDOS}s", lambda: self.mover(-SALTO_SEGUNDOS))
        self.btn_play = ctk.CTkButton(
            fila, text="▶", width=48, height=40, font=self._f(18, True),
            fg_color=C_ACENTO, hover_color=C_ACENTO_H, text_color="white",
            text_color_disabled="#8FA6C8", corner_radius=20,
            command=self.pausar_reanudar, state="disabled")
        self.btn_play.pack(side="left", padx=6)
        self.btn_adelante = self._boton_rep(
            fila, f"{SALTO_SEGUNDOS}s ⏩", lambda: self.mover(SALTO_SEGUNDOS))
        self.btn_detener = self._boton_rep(fila, "⏹  Detener", self.detener)

        self.lbl_tiempo = ctk.CTkLabel(
            fila, text="00:00 / 00:00", font=self._f(13, True),
            text_color=C_TEXTO)
        self.lbl_tiempo.pack(side="left", padx=16)

        # Velocidad de reproducción
        self.sel_velocidad = ctk.CTkSegmentedButton(
            fila, values=["0.5x", "1x", "2x"], command=self.cambiar_velocidad,
            font=self._f(12), selected_color=C_ACENTO,
            selected_hover_color=C_ACENTO_H, unselected_color=C_TARJETA,
            unselected_hover_color=C_TARJETA_H, fg_color=C_TARJETA)
        self.sel_velocidad.set("1x")
        self.sel_velocidad.pack(side="right", padx=4)
        ctk.CTkLabel(fila, text="Velocidad", font=self._f(12),
                     text_color=C_TENUE).pack(side="right", padx=6)

        # Ahora sí el video, con el espacio que queda
        vf.pack(fill="both", expand=True, padx=18)

        self._estado_botones_reproductor(False)

    def _boton_rep(self, padre, texto, comando):
        """Botón pequeño del reproductor."""
        b = ctk.CTkButton(
            padre, text=texto, width=78, height=36, font=self._f(12),
            fg_color=C_TARJETA, hover_color=C_TARJETA_H, text_color=C_TEXTO,
            text_color_disabled="#5C6370", corner_radius=8, command=comando)
        b.pack(side="left", padx=4)
        return b

    def _estado_botones_reproductor(self, activo):
        """Activa o desactiva los botones del reproductor.
        Con cámara solo funcionan Pausa y Detener (no se puede adelantar)."""
        es_video = activo and self.fuente_es_video
        self.btn_play.configure(state="normal" if activo else "disabled")
        self.btn_detener.configure(state="normal" if activo else "disabled")
        self.btn_captura.configure(state="normal" if activo else "disabled")
        for b in (self.btn_atras, self.btn_adelante):
            b.configure(state="normal" if es_video else "disabled")
        self.slider.configure(state="normal" if es_video else "disabled")
        self.sel_velocidad.configure(
            state="normal" if es_video else "disabled")

    # Estados del programa: (texto, color de fondo, color de letra)
    ESTADOS = {
        "cargando":   ("CARGANDO MODELOS…", "#CA8A04", "#FFFFFF"),
        "listo":      ("LISTO", C_TARJETA_H, C_TEXTO),
        "sin_modelo": ("SIN MODELO", "#DC2626", "#FFFFFF"),
        "analizando": ("ANALIZANDO", "#16A34A", "#FFFFFF"),
        "en_vivo":    ("EN VIVO", "#DC2626", "#FFFFFF"),
        "pausa":      ("EN PAUSA", "#CA8A04", "#FFFFFF"),
        "terminado":  ("ANÁLISIS TERMINADO", "#2563EB", "#FFFFFF"),
        "detenido":   ("DETENIDO", C_TARJETA_H, C_TEXTO),
    }

    def _poner_estado(self, clave):
        """Cambia la etiqueta de estado de arriba a la derecha."""
        texto, fondo, letra = self.ESTADOS[clave]
        self.lbl_estado.configure(text=f"   {texto}   ", fg_color=fondo,
                                  text_color=letra)

    # ═══════════════════════════════════════════════════════════════════
    #  PANTALLA DE BIENVENIDA
    # ═══════════════════════════════════════════════════════════════════
    def _mostrar_bienvenida(self):
        """Dibuja un mensaje de inicio en el área de video."""
        w, h = 900, 480
        img = Image.new("RGB", (w, h), (11, 12, 15))
        draw = ImageDraw.Draw(img)
        try:
            font_g = ImageFont.truetype("segoeuib.ttf", 40)
            font_c = ImageFont.truetype("segoeui.ttf", 16)
        except OSError:
            font_g = ImageFont.load_default()
            font_c = font_g

        lineas = [
            ("TacnaVial", font_g, (232, 234, 237)),
            ("Detección de baches y deterioro en vías de Tacna",
             font_c, (154, 163, 178)),
            ("Abre un video o inicia la cámara desde el panel izquierdo",
             font_c, (100, 108, 122)),
        ]
        y = h // 2 - 30
        for texto, fuente, color in lineas:
            bbox = draw.textbbox((0, 0), texto, font=fuente)
            tw = bbox[2] - bbox[0]
            draw.text(((w - tw) // 2, y), texto, fill=color, font=fuente)
            y += (bbox[3] - bbox[1]) + 18
            if fuente is font_g:
                y += 14         # Más aire debajo del título grande

        # Ícono: una pista con sus líneas
        cx = w // 2
        draw.polygon([(cx - 70, h // 2 - 50), (cx + 70, h // 2 - 50),
                      (cx + 30, h // 2 - 150), (cx - 30, h // 2 - 150)],
                     fill=(36, 40, 51))
        for yy in range(h // 2 - 140, h // 2 - 55, 22):
            draw.rectangle([cx - 3, yy, cx + 3, yy + 11], fill=(234, 179, 8))

        self._foto_bienvenida = ImageTk.PhotoImage(img)
        self.lbl_video.configure(image=self._foto_bienvenida)

    # ═══════════════════════════════════════════════════════════════════
    #  CARGA DE MODELOS
    # ═══════════════════════════════════════════════════════════════════
    def _cargar_modelo(self):
        self._log("Cargando modelos, espera unos segundos...")
        self._poner_estado("cargando")
        self.update_idletasks()     # Dibuja el mensaje antes de cargar

        # ── Modelo de BACHES (obligatorio) ───────────────────────────
        ruta = RUTA_MODELO_BACHES
        if not os.path.exists(ruta):
            messagebox.showwarning(
                "Modelo no encontrado",
                f"No se encontró el modelo de baches en:\n{ruta}\n\n"
                "Selecciona el archivo best_baches.pt manualmente.")
            ruta = filedialog.askopenfilename(
                title="Seleccionar modelo de BACHES (.pt)",
                filetypes=[("Modelo PyTorch", "*.pt")])
            if not ruta:
                self._info("baches", "No cargado", C_ROJO)
                self._log("Modelo no cargado. Reinicia el programa.", "ERROR")
                self._poner_estado("sin_modelo")
                return

        try:
            from ultralytics import YOLO
            import torch
            # Si la PC tiene tarjeta NVIDIA se usa (mucho más rápido)
            if torch.cuda.is_available():
                self.dispositivo = 0
                texto_disp = "GPU"
            else:
                self.dispositivo = "cpu"
                texto_disp = "CPU"
            self._log(f"Procesando con {texto_disp}")

            self.modelo = YOLO(ruta)
            self._log("Modelo de baches cargado (best_baches.pt)")
        except Exception as e:
            self._info("baches", "Error", C_ROJO)
            self._log(f"No se pudo cargar el modelo: {e}", "ERROR")
            self._poner_estado("sin_modelo")
            return

        # ── Modelo de DETERIORO (opcional) ───────────────────────────
        if os.path.exists(RUTA_MODELO_DETERIORO):
            try:
                self.modelo_det = YOLO(RUTA_MODELO_DETERIORO)
                self._log("Modelo de deterioro cargado (best_deterioro.pt)")
            except Exception as e:
                self.modelo_det = None
                self._log(f"No se pudo cargar el modelo de deterioro: {e}",
                          "AVISO")
        else:
            self._log("No está best_deterioro.pt: solo se detectarán baches",
                      "AVISO")

        self._poner_estado("listo")
        self._info("baches", "YOLOv8n", C_VERDE)
        if self.modelo_det:
            self._info("deterioro", "YOLOv8n-seg", C_VERDE)
        else:
            self._info("deterioro", "No cargado", C_ROJO)
        self._info("confianza", f"{int(CONFIANZA * 100)} %")
        self._info("equipo", texto_disp)

    # ═══════════════════════════════════════════════════════════════════
    #  ABRIR VIDEO / CÁMARA
    # ═══════════════════════════════════════════════════════════════════
    def cargar_video(self):
        """Abre un diálogo para seleccionar un archivo de video."""
        if self.modelo is None:
            messagebox.showerror("Error", "Los modelos aún no se cargan.")
            return

        ruta = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"),
                ("Todos los archivos", "*.*")])
        if not ruta:
            return

        self.detener()
        self._obtener_sector()
        self.ruta_video_actual = ruta
        self.fuente_es_video = True
        self.lbl_titulo_video.configure(text=os.path.basename(ruta))
        self._log(f"Video abierto: {os.path.basename(ruta)}", "VIDEO")
        self._iniciar(ruta)

    def iniciar_camara(self):
        """Inicia la detección usando la cámara web."""
        if self.modelo is None:
            messagebox.showerror("Error", "Los modelos aún no se cargan.")
            return

        self.detener()
        self._obtener_sector()
        self.fuente_es_video = False
        self.ruta_video_actual = None
        self.lbl_titulo_video.configure(text="Cámara en vivo")
        self._log("Cámara en vivo iniciada", "VIDEO")
        self._iniciar(0)

    def _obtener_sector(self):
        """Toma el sector escrito en el panel. Si está vacío, lo pregunta."""
        texto = self.entry_sector.get().strip()
        if texto == "":
            dialogo = ctk.CTkInputDialog(
                title="Sector",
                text="¿En qué calle o sector de Tacna se grabó?\n"
                     "(ej: Av. Bolognesi)")
            texto = dialogo.get_input()
            texto = texto.strip() if texto else ""
            if texto == "":
                texto = "Sin sector"
            self.entry_sector.delete(0, "end")
            self.entry_sector.insert(0, texto)
        self.sector = texto
        self._log(f"Sector: {self.sector}")

    def _cambiar_sector(self, evento=None):
        """Enter en la caja del sector: las siguientes detecciones se
        guardan con el nuevo nombre. Quita el cursor de la caja para que
        vuelvan a funcionar los atajos de teclado."""
        texto = self.entry_sector.get().strip()
        if texto and texto != self.sector:
            self.sector = texto
            self._log(f"Sector cambiado a: {self.sector}")
        self.focus_set()

    # ═══════════════════════════════════════════════════════════════════
    #  CONTROLES DEL REPRODUCTOR
    # ═══════════════════════════════════════════════════════════════════
    def pausar_reanudar(self):
        """Botón ▶ / ⏸. Si el video ya terminó, lo vuelve a reproducir
        desde el inicio (sin volver a contar lo que ya se contó)."""
        if not self.corriendo:
            return
        if self.fin_video:
            self.pedido_salto = 0
            self.pausado = False
        else:
            self.pausado = not self.pausado
        self._refrescar_play()

    def _refrescar_play(self):
        """Pone el ícono correcto en el botón de play y el estado."""
        if self.pausado:
            self.btn_play.configure(text="▶")
            if self.fin_video:
                self._poner_estado("terminado")
            else:
                self._poner_estado("pausa")
        else:
            self.btn_play.configure(text="⏸")
            self._poner_estado(
                "analizando" if self.fuente_es_video else "en_vivo")

    def mover(self, segundos):
        """Retrocede (negativo) o adelanta (positivo) el video."""
        if not self.corriendo or not self.fuente_es_video:
            return
        self._saltar_a(self.frame_idx + segundos * self.fps_video)

    def _saltar_a(self, frame):
        """Pide al hilo de detección que vaya a un frame del video."""
        if not self.corriendo or not self.fuente_es_video:
            return
        if self.ruta_guardar is not None:
            self._log("Mientras se graba no se puede mover el video", "AVISO")
            return
        frame = int(max(0, min(frame, self.total_frames - 1)))
        self.pedido_salto = frame

    def _al_mover_barra(self, valor):
        """Mientras se arrastra la barra solo se muestra el tiempo."""
        self.arrastrando = True
        total = self.total_frames / self.fps_video
        self.lbl_tiempo.configure(
            text=f"{self._formato_tiempo(valor / self.fps_video)} / "
                 f"{self._formato_tiempo(total)}")

    def _al_soltar_barra(self, evento=None):
        """Al soltar la barra, el video salta a ese punto."""
        if self.arrastrando:
            self.arrastrando = False
            self._saltar_a(self.slider.get())

    def cambiar_velocidad(self, valor):
        """0.5x, 1x o 2x."""
        self.velocidad = float(valor.replace("x", ""))
        self._log(f"Velocidad: {valor}")

    def _clic_video(self, evento=None):
        self.focus_set()          # Sale de la caja del sector
        self.pausar_reanudar()

    def _tecla(self, accion):
        """Atajos de teclado (no funcionan mientras se escribe el sector)."""
        try:
            if isinstance(self.focus_get(), tk.Entry):
                return
        except Exception:
            pass
        if accion == "play":
            self.pausar_reanudar()
        elif accion == "atras":
            self.mover(-SALTO_SEGUNDOS)
        elif accion == "adelante":
            self.mover(SALTO_SEGUNDOS)

    def detener(self):
        """Detiene la detección, guarda el CSV y libera el video."""
        if not self.corriendo:
            return

        self.corriendo = False

        # Esperar a que el hilo termine
        if (self.hilo is not None
                and self.hilo is not threading.current_thread()):
            self.hilo.join(timeout=3)
        self.hilo = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Cerrar el video guardado (si se estaba grabando)
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self._log(f"Video procesado guardado: {self.ruta_guardar}",
                      "GUARDADO")
        self.ruta_guardar = None
        self.parar_grabacion = False
        self._reset_ruta_guardar()

        # Resetear controles
        self.pausado = False
        self.fin_video = False
        self.btn_play.configure(text="▶")
        self._estado_botones_reproductor(False)
        self.slider.set(0)
        self.lbl_tiempo.configure(text="00:00 / 00:00")
        self._poner_estado("detenido")

        self._guardar_csv()

        total = sum(sum(n.values()) for n in self.conteo_global.values())
        if total > 0 and not self.resumen_mostrado:
            self._mostrar_resumen()

        self._log("Análisis detenido")

    # ═══════════════════════════════════════════════════════════════════
    #  CSV, CAPTURAS Y GRABACIÓN
    # ═══════════════════════════════════════════════════════════════════
    def _guardar_csv(self):
        """Agrega las detecciones a resultados/detecciones.csv (no lo borra)."""
        if len(self.registros) == 0:
            return
        os.makedirs(RUTA_SALIDA, exist_ok=True)
        df = pd.DataFrame(self.registros)
        try:
            # Se junta con lo que ya había y se escribe todo de nuevo
            # (así las columnas siempre quedan en orden)
            if os.path.exists(RUTA_CSV):
                viejo = pd.read_csv(RUTA_CSV, encoding="utf-8-sig")
                df = pd.concat([viejo, df], ignore_index=True)
            df.to_csv(RUTA_CSV, index=False, encoding="utf-8-sig")
            self._log(f"{len(self.registros)} detecciones guardadas "
                      f"en detecciones.csv", "GUARDADO")
        except PermissionError:
            self._log("No se pudo guardar el CSV (¿está abierto en Excel?)",
                      "ERROR")
            return
        self.registros = []
        self.fila_id = {}

    def capturar_pantalla(self):
        """Guarda el frame actual como imagen PNG."""
        with self.lock:
            frame = (self.frame_actual.copy()
                     if self.frame_actual is not None else None)
        if frame is None:
            self._log("No hay imagen para capturar", "AVISO")
            return
        os.makedirs(RUTA_CAPTURAS, exist_ok=True)
        nombre = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(os.path.join(RUTA_CAPTURAS, nombre), frame)
        self._log(f"Captura guardada: capturas/{nombre}", "GUARDADO")

    def abrir_resultados(self):
        """Abre la carpeta resultados en el explorador de Windows."""
        os.makedirs(RUTA_SALIDA, exist_ok=True)
        os.startfile(RUTA_SALIDA)

    def alternar_grabacion(self):
        """1er clic: elegir dónde guardar y empezar a grabar.
        2do clic: terminar la grabación."""
        if self.ruta_guardar is not None:
            if self.corriendo:
                self.parar_grabacion = True    # El hilo cierra el archivo
            else:
                self.ruta_guardar = None
                self._reset_ruta_guardar()
                self._log("Grabación cancelada")
            return

        os.makedirs(RUTA_SALIDA, exist_ok=True)
        if self.fuente_es_video and self.ruta_video_actual:
            base = os.path.splitext(
                os.path.basename(self.ruta_video_actual))[0]
            sugerido = f"{base}_resultado.mp4"
        else:
            sugerido = f"deteccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        ruta = filedialog.asksaveasfilename(
            title="Guardar video procesado como…", initialdir=RUTA_SALIDA,
            initialfile=sugerido, defaultextension=".mp4",
            filetypes=[("Video MP4", "*.mp4")])
        if not ruta:
            return

        self.ruta_guardar = ruta
        self.btn_grabar.configure(text="●  Grabando — clic para terminar",
                                  fg_color=C_ROJO, hover_color=C_ROJO_H,
                                  text_color="#FFFFFF", border_width=0)
        if self.corriendo:
            self._log(f"Grabando desde este momento en "
                      f"{os.path.basename(ruta)}", "VIDEO")
        else:
            self._log(f"Se grabará en {os.path.basename(ruta)} cuando "
                      f"empiece el video", "VIDEO")

    def _reset_ruta_guardar(self):
        """Vuelve el botón de grabar a como estaba."""
        self.btn_grabar.configure(text="Grabar video procesado",
                                  fg_color=C_TARJETA, hover_color=C_TARJETA_H,
                                  text_color=C_TEXTO, border_width=1)

    def _crear_writer(self, ancho, alto, fps):
        """Crea el VideoWriter en la ruta elegida. Devuelve True si pudo."""
        writer = cv2.VideoWriter(
            self.ruta_guardar, cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (ancho, alto))
        if not writer.isOpened():
            self._desde_hilo(self._log,
                             f"No se pudo crear el video en: {self.ruta_guardar}",
                             "ERROR")
            self.ruta_guardar = None
            self._desde_hilo(self._reset_ruta_guardar)
            return False
        self.writer = writer
        return True

    def _cerrar_grabacion(self):
        """(Hilo) Cierra el archivo de video que se estaba grabando."""
        ruta = self.ruta_guardar
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self._desde_hilo(self._log, f"Video procesado guardado: {ruta}",
                             "GUARDADO")
        self.ruta_guardar = None
        self.parar_grabacion = False
        self._desde_hilo(self._reset_ruta_guardar)

    # ═══════════════════════════════════════════════════════════════════
    #  INICIAR PROCESAMIENTO
    # ═══════════════════════════════════════════════════════════════════
    def _iniciar(self, fuente):
        """Abre la fuente de video e inicia el hilo de detección."""
        self.cap = cv2.VideoCapture(fuente)
        if not self.cap.isOpened():
            self._log("No se pudo abrir la fuente de video", "ERROR")
            self.lbl_titulo_video.configure(text="Ningún video abierto")
            self.ruta_video_actual = None
            messagebox.showerror(
                "Error", "No se pudo acceder a la cámara o abrir el video.")
            self.cap = None
            return

        if not self.fuente_es_video:
            # Cámara: buffer mínimo para ver siempre la imagen más reciente
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0 or fps > 120:
            fps = 30.0
        self.fps_video = fps

        # Reiniciar conteos y estado
        self.ids_vistos.clear()
        self.registros = []
        self.nivel_id = {}
        self.fila_id = {}
        self.det_previo = []
        self.det_frame = -999
        self.cuenta_proc = 0
        self.det_recientes = []
        self.det_contador = 0
        self.conteo_global = self._conteo_vacio()
        self.frame_idx = 0
        self.frame_max = 0
        self.frames_escritos = 0
        self.frame_actual = None
        self._ultimo_mostrado = None
        self.pedido_salto = None
        self.fin_video = False
        self.resumen_mostrado = False
        self.estado_via = 0
        self.t_estado = 0.0

        if self.fuente_es_video:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.configure(to=max(1, self.total_frames))
        else:
            self.total_frames = 0
        self.slider.set(0)

        self.corriendo = True
        self.pausado = False
        self._estado_botones_reproductor(True)
        self._refrescar_play()

        # Lanzar hilo de procesamiento
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    # ═══════════════════════════════════════════════════════════════════
    #  BUCLE DE PROCESAMIENTO (hilo secundario)
    # ═══════════════════════════════════════════════════════════════════
    def _bucle(self):
        """Lee frames, ejecuta YOLO y dibuja los resultados.

        El video va a VELOCIDAD REAL (por la velocidad elegida): si la
        detección es lenta se saltan frames, si es rápida se espera.
        Aquí también se atienden los pedidos del reproductor: saltar a
        otro punto, cambiar velocidad y terminar la grabación."""

        fps = self.fps_video
        vel = self.velocidad
        t_inicio = time.perf_counter()
        estaba_pausado = False
        mostrar_uno = False     # Procesar 1 frame aunque esté en pausa

        def reloj_desde_aqui():
            # Ajusta el reloj para que el frame actual sea "ahora"
            return time.perf_counter() - self.frame_idx / (fps * vel)

        while self.corriendo:
            # ── Pedido: terminar grabación ───────────────────────────
            if self.parar_grabacion:
                self._cerrar_grabacion()

            # ── Pedido: saltar a otro punto del video ────────────────
            if self.pedido_salto is not None:
                destino = self.pedido_salto
                self.pedido_salto = None
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, destino)
                self.frame_idx = destino
                self.fin_video = False
                self.det_previo = []
                self.det_frame = -999          # Recalcular el deterioro
                self.det_recientes = []
                t_inicio = reloj_desde_aqui()
                mostrar_uno = True     # Para ver la imagen aunque esté pausado
                self._desde_hilo(self._refrescar_play)

            # ── Pedido: cambiar velocidad ────────────────────────────
            if self.velocidad != vel:
                vel = self.velocidad
                t_inicio = reloj_desde_aqui()

            # ── Pausa ────────────────────────────────────────────────
            if self.pausado and not mostrar_uno:
                estaba_pausado = True
                time.sleep(0.05)
                continue
            if estaba_pausado and not self.pausado:
                t_inicio = reloj_desde_aqui()   # El reloj sigue desde aquí
                estaba_pausado = False

            # ── Sincronización con el tiempo real (solo videos) ──────
            if self.fuente_es_video and not mostrar_uno:
                transcurrido = time.perf_counter() - t_inicio
                objetivo = int(transcurrido * fps * vel)

                # Vamos adelantados: esperar
                if self.frame_idx > objetivo:
                    time.sleep(max(0.001, self.frame_idx / (fps * vel)
                                   - transcurrido))
                    continue

                # Vamos atrasados: saltar frames
                fin = False
                while self.frame_idx < objetivo:
                    if not self.cap.grab():
                        fin = True
                        break
                    self.frame_idx += 1
                if fin:
                    self._fin_del_video()
                    continue
            mostrar_uno = False

            ret, frame = self.cap.read()
            if not ret:
                if self.fuente_es_video:
                    self._fin_del_video()
                    continue
                self._desde_hilo(self._log, "Error al leer la cámara", "ERROR")
                self._desde_hilo(self.detener)
                break

            self.frame_idx += 1

            # Solo se cuentan baches en partes del video que todavía no se
            # analizaron. Si retrocedes, se ven, pero no se cuentan 2 veces.
            zona_nueva = self.frame_idx > self.frame_max
            if zona_nueva:
                self.frame_max = self.frame_idx

            # Achicar frames muy grandes (4K) para que no se congele
            if frame.shape[1] > ANCHO_MAX:
                nuevo_alto = int(frame.shape[0] * ANCHO_MAX / frame.shape[1])
                frame = cv2.resize(frame, (ANCHO_MAX, nuevo_alto),
                                   interpolation=cv2.INTER_AREA)

            alto_f, ancho_f = frame.shape[:2]
            area_frame = ancho_f * alto_f
            grosor = max(2, ancho_f // 400)
            escala = max(0.5, ancho_f / 1400)

            # ── Detección YOLO ───────────────────────────────────────
            try:
                # BACHES: en todos los frames (con seguimiento)
                res_b = self.modelo.track(
                    source=frame, persist=True, imgsz=IMGSZ,
                    conf=CONFIANZA, device=self.dispositivo, verbose=False)[0]

                # DETERIORO: 1 de cada CADA_DETERIORO frames procesados,
                # o antes si el último resultado ya está viejo
                self.cuenta_proc += 1
                edad = self.frame_idx - self.det_frame
                toca = (self.cuenta_proc % CADA_DETERIORO == 0
                        or edad > MAX_EDAD_DET)
                res_d = None
                if self.modelo_det is not None and toca:
                    res_d = self.modelo_det.predict(
                        source=frame, imgsz=IMGSZ_DET,
                        conf=CONFIANZA_DET, device=self.dispositivo,
                        verbose=False)[0]
                    self.det_frame = self.frame_idx
            except Exception as e:
                self._desde_hilo(self._log, f"Error en la detección: {e}",
                                 "ERROR")
                self._desde_hilo(self.detener)
                break

            # ── Juntar los objetos de los 2 modelos en una sola lista ──
            # Cada objeto: (tipo, id, confianza, x1, y1, x2, y2, poligono, nuevo)
            objetos = []
            for caja in res_b.boxes:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                id_obj = int(caja.id[0]) if caja.id is not None else None
                objetos.append(("bache", id_obj, float(caja.conf[0]),
                                x1, y1, x2, y2, None, True))

            if res_d is not None:
                lista_det = []
                if res_d.masks is not None:
                    for i, caja in enumerate(res_d.boxes):
                        poly = res_d.masks.xy[i].astype(np.int32)
                        if len(poly) < 3:
                            continue
                        x1, y1, x2, y2 = map(int, caja.xyxy[0])
                        if self._tapado_por_bache(x1, y1, x2, y2, objetos):
                            continue
                        id_obj = self._id_deterioro(x1, y1, x2, y2)
                        lista_det.append(("deterioro", id_obj,
                                          float(caja.conf[0]),
                                          x1, y1, x2, y2, poly, True))
                self.det_previo = lista_det
                objetos += lista_det
            else:
                # Reusar el último deterioro (es reciente: máximo
                # MAX_EDAD_DET frames). Se dibuja pero no se registra.
                for d in self.det_previo:
                    objetos.append(d[:8] + (False,))

            # Límites de la zona de medición (en píxeles)
            y_zona_ini = int(alto_f * ZONA_INICIO)
            y_zona_fin = int(alto_f * ZONA_FIN)

            c_frame = {"PEQUENO": 0, "MEDIO": 0, "GRANDE": 0}
            dibujos = []
            peor = 0     # Semáforo de este frame: 0 bueno, 1 regular, 2 malo

            for (tipo_cls, id_obj, conf_val,
                 x1, y1, x2, y2, poly, nuevo) in objetos:

                forma = "poly" if poly is not None else "caja"
                datos = poly if poly is not None else (x1, y1, x2, y2)
                clave = (tipo_cls, id_obj)

                centro_y = (y1 + y2) / 2
                en_zona = y_zona_ini <= centro_y <= y_zona_fin

                if poly is not None:
                    ratio = cv2.contourArea(poly) / area_frame
                    nivel, color = self._nivel_por_tamano(
                        ratio, DET_MEDIO, DET_GRANDE)
                else:
                    ratio = ((x2 - x1) * (y2 - y1)) / area_frame
                    nivel, color = self._nivel_por_tamano(
                        ratio, BACHE_MEDIO, BACHE_GRANDE)

                if tipo_cls == "deterioro":
                    # DETERIORO: siempre se dibuja, morado y sin tamaño
                    dibujos.append((forma, datos, CV_DETERIORO,
                                    f"deterioro {conf_val:.2f}", x1, y1))
                    peor = max(peor, 1)
                else:
                    # BACHE: bache_pequeno / mediano / grande
                    if en_zona:
                        dibujos.append((forma, datos, color,
                                        f"{NOMBRE_BACHE[nivel]} {conf_val:.2f}",
                                        x1, y1))
                        c_frame[nivel] += 1
                        peor = max(peor, ORDEN_NIVEL[nivel])
                    elif clave in self.nivel_id:
                        # Ya se midió antes: se sigue mostrando con su tamaño
                        n = self.nivel_id[clave]
                        col = {"PEQUENO": CV_AZUL, "MEDIO": CV_AMARILLO,
                               "GRANDE": CV_ROJO}[n]
                        dibujos.append((forma, datos, col,
                                        f"{NOMBRE_BACHE[n]} {conf_val:.2f}",
                                        x1, y1))
                        peor = max(peor, ORDEN_NIVEL[n])
                    # Si todavía no llega a la zona, no se dibuja

                if not en_zona:
                    continue
                if id_obj is None or not nuevo or not zona_nueva:
                    continue

                # Registrar objetos nuevos (la primera vez que entran a la zona)
                if clave not in self.ids_vistos:
                    self.ids_vistos.add(clave)
                    self.conteo_global[tipo_cls][nivel] += 1

                    if self.fuente_es_video:
                        fuente = os.path.basename(self.ruta_video_actual)
                    else:
                        fuente = "camara"
                    self.registros.append({
                        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "sector": self.sector,
                        "fuente": fuente,
                        "segundo": round(self.frame_idx / fps, 1),
                        "id": id_obj,
                        "tipo": tipo_cls,
                        "severidad": nivel,
                        "confianza": round(conf_val, 3),
                        "area_pct": round(ratio * 100, 2),
                    })
                    self.nivel_id[clave] = nivel
                    self.fila_id[clave] = len(self.registros) - 1
                    tiempo = self._formato_tiempo(self.frame_idx / fps)
                    if tipo_cls == "deterioro":
                        self._desde_hilo(
                            self._log, f"#{id_obj:<4}{'':<10}{conf_val:.0%}",
                            "DETERIORO", tiempo)
                    else:
                        self._desde_hilo(
                            self._log,
                            f"#{id_obj:<4}{TAMANO_TEXTO[nivel]:<10}"
                            f"{conf_val:.0%}", nivel, tiempo)

                # Si un objeto ya visto se ve MÁS GRANDE, vale el tamaño máximo
                elif (clave in self.nivel_id
                      and ORDEN_NIVEL[nivel] > ORDEN_NIVEL[self.nivel_id[clave]]):
                    viejo = self.nivel_id[clave]
                    self.conteo_global[tipo_cls][viejo] -= 1
                    self.conteo_global[tipo_cls][nivel] += 1
                    self.nivel_id[clave] = nivel
                    fila = self.fila_id.get(clave)
                    if fila is not None and fila < len(self.registros):
                        self.registros[fila]["severidad"] = nivel
                        self.registros[fila]["area_pct"] = round(ratio * 100, 2)
                    if tipo_cls == "bache":
                        self._desde_hilo(
                            self._log,
                            f"#{id_obj:<4}creció a {TAMANO_TEXTO[nivel]} "
                            f"(antes {TAMANO_TEXTO[viejo]})", nivel,
                            self._formato_tiempo(self.frame_idx / fps))

            # ── Dibujar sobre el frame ───────────────────────────────
            # 1. Relleno semitransparente de los polígonos de deterioro
            overlay = frame.copy()
            for tipo, datos, color, *_ in dibujos:
                if tipo == "poly":
                    cv2.fillPoly(overlay, [datos], color)
            frame = cv2.addWeighted(
                overlay, TRANSPARENCIA, frame, 1 - TRANSPARENCIA, 0)

            # 2. Zona de medición (solo si MOSTRAR_ZONA = True)
            if MOSTRAR_ZONA:
                cv2.line(frame, (0, y_zona_ini), (ancho_f, y_zona_ini),
                         CV_ZONA, grosor)
                cv2.line(frame, (0, y_zona_fin), (ancho_f, y_zona_fin),
                         CV_ZONA, grosor)

            # 3. Bordes, rectángulos y etiquetas (letra grande y en negrita
            #    para que se lea bien qué es cada cosa)
            esc_t = max(0.55, ancho_f / 1120)
            grosor_t = max(2, ancho_f // 640)
            for tipo, datos, color, etiqueta, tx, ty in dibujos:
                if tipo == "poly":
                    cv2.polylines(frame, [datos], True, color, grosor)
                else:
                    bx1, by1, bx2, by2 = datos
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2),
                                  color, grosor)
                (tw, th), _ = cv2.getTextSize(
                    etiqueta, cv2.FONT_HERSHEY_SIMPLEX, esc_t, grosor_t)
                pad = int(6 * esc_t)
                # Que la etiqueta no se salga de la imagen
                ex = max(0, min(tx, ancho_f - tw - 2 * pad))
                ey = max(0, ty - th - 2 * pad)
                cv2.rectangle(frame, (ex, ey),
                              (ex + tw + 2 * pad, ey + th + 2 * pad), color, -1)
                # Letra negra en fondos claros y blanca en fondos oscuros
                b, g, r = color
                claro = 0.114 * b + 0.587 * g + 0.299 * r > 140
                cv2.putText(frame, etiqueta, (ex + pad, ey + pad + th),
                            cv2.FONT_HERSHEY_SIMPLEX, esc_t,
                            (0, 0, 0) if claro else (255, 255, 255),
                            grosor_t, cv2.LINE_AA)

            # 4. Semáforo de la vía (no se dibuja: se usará para el Arduino).
            #    Sube al toque, pero baja recién después de 1.5 s sin ver
            #    nada peor (así no parpadea)
            ahora_s = time.perf_counter()
            if peor >= self.estado_via:
                self.estado_via = peor
                self.t_estado = ahora_s
            elif ahora_s - self.t_estado > 1.5:
                self.estado_via = peor

            # 5. Conteo total en la esquina (también sale en el video grabado)
            frame = self._dibujar_conteo(frame)

            # ── Guardar en el video de salida ────────────────────────
            if self.ruta_guardar and self.writer is None:
                ahora = time.perf_counter() - t_inicio
                if self._crear_writer(ancho_f, alto_f, fps):
                    self._desde_hilo(self._log, "Grabando video procesado...",
                                     "VIDEO")
                    if self.fuente_es_video:
                        self.frames_escritos = self.frame_idx - 1
                    else:
                        self.frames_escritos = int(ahora * fps)

            if self.writer is not None:
                if self.fuente_es_video:
                    # Incluye los frames saltados: dura lo mismo que el original
                    objetivo_w = self.frame_idx
                else:
                    ahora = time.perf_counter() - t_inicio
                    objetivo_w = max(self.frames_escritos + 1,
                                     int(ahora * fps))
                while self.frames_escritos < objetivo_w:
                    self.writer.write(frame)
                    self.frames_escritos += 1

            # Compartir con el hilo principal
            with self.lock:
                self.frame_actual = frame

    def _dibujar_conteo(self, frame):
        """Cuadro semitransparente arriba a la izquierda con el CONTEO
        TOTAL acumulado (solo va subiendo)."""
        alto_f, ancho_f = frame.shape[:2]
        k = ancho_f / 1280                 # Escala según el tamaño del video
        letra = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.62 * k                      # Tamaño de letra
        gr = max(1, int(round(2 * k)))     # Grosor de letra
        pad = int(14 * k)
        paso = int(30 * k)

        ancho_i = int(250 * k)
        alto_i = pad * 2 + int(20 * k) + paso * 4
        xi, yi = int(10 * k), int(10 * k)

        # Fondo negro semitransparente
        fondo = frame.copy()
        cv2.rectangle(fondo, (xi, yi), (xi + ancho_i, yi + alto_i),
                      (0, 0, 0), -1)
        frame = cv2.addWeighted(fondo, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "CONTEO TOTAL", (xi + pad, yi + pad + int(14 * k)),
                    letra, fs * 0.8, (200, 200, 200), gr, cv2.LINE_AA)
        cb = self.conteo_global["bache"]
        n_det = sum(self.conteo_global["deterioro"].values())
        filas = [("Grandes", cb["GRANDE"], CV_ROJO),
                 ("Medianos", cb["MEDIO"], CV_AMARILLO),
                 ("Pequenos", cb["PEQUENO"], CV_AZUL),
                 ("Deterioros", n_det, CV_DETERIORO)]
        y = yi + pad + int(20 * k) + paso
        lado = int(14 * k)
        for nombre, n, col in filas:
            # Cuadradito de color + nombre + número a la derecha
            cv2.rectangle(frame, (xi + pad, y - lado), (xi + pad + lado, y),
                          col, -1)
            cv2.putText(frame, nombre, (xi + pad + int(24 * k), y),
                        letra, fs, (255, 255, 255), gr, cv2.LINE_AA)
            num = str(n)
            (tw, _), _ = cv2.getTextSize(num, letra, fs * 1.1, gr + 1)
            cv2.putText(frame, num, (xi + ancho_i - pad - tw, y),
                        letra, fs * 1.1, col, gr + 1, cv2.LINE_AA)
            y += paso
        return frame

    def _fin_del_video(self):
        """(Hilo) El video llegó al final: se queda pausado en el último
        frame para que se pueda retroceder y volver a ver."""
        self.fin_video = True
        self.pausado = True
        self.frame_idx = max(self.frame_idx, self.total_frames)
        if self.writer is not None or self.ruta_guardar is not None:
            self._cerrar_grabacion()
        self._desde_hilo(self._video_terminado)

    # ═══════════════════════════════════════════════════════════════════
    #  ACTUALIZACIÓN DE LA UI (hilo principal, ~30 veces por segundo)
    # ═══════════════════════════════════════════════════════════════════
    def _tick_ui(self):
        self._mostrar_estado_via()
        if self.corriendo:
            with self.lock:
                frame = self.frame_actual

            # Solo redibujar si hay un frame nuevo
            if frame is not None and frame is not self._ultimo_mostrado:
                self._ultimo_mostrado = frame
                lw = self.lbl_video.winfo_width()
                lh = self.lbl_video.winfo_height()
                img = frame
                if lw > 50 and lh > 50:
                    ih, iw = frame.shape[:2]
                    escala = min(lw / iw, lh / ih)
                    nw, nh = int(iw * escala), int(ih * escala)
                    if nw > 0 and nh > 0:
                        img = cv2.resize(frame, (nw, nh),
                                         interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.foto_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.lbl_video.configure(image=self.foto_tk)

            # Barra de tiempo
            if self.fuente_es_video and self.total_frames > 0:
                actual = min(self.frame_idx, self.total_frames)
                if not self.arrastrando:
                    self.slider.set(actual)
                    self.lbl_tiempo.configure(
                        text=f"{self._formato_tiempo(actual / self.fps_video)}"
                             f" / {self._formato_tiempo(self.total_frames / self.fps_video)}")
            else:
                self.lbl_tiempo.configure(text="● EN VIVO")

        self.after(33, self._tick_ui)

    # ═══════════════════════════════════════════════════════════════════
    #  VIDEO TERMINADO
    # ═══════════════════════════════════════════════════════════════════
    def _video_terminado(self):
        self._log("Video analizado completo", "VIDEO")
        self._refrescar_play()
        self._guardar_csv()
        total = sum(sum(n.values()) for n in self.conteo_global.values())
        if total > 0 and not self.resumen_mostrado:
            self._mostrar_resumen()

    # ═══════════════════════════════════════════════════════════════════
    #  RESUMEN FINAL (ventana emergente)
    # ═══════════════════════════════════════════════════════════════════
    def _mostrar_resumen(self):
        """Ventana con el resumen de lo detectado."""
        self.resumen_mostrado = True
        cg = self.conteo_global
        t_baches = sum(cg["bache"].values())
        t_det = sum(cg["deterioro"].values())

        win = ctk.CTkToplevel(self)
        win.title("Resumen de detección")
        # Centrada sobre la ventana principal
        x = self.winfo_rootx() + (self.winfo_width() - 400) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 420) // 2
        win.geometry(f"400x420+{max(0, x)}+{max(0, y)}")
        win.configure(fg_color=C_FONDO)
        win.resizable(False, False)
        win.transient(self)
        # Esperar un poquito a que se dibuje y traerla al frente
        # (si no, en Windows a veces aparece detrás o da error grab_set)
        def al_frente():
            try:
                win.lift()
                win.focus_force()
                win.grab_set()
            except Exception:
                pass
        win.after(150, al_frente)

        ctk.CTkLabel(win, text="Resumen", font=self._f(22, True),
                     text_color=C_TEXTO).pack(pady=(24, 2))
        ctk.CTkLabel(win, text=f"Sector: {self.sector}", font=self._f(13),
                     text_color=C_TENUE).pack(pady=(0, 14))

        filas = [
            ("Baches en total", t_baches, C_TEXTO),
            ("    Grandes", cg["bache"]["GRANDE"], C_ROJO),
            ("    Medianos", cg["bache"]["MEDIO"], C_AMARILLO),
            ("    Pequeños", cg["bache"]["PEQUENO"], C_AZUL),
            ("Deterioros", t_det, C_MORADO),
        ]
        caja = ctk.CTkFrame(win, fg_color=C_PANEL, corner_radius=10)
        caja.pack(fill="x", padx=28)
        for texto, valor, color in filas:
            f = ctk.CTkFrame(caja, fg_color="transparent")
            f.pack(fill="x", padx=18, pady=5)
            ctk.CTkLabel(f, text=texto, font=self._f(14),
                         text_color=C_TEXTO).pack(side="left")
            ctk.CTkLabel(f, text=str(valor), font=self._f(15, True),
                         text_color=color).pack(side="right")

        ctk.CTkButton(
            win, text="Cerrar", width=140, height=38, font=self._f(13, True),
            fg_color=C_ACENTO, hover_color=C_ACENTO_H, text_color="white",
            corner_radius=8, command=win.destroy).pack(pady=22)

    # ═══════════════════════════════════════════════════════════════════
    #  REGISTRO (log)
    # ═══════════════════════════════════════════════════════════════════
    # Tipos de mensaje del registro: (texto de la columna, color)
    TIPOS_LOG = {
        "SISTEMA":   ("SISTEMA", C_TENUE),
        "VIDEO":     ("VIDEO", C_ACENTO),
        "GUARDADO":  ("GUARDADO", C_VERDE),
        "AVISO":     ("AVISO", C_AMARILLO),
        "ERROR":     ("ERROR", C_ROJO),
        "GRANDE":    ("BACHE", C_ROJO),
        "MEDIO":     ("BACHE", C_AMARILLO),
        "PEQUENO":   ("BACHE", C_AZUL),
        "DETERIORO": ("DETERIORO", C_MORADO),
    }

    def _log(self, mensaje, tipo="SISTEMA", t_video=""):
        """Agrega una línea al registro, en columnas:
        hora | segundo del video | tipo (con color) | mensaje"""
        hora = datetime.now().strftime("%H:%M:%S")
        texto_tipo, _ = self.TIPOS_LOG.get(tipo, (tipo, C_TEXTO))
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{hora}  {t_video:<5}  ", "tenue")
        self.log_box.insert("end", f"{texto_tipo:<10} ", tipo)
        self.log_box.insert("end", f"{mensaje}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _limpiar_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ═══════════════════════════════════════════════════════════════════
    #  AL CERRAR LA VENTANA
    # ═══════════════════════════════════════════════════════════════════
    def _on_cerrar(self):
        self.corriendo = False
        if self.hilo is not None:
            self.hilo.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        self._guardar_csv()     # Que no se pierdan las detecciones
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = DetectorApp()
    app.mainloop()
