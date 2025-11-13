"""
Visualización interactiva de bloques (estibas) en un contenedor Dry 20'.

Uso típico (en Colab o Jupyter):
import requests

url = "https://raw.githubusercontent.com/Judival30/MMAF_ECO_lib/main/PackViz.py"
exec(requests.get(url).text)

# Esto viene definido dentro del archivo .py
mostrar_contenedor_interactivo()

"""

import sys
import subprocess

# ============================================================
# INSTALACIÓN AUTOMÁTICA DE LIBRERÍAS
# ============================================================

# (nombre_para_pip, nombre_para_import)
_REQUIRED_LIBS = [
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("ipywidgets", "ipywidgets"),
]

def _install_libraries():
    for pip_name, import_name in _REQUIRED_LIBS:
        try:
            __import__(import_name)
        except ImportError:
            print(f"Instalando {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

_install_libraries()

# ============================================================
# IMPORTS PRINCIPALES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
import ipywidgets as widgets
from IPython.display import display

# Habilitar manejo de widgets en Google Colab, si existe
try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

# ============================================================
# PARÁMETROS DEL PROBLEMA
# ============================================================

# Caja: h = 50 mm; ancho = 59 + 3h; largo = 6h - 22
h = 50  # mm
box_w = 59 + 3*h          # ancho  = 59 + 3h = 209
box_l = 6*h - 22          # largo  = 6h - 22 = 278
box_h = h                 # alto   = 50

# Bloque (estiba): 10 alto, 5 ancho, 4 largo (en múltiplos de la caja)
block_L0 = 4 * box_l      # 1112
block_W0 = 5 * box_w      # 1045
block_H0 = 10 * box_h     # 500

# Contenedor (dimensiones internas) Dry 20'
CONT_L, CONT_W, CONT_H = 5898, 2352, 2393  # largo, ancho, alto (mm)

BOXES_PER_LAYER_BLOCK = 4 * 5      # 20 cajas por tendido
BOXES_PER_BLOCK = BOXES_PER_LAYER_BLOCK * 10  # 200 cajas por bloque

# Orientaciones posibles del BLOQUE dentro del contenedor
ORIENTATIONS = {
    "Orientación 1 (1112×1045×500)": (block_L0, block_W0, block_H0),
    "Orientación 2 (1045×1112×500)": (block_W0, block_L0, block_H0),
    "Orientación 3 (1112×500×1045)": (block_L0, block_H0, block_W0),
    "Orientación 4 (500×1112×1045)": (block_H0, block_L0, block_W0),
    "Orientación 5 (1045×500×1112)": (block_W0, block_H0, block_L0),
    "Orientación 6 (500×1045×1112)": (block_H0, block_W0, block_L0),
}

# ============================================================
# UTILIDADES GEOMÉTRICAS
# ============================================================

@dataclass
class Cuboid:
    x: float
    y: float
    z: float
    L: float
    W: float
    H: float

def cuboid_faces(c: Cuboid):
    x, y, z = c.x, c.y, c.z
    X = [x, x + c.L]
    Y = [y, y + c.W]
    Z = [z, z + c.H]
    v = np.array([
        [X[0], Y[0], Z[0]], [X[1], Y[0], Z[0]], [X[1], Y[1], Z[0]], [X[0], Y[1], Z[0]],
        [X[0], Y[0], Z[1]], [X[1], Y[0], Z[1]], [X[1], Y[1], Z[1]], [X[0], Y[1], Z[1]]
    ])
    return [
        [v[0], v[1], v[2], v[3]],  # base
        [v[4], v[5], v[6], v[7]],  # techo
        [v[0], v[1], v[5], v[4]],  # frente
        [v[2], v[3], v[7], v[6]],  # atrás
        [v[1], v[2], v[6], v[5]],  # derecha
        [v[3], v[0], v[4], v[7]],  # izquierda
    ]

def add_cuboid(ax, c: Cuboid, alpha=0.18, linewidth=0.6):
    poly = Poly3DCollection(
        cuboid_faces(c),
        alpha=alpha,
        edgecolor='k',
        linewidth=linewidth
    )
    ax.add_collection3d(poly)

def compute_fit(container_L, container_W, container_H, bl_L, bl_W, bl_H):
    """Cuántos bloques caben en (L, W, H) y sobrantes."""
    nx = container_L // bl_L
    ny = container_W // bl_W
    nz = container_H // bl_H
    remL = container_L - nx * bl_L
    remW = container_W - ny * bl_W
    remH = container_H - nz * bl_H
    return int(nx), int(ny), int(nz), int(nx*ny*nz), (remL, remW, remH)

def block_positions(nx, ny, nz, bl_L, bl_W, bl_H):
    """Orden de llenado: piso por piso (z), luego a lo largo (x), luego a lo ancho (y)."""
    pos = []
    for iz in range(nz):           # altura
        for ix in range(nx):       # largo
            for iy in range(ny):   # ancho
                pos.append((ix*bl_L, iy*bl_W, iz*bl_H))
    return pos

# ============================================================
# CÁLCULO DE LA MEJOR ORIENTACIÓN (MÁS BLOQUES)
# ============================================================

best_label = None
best_state = None

for label, (bL, bW, bH) in ORIENTATIONS.items():
    nx, ny, nz, nblocks, rem = compute_fit(CONT_L, CONT_W, CONT_H, bL, bW, bH)
    if (best_state is None) or (nblocks > best_state["nblocks"]):
        best_label = label
        best_state = {
            "bl_L": bL, "bl_W": bW, "bl_H": bH,
            "nx": nx, "ny": ny, "nz": nz,
            "nblocks": nblocks,
            "positions": block_positions(nx, ny, nz, bL, bW, bH),
            "rem": rem,
        }

def recompute_for(label):
    bL, bW, bH = ORIENTATIONS[label]
    nx, ny, nz, nblocks, rem = compute_fit(CONT_L, CONT_W, CONT_H, bL, bW, bH)
    positions = block_positions(nx, ny, nz, bL, bW, bH)
    return {
        "bl_L": bL, "bl_W": bW, "bl_H": bH,
        "nx": nx, "ny": ny, "nz": nz,
        "nblocks": nblocks,
        "positions": positions,
        "rem": rem,
    }

# ============================================================
# FUNCIÓN PRINCIPAL PÚBLICA
# ============================================================

def mostrar_contenedor_interactivo():
    """
    Muestra una visualización interactiva del contenedor Dry 20'
    con bloques (estibas) usando ipywidgets (Jupyter/Colab).

    Requisitos:
      - Ejecutar en Jupyter Notebook / JupyterLab / Google Colab.
      - Librerías: numpy, matplotlib, ipywidgets (se instalan automáticamente).
    """

    # ------------------------------
    # Figura base
    # ------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Eje para texto resumen arriba
    summary_ax = fig.add_axes([0.02, 0.88, 0.96, 0.1])
    summary_ax.axis('off')
    summary_text = summary_ax.text(
        0.01, 0.5, "",
        fontsize=10,
        va='center',
        ha='left',
        family='monospace'
    )

    # Estado actual de orientación / bloques
    state = best_state.copy()

    def redraw(n_to_draw: int):
        """Redibuja el contenedor y los primeros n_to_draw bloques."""
        ax.cla()
        ax.set_title("Contenedor Dry 20' — relleno progresivo")

        # contenedor vacío
        add_cuboid(ax, Cuboid(0, 0, 0, CONT_L, CONT_W, CONT_H), alpha=0.03, linewidth=0.6)

        # Dibuja los primeros n_to_draw bloques
        n_to_draw_clamped = int(max(0, min(n_to_draw, state["nblocks"])))
        for k in range(n_to_draw_clamped):
            x0, y0, z0 = state["positions"][k]
            add_cuboid(ax, Cuboid(x0, y0, z0, state["bl_L"], state["bl_W"], state["bl_H"]),
                       alpha=0.22, linewidth=0.5)

        ax.set_xlabel("Largo (mm)")
        ax.set_ylabel("Ancho (mm)")
        ax.set_zlabel("Alto (mm)")
        ax.set_xlim(0, CONT_L)
        ax.set_ylim(0, CONT_W)
        ax.set_zlim(0, CONT_H)
        ax.view_init(elev=22, azim=35)

        total_boxes = n_to_draw_clamped * BOXES_PER_BLOCK
        remL = CONT_L - state["nx"] * state["bl_L"]
        remW = CONT_W - state["ny"] * state["bl_W"]
        remH = CONT_H - state["nz"] * state["bl_H"]
        summary = (
            f"Orientación bloque: {state['bl_L']}×{state['bl_W']}×{state['bl_H']} mm  |  "
            f"Cap: {state['nx']}×{state['ny']}×{state['nz']} = {state['nblocks']} bloques  |  "
            f"Dibujados: {n_to_draw_clamped} → Cajas: {total_boxes}  |  "
            f"Sobrante (llenado completo): L={remL} W={remW} H={remH} mm"
        )
        summary_text.set_text(summary)

    # ------------------------------
    # Widgets interactivos
    # ------------------------------
    orient_widget = widgets.Dropdown(
        options=list(ORIENTATIONS.keys()),
        value=best_label,
        description='Orientación:',
        layout=widgets.Layout(width='450px')
    )

    bloques_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=best_state["nblocks"],
        step=1,
        description='Bloques:',
        continuous_update=False,
        layout=widgets.Layout(width='450px')
    )

    def actualizar(orientacion, bloques):
        # Recalcular orientación según selección
        st = recompute_for(orientacion)
        state.update(st)

        # Ajustar el máximo del slider a la nueva capacidad
        bloques_widget.max = max(state["nblocks"], 1)
        if bloques > bloques_widget.max:
            bloques = bloques_widget.max
            bloques_widget.value = bloques

        # Redibujar figura
        redraw(bloques)

        # En Colab/Jupyter inline, hay que volver a mostrar la figura
        plt.close(fig)
        display(fig)

    out = widgets.interactive_output(
        actualizar,
        {'orientacion': orient_widget, 'bloques': bloques_widget}
    )

    # Mostrar controles y salida
    display(orient_widget, bloques_widget, out)

    # Dibujo inicial
    redraw(0)
    plt.close(fig)
    display(fig)

mostrar_contenedor_interactivo()
