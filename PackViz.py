# Requisitos: numpy, matplotlib, ipywidgets (ya viene en Colab)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass

import ipywidgets as widgets
from IPython.display import display, clear_output

# -------------------------------------------------
# Parámetros del contenedor (constantes)
# -------------------------------------------------
# Contenedor (dimensiones internas)
CONT_L, CONT_W, CONT_H = 5898, 2352, 2393  # largo, ancho, alto (mm)

# Cada bloque siempre tiene 4×5×10 cajas
BOXES_PER_LAYER_BLOCK = 4 * 5            # 20 cajas por tendido
BOXES_PER_BLOCK = BOXES_PER_LAYER_BLOCK * 10  # 200 cajas por bloque


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
    poly = Poly3DCollection(cuboid_faces(c), alpha=alpha, edgecolor='k', linewidth=linewidth)
    ax.add_collection3d(poly)

def compute_fit(container_L, container_W, container_H, bl_L, bl_W, bl_H):
    """Cuántos bloques caben en (L, W, H) y sobrantes."""
    nx = container_L // bl_L
    ny = container_W // bl_W
    nz = container_H // bl_H
    remL = container_L - nx * bl_L
    remW = container_W - ny * bl_W
    remH = container_H - nz * bl_H
    return int(nx), int(ny), int(nz), int(nx * ny * nz), (remL, remW, remH)

def block_positions(nx, ny, nz, bl_L, bl_W, bl_H):
    """Orden de llenado: piso por piso (z), luego a lo largo (x), luego a lo ancho (y)."""
    pos = []
    for iz in range(nz):
        for ix in range(nx):
            for iy in range(ny):
                pos.append((ix * bl_L, iy * bl_W, iz * bl_H))
    return pos

# -------------------------------------------------
# Función principal que recibe h
# -------------------------------------------------
def plotContenedor(h):
    """
    h: altura de la caja (mm).
    """

    box_w = 59 + 3*h       # ancho
    box_l = 6*h - 22       # largo
    box_h = h              # alto

    block_L0 = 4 * box_l      # 1112 para h=50
    block_W0 = 5 * box_w      # 1045 para h=50
    block_H0 = 10 * box_h     # 500  para h=50

    # Orientaciones posibles del BLOQUE dentro del contenedor
    ORIENTATIONS = {
        "Orientación 1 ": (block_L0, block_W0, block_H0),
        "Orientación 2 ": (block_W0, block_L0, block_H0),
        "Orientación 3 ":  (block_L0, block_H0, block_W0),
        "Orientación 4 ":  (block_H0, block_L0, block_W0),
        "Orientación 5 ":  (block_W0, block_H0, block_L0),
        "Orientación 6 ":  (block_H0, block_W0, block_L0),
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

    # Estado mutable (dict)
    state = best_state.copy()


    orientation_dd = widgets.Dropdown(
        options=list(ORIENTATIONS.keys()),
        value=best_label,
        description='Orientación:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )

    blocks_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=state["nblocks"],
        step=1,
        description='Bloques:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )

    summary_html = widgets.HTML()
    output_plot = widgets.Output()

    # ---------------------------------------------
    # Funciones internas que usan el estado
    # ---------------------------------------------
    def draw():
        """Redibuja la figura completa según el estado actual."""
        with output_plot:
            clear_output(wait=True)

            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Contenedor Dry — relleno progresivo")

            # contenedor
            add_cuboid(ax, Cuboid(0, 0, 0, CONT_L, CONT_W, CONT_H), alpha=0.03, linewidth=0.6)

            # Número de bloques a dibujar
            n_to_draw = int(max(0, min(blocks_slider.value, state["nblocks"])))

            # Dibuja bloques
            for k in range(n_to_draw):
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

            total_boxes = n_to_draw * BOXES_PER_BLOCK
            remL = CONT_L - state["nx"] * state["bl_L"]
            remW = CONT_W - state["ny"] * state["bl_W"]
            remH = CONT_H - state["nz"] * state["bl_H"]

            summary = (
                f"<b>Orientación bloque:</b> {state['bl_L']}×{state['bl_W']}×{state['bl_H']} mm &nbsp;|&nbsp; "
                f"<b>Capacidad:</b> {state['nx']}×{state['ny']}×{state['nz']} = {state['nblocks']} bloques &nbsp;|&nbsp; "
                f"<b>Dibujados:</b> {n_to_draw} → <b>Cajas:</b> {total_boxes} &nbsp;|&nbsp; "
                f"<b>Sobrante (llenado completo):</b> L={remL} mm, W={remW} mm, H={remH} mm"
            )
            summary_html.value = summary

            plt.show()

    def on_orientation_change(change):
        if change['name'] == 'value':
            label = change['new']
            st = recompute_for(label)
            state.update(st)
            # actualizar rango del slider
            blocks_slider.max = max(state["nblocks"], 1)
            if blocks_slider.value > blocks_slider.max:
                blocks_slider.value = blocks_slider.max
            draw()

    def on_blocks_change(change):
        if change['name'] == 'value':
            draw()

    orientation_dd.observe(on_orientation_change, names='value')
    blocks_slider.observe(on_blocks_change, names='value')

    # Mostrar UI
    ui = widgets.VBox([
        orientation_dd,
        blocks_slider,
        summary_html
    ])

    display(ui, output_plot)

    # dibujo inicial
    draw()
