#@title Ocultar código
import math
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 1. Función para resolver x
# ===============================

def resolver_para_area(A):
    """
    Resuelve la ecuación:
        (8x + 40)(7x + 40) = A
    que proviene del modelo geométrico del problema.
    Retorna (x, ancho W, largo L, área calculada) o None si no tiene solución.
    """
    a = 56.0
    b = 600.0
    c = 1600.0 - float(A)

    D = b*b - 4*a*c
    if D < 0:
        return None

    sqrtD = math.sqrt(D)
    x1 = (-b + sqrtD) / (2*a)
    x2 = (-b - sqrtD) / (2*a)

    # Seleccionar raíz válida
    x = None
    for r in (x1, x2):
        if r >= 0:
            x = r
            break

    if x is None:
        return None

    W = 8*x + 40
    L = 7*x + 40
    area_calc = W * L

    return x, W, L, area_calc


# ===============================
# 2. Solicitud de varias áreas
# ===============================

def solicitar_areas():

  texto = input("Ingrese áreas separadas por comas (ej: 2000,3000,5000): ")
  areas = [int(a.strip()) for a in texto.split(",") if a.strip().isdigit()]

  # ===============================
  # 3. Construcción de tabla
  # ===============================

  filas = []

  for A in areas:
      res = resolver_para_area(A)
      if res is None:
          filas.append({
              "Área solicitada": A,
              "x": None,
              "Ancho W": None,
              "Largo L": None,
              "Distancia lateral interna (x)": None,
              "Distancia frontal entre filas (3x)": None,
              "Margen lateral del lote (2x)": None,
              "Margen posterior del lote (2x)": None,
              "Margen frontal del lote (2x)": None,
              "Área calculada": None
          })
      else:
          x, W, L, area_calc = res
          filas.append({
              "Área solicitada": A,
              "x": x,
              "Ancho W": W,
              "Largo L": L,
              "Distancia lateral interna (x)": x,
              "Distancia frontal entre filas (3x)": 3*x,
              "Margen lateral del lote (2x)": 2*x,
              "Margen posterior del lote (2x)": 2*x,
              "Margen frontal del lote (2x)": 2*x,
              "Área calculada": area_calc
          })

  df = pd.DataFrame(filas)

  
  # ============================================
  # 4. Guardar la tabla en Excel
  # ============================================

  df.to_excel("tabla_lote_con_distancias.xlsx", index=False)
  #print("\nArchivo generado: tabla_lote_con_distancias.xlsx")
  # ============================================
  # 5. Gráfica opcional: Área vs x  (TU GRÁFICA ORIGINAL, TÍTULO MÁS CORTO)
  # ============================================

  if not df.empty:
      plt.figure(figsize=(8,6))
      plt.plot(df["Área solicitada"], df["x"])
      plt.xlabel("Área del lote (m²)")
      plt.ylabel("Distancia lateral x (m)")
      plt.title("Área del lote vs distancia x", pad=15)
      plt.grid(True)
      plt.savefig("grafica_area_vs_x.png", dpi=300, bbox_inches="tight")
      print("\nGráfica generada: grafica_area_vs_x.png")
      plt.show()
  else:
      print("\nNo hay datos válidos para generar la gráfica.")

  # ============================================
  # 6. Visualización estética de la TABLA (mismas filas, títulos más cortos)
  # ============================================

  if not df.empty:
      df_to_show = df.copy()

      # Redondeo suave para columnas numéricas
      numeric_cols = df_to_show.select_dtypes(include="number").columns
      df_to_show[numeric_cols] = df_to_show[numeric_cols].round(2)

      # Renombrar columnas con títulos más cortos para que no se monten
      col_renames = {
          "Área solicitada": "Área lote (m²)",
          "Ancho W": "Ancho W (m)",
          "Largo L": "Largo L (m)",
          "Distancia lateral interna (x)": "Dist. lateral (x)",
          "Distancia frontal entre filas (3x)": "Dist. frontal (3x)",
          "Margen lateral del lote (2x)": "Margen lat. (2x)",
          "Margen posterior del lote (2x)": "Margen post. (2x)",
          "Margen frontal del lote (2x)": "Margen front. (2x)",
          "Área calculada": "Área calc. (m²)"
      }
      df_to_show.rename(columns=col_renames, inplace=True)

      # Tamaño de la figura según filas/columnas
      n_rows, n_cols = df_to_show.shape
      fig_width = min(1 + n_cols * 1.2, 20)
      fig_height = min(1 + n_rows * 0.5, 15)

      fig, ax = plt.subplots(figsize=(fig_width, fig_height))
      ax.axis("off")

      tabla = ax.table(
          cellText=df_to_show.values,
          colLabels=df_to_show.columns,
          cellLoc="center",
          loc="center"
      )

      tabla.auto_set_font_size(False)
      tabla.set_fontsize(9)
      tabla.scale(1, 1.4)

      plt.title("Parámetros del lote", pad=16)
      plt.tight_layout()
      plt.show()
  else:
      print("No hay datos válidos para graficar la tabla.")
