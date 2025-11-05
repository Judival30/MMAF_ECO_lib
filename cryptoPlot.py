import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle




def cargarDatos(fecha_inicio=None, fecha_fin=None):
    """
    Obtiene los precios de cierre de Bitcoin según tres posibles casos:
     Sin parámetros: usa la fecha actual y obtiene los últimos 30 días.
     Con una sola fecha: usa esa fecha como final y obtiene los últimos 30 días.
     Con rango de fechas: usa ambas fechas proporcionadas.
    """

    try:

        if fecha_inicio is None and fecha_fin is None:
            fecha_fin = datetime.now()
            fecha_inicio = fecha_fin - timedelta(days=30)


        elif fecha_inicio is not None and fecha_fin is None:
            fecha_fin = datetime.strptime(fecha_inicio, "%Y-%m-%d")
            fecha_inicio = fecha_fin - timedelta(days=30)

        elif fecha_inicio is not None and fecha_fin is not None:
            fecha_inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d")
            fecha_fin = datetime.strptime(fecha_fin, "%Y-%m-%d")

        ts_inicio = int(fecha_inicio.timestamp())
        ts_fin = int(fecha_fin.timestamp())

        # URL y parámetros
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': ts_inicio,
            'to': ts_fin
        }

        print(f"Obteniendo datos de Bitcoin desde {fecha_inicio.date()} hasta {fecha_fin.date()}...")
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()

        data = r.json()
        precios = data.get('prices', [])
        if not precios:
            print("No hay datos disponibles en el rango solicitado.")
            return None

        # Procesar datos
        registros = []
        for p in precios:
            fecha = datetime.fromtimestamp(p[0] / 1000)
            valor = p[1]
            registros.append({'fecha': fecha.date(), 'precio_cierre': valor})

        df = pd.DataFrame(registros)
        df = df.groupby('fecha')['precio_cierre'].last().reset_index()
        df['variacion_porcentual'] = df['precio_cierre'].pct_change() * 100

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None


def mostrar_datos_bitcoin(datos):
    """
    Muestra los datos de Bitcoin y genera gráficos con el mismo estilo visual.
    """
    
    if datos is None or datos.empty:
        print("No se pudieron obtener datos.")
        return
    # Gráficos 
    plt.figure(figsize=(14, 8))

    # Subplot 1: precios
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(datos['fecha'], datos['precio_cierre'],
             color='blue', linewidth=2.5, label='Precio de Cierre')
    ax1.scatter(datos['fecha'], datos['precio_cierre'],
                color='red', s=60, zorder=5, label='Puntos de Cierre')
    ax1.set_title('Precio de Cierre de Bitcoin', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Precio (USD)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Subplot 2: variación porcentual
    ax2 = plt.subplot(2, 1, 2)
    colores = ['green' if v >= 0 else 'red' for v in datos['variacion_porcentual'].iloc[1:]]
    barras = ax2.bar(datos['fecha'].iloc[1:], datos['variacion_porcentual'].iloc[1:],
                     color=colores, alpha=0.7, label='Variación Diaria')
    ax2.set_title('Variación Porcentual Diaria', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Variación (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Etiquetas 
    for barra in barras:
        altura = barra.get_height()
        va = 'bottom' if altura >= 0 else 'top'
        color = 'green' if altura >= 0 else 'red'
        ax2.text(barra.get_x() + barra.get_width()/2., altura,
                 f'{altura:+.1f}%', ha='center', va=va, fontweight='bold', color=color)

    plt.tight_layout()
    plt.show()





def mostrar_tabla_bitcoin(datos):
    """
    Muestra una tabla  
    Fecha | Precio de Cierre | Variación %
    """

    if datos is None or datos.empty:
        print("No se pudieron obtener datos para la tabla.")
        return


    tabla = datos.copy()
    tabla['fecha'] = tabla['fecha'].astype(str)
    tabla['precio_cierre'] = tabla['precio_cierre'].apply(lambda x: f"${x:,.2f}")
    tabla['variacion_porcentual'] = tabla['variacion_porcentual'].apply(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
    )

    # Datos y encabezados
    columnas = ['Fecha', 'Precio de Cierre', 'Variación %']
    celdas = tabla.values.tolist()

    #  figura y eje
    fig, ax = plt.subplots(figsize=(10, len(tabla) * 0.45 + 1))
    ax.axis('off')
    fig.patch.set_facecolor("#f5f7fa")

    #  tabla
    tabla_visual = ax.table(
        cellText=celdas,
        colLabels=columnas,
        loc='center',
        cellLoc='center',
        colColours=["#000A66", "#003366", "#003366"],  
        edges='closed'
    )

    # 
    for (row, col), cell in tabla_visual.get_celld().items():
        if row == 0:
            # Encabezado
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#003366')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f0f4f7')
            else:
                cell.set_facecolor('white')
            if col == 2:
                valor_str = tabla.iloc[row - 1]['variacion_porcentual']
                try:
                    valor = float(valor_str.replace('%', '').replace('+', ''))
                    if valor > 0:
                        cell.set_facecolor('#c6f6c6')  
                    elif valor < 0:
                        cell.set_facecolor('#f7c6c6')  
                except:
                    pass

    # Ajustar tamaño y tipografía
    tabla_visual.auto_set_font_size(False)
    tabla_visual.set_fontsize(11)
    tabla_visual.scale(1.2, 1.3)

    # Estilo del título
    ax.set_title(
        "Evolución del Precio de Cierre de Bitcoin",
        fontsize=16,
        fontweight='bold',
        color="#1a1a1a",
        pad=25
    )
    ax.add_patch(Rectangle(
        (-0.05, -0.05), 1.1, 1.1,
        transform=ax.transAxes,
        color='grey', alpha=0.05, zorder=-1
    ))

    plt.tight_layout()
    plt.show()


# mostrar_datos_bitcoin()  # Caso 1: sin parámetros → últimos 30 días
# mostrar_datos_bitcoin("2025-10-01")  # Caso 2: una sola fecha
# mostrar_datos_bitcoin("2025-09-01", "2025-09-25")  # Caso 3: rango de fechas
#mostrar_datos_bitcoin()