import pandas as pd
import plotly.graph_objects as go
import colorsys
import matplotlib.colors as mcolors

def lighten_color(hex_color, increase=0.1):
    """
    Erhöht die Helligkeit einer Farbe um den angegebenen Prozentsatz.

    :param hex_color: Hex-Code der Farbe (z.B. '#f03b3e')
    :param increase: Erhöhungsfaktor (z.B. 0.1 für 10%)
    :return: Hex-Code der aufgehellten Farbe
    """
    rgb = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = min(1, l + increase * l)  # Sicherstellen, dass l nicht über 1 geht
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex((r, g, b))

# Originale Hex-Farben
original_red = '#23c93f'
original_green = '#ee1111'

# Aufgehellte Farben
light_red = lighten_color(original_red, 0.1)     # 10% heller Rot
light_green = lighten_color(original_green, 0.1) # 10% heller Grün

print(f"Aufgehelltes Rot: {light_red}")
print(f"Aufgehelltes Grün: {light_green}")

# Definieren der benutzerdefinierten Farbskala mit aufgehellten Farben
colorscale = [
    [0.0, light_red],    # Start bei aufgehelltem Rot
    [1.0, light_green]   # Ende bei aufgehelltem Grün
]

# CSV-Datei laden
file_path = "src/models/lstm_param_results.csv"
try:
    df = pd.read_csv(file_path)
    if 'Val Loss' not in df.columns:
        raise ValueError("Die Spalte 'Val Loss' fehlt in der CSV-Datei.")
except Exception as e:
    print(f"Fehler beim Laden der CSV-Datei: {e}")
    df = None

if df is not None:
    # Sicherstellen, dass nur die gewünschten learning_rate-Werte vorhanden sind
    desired_lrs = [0.0001, 0.001, 0.01]
    df = df[df['learning_rate'].isin(desired_lrs)]

    # Konvertieren zu String für konsistente Beschriftung
    df['learning_rate'] = df['learning_rate'].astype(str)

    # Kombinieren von 'hidden_size' und 'num_layers' für die y-Achse
    df['hidden_num_layers'] = df.apply(lambda row: f"HS:{row['hidden_size']} NL:{row['num_layers']}", axis=1)

    # Pivot-Tabelle erstellen
    pivot_df = df.pivot_table(index='hidden_num_layers',
                              columns='learning_rate',
                              values='Val Loss')

    # Sicherstellen, dass die Spalten in der gewünschten Reihenfolge sind
    desired_lrs_str = ['0.0001', '0.001', '0.01']
    pivot_df = pivot_df.reindex(columns=desired_lrs_str)

    # Dynamische Festlegung von zmin und zmax basierend auf den Quantilen
    zmin = 25
    zmax = 37

    print(f"Dynamische Val Loss - zmin: {zmin}, zmax: {zmax}")

    # Erstellen der Heatmap mit angepasster Farbskala und zmin/zmax
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title='Val Loss'),
        text=pivot_df.values,  # Hinzufügen der Val Loss Werte als Text
        texttemplate="%{text:.2f}",  # Formatierung der angezeigten Werte (2 Dezimalstellen)
        textfont={"size": 12, "color": "black"},  # Textstil anpassen (schwarz für bessere Lesbarkeit)
        hovertemplate='Val Loss: %{z}<br>Learning Rate: %{x}<br>Hidden/Layer: %{y}<extra></extra>',
        xgap=2,  # Horizontale Lücke zwischen den Zellen
        ygap=2   # Vertikale Lücke zwischen den Zellen
    ))

    # Anpassen des Layouts
    fig.update_layout(
        title='Heatmap des Val Loss über Parameterkombinationen',
        xaxis_title='Learning Rate',
        yaxis_title='Hidden Size / Num Layers',
        xaxis=dict(tickmode='array', tickvals=list(range(len(desired_lrs_str))), ticktext=desired_lrs_str),
        yaxis=dict(tickmode='array', tickvals=list(range(len(pivot_df.index))), ticktext=pivot_df.index),
        template='plotly_white',
        width=800,
        height=600
    )

    fig.show()
else:
    print("Heatmap kann nicht generiert werden. Bitte überprüfe deine CSV-Datei.")
