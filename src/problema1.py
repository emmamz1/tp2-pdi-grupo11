import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from collections import Counter


# =====================================================================
# 1. CARGA DE IMAGEN
# =====================================================================

# Se prueban varias rutas habituales para no depender de la carpeta actual
RUTAS_POSIBLES = [
    os.path.join("..", "images", "monedas.jpg"),
    os.path.join("images", "monedas.jpg"),
    "monedas.jpg",
]

imagen = None
for ruta in RUTAS_POSIBLES:
    imagen = cv2.imread(ruta)
    if imagen is not None:
        print("Imagen cargada desde:", ruta)
        break

if imagen is None:
    raise FileNotFoundError(
        "No se pudo cargar la imagen. Rutas probadas: " + str(RUTAS_POSIBLES)
    )

# OpenCV trabaja en BGR, Matplotlib utiliza RGB
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

alto, ancho = imagen.shape[:2]

print("Dimensiones:", ancho, "x", alto)

# Factor de escala: los tamaños de kernel y áreas están calibrados para una
# imagen de 2576 px de ancho; así el script funciona con otras resoluciones.
esc = ancho / 2576


def impar(k):
    """Tamaño de kernel escalado, siempre impar y >= 3."""
    return max(3, int(k * esc) | 1)


plt.figure(figsize=(12, 7))
plt.imshow(imagen_rgb)
plt.title("Imagen original")
plt.axis("off")
plt.show()


# =====================================================================
# 2. ESCALA DE GRISES Y ESPACIO HSV
# =====================================================================

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

H_canal = hsv[:, :, 0]
S_canal = hsv[:, :, 1]
V_canal = hsv[:, :, 2]

plt.figure(figsize=(12, 7))
plt.imshow(gris, cmap="gray")
plt.title("Imagen en escala de grises")
plt.axis("off")
plt.show()


# =====================================================================
# 3. HISTOGRAMA
# =====================================================================

histograma = cv2.calcHist([gris], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 5))
plt.plot(histograma)
plt.title("Histograma de intensidades")
plt.xlabel("Intensidad")
plt.ylabel("Cantidad de píxeles")
plt.grid()
plt.show()


# =====================================================================
# 4. ESTIMACIÓN DEL FONDO (intensidad no uniforme)
# =====================================================================
# Mediana de gran tamaño sobre la imagen reducida: los objetos (más chicos
# que la ventana) desaparecen y queda solamente el fondo.

chica = cv2.resize(gris, None, fx=0.25, fy=0.25)
fondo = cv2.resize(
    cv2.medianBlur(chica, impar(101)),
    (ancho, alto)
)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))

ax[0].imshow(gris, cmap="gray")
ax[0].set_title("Original")

ax[1].imshow(fondo, cmap="gray")
ax[1].set_title("Fondo estimado")

ax[2].imshow(cv2.absdiff(gris, fondo), cmap="gray")
ax[2].set_title("|Original - Fondo|")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


# =====================================================================
# 5. DETECCIÓN DE DADOS (primero)
# =====================================================================
# Los puntos de la cara superior de un dado son manchas negras rodeadas de
# blanco. Las monedas no tienen ese patrón: el entorno de sus zonas oscuras
# es gris. Los puntos de las caras laterales están en sombra y por eso
# no tienen anillo blanco: solo se cuentan los de la cara superior.

UMBRAL_NEGRO = 60          # V < 60 -> píxel negro
UMBRAL_ANILLO_BLANCO = 185  # V medio del entorno del punto

oscuro = (V_canal < UMBRAL_NEGRO).astype(np.uint8) * 255
oscuro = cv2.morphologyEx(
    oscuro,
    cv2.MORPH_OPEN,
    np.ones((3, 3), np.uint8)
)

cont_oscuros, _ = cv2.findContours(
    oscuro,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

puntos = []   # (cx, cy, contorno)

for c in cont_oscuros:

    area = cv2.contourArea(c)

    if not (200 * esc ** 2 < area < 4000 * esc ** 2):
        continue

    mk = np.zeros((alto, ancho), np.uint8)
    cv2.drawContours(mk, [c], -1, 255, cv2.FILLED)

    # Anillo alrededor de la mancha
    anillo = (
        cv2.dilate(mk, np.ones((impar(31),) * 2, np.uint8))
        - cv2.dilate(mk, np.ones((impar(11),) * 2, np.uint8))
    )

    if V_canal[anillo > 0].mean() > UMBRAL_ANILLO_BLANCO:
        M = cv2.moments(c)
        puntos.append((M["m10"] / M["m00"], M["m01"] / M["m00"], c))

print("Puntos de dado detectados:", len(puntos))

# Región blanca de cada dado (blanco RELATIVO al fondo local, para que el
# fondo claro del lado derecho no se confunda con el dado)
mask_puntos = np.zeros((alto, ancho), np.uint8)
for _, _, c in puntos:
    cv2.drawContours(mask_puntos, [c], -1, 255, cv2.FILLED)

blanco = (
    ((gris.astype(np.int16) - fondo.astype(np.int16) > 35) & (S_canal < 60))
    .astype(np.uint8) * 255
) | mask_puntos

blanco = cv2.morphologyEx(
    blanco,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (impar(25),) * 2)
)

blanco = cv2.morphologyEx(
    blanco,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (impar(15),) * 2)
)

n_comp, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(blanco)

dados = []
resultados_dados = []
mascara_dados = np.zeros((alto, ancho), np.uint8)

for i in range(1, n_comp):

    region = etiquetas == i
    area_region = estadisticas[i, cv2.CC_STAT_AREA]

    # Puntos de dado que caen dentro de esta región blanca
    pts = [
        (x, y) for x, y, _ in puntos
        if region[int(y), int(x)]
    ]

    if len(pts) >= 1 and area_region > 3000 * esc ** 2:

        m = ndi.binary_fill_holes(region).astype(np.uint8) * 255
        mascara_dados |= m

        cs, _ = cv2.findContours(
            m,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        c = max(cs, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        dados.append({
            "contorno": c,
            "area": cv2.contourArea(c),
            "perimetro": cv2.arcLength(c, True),
            "circularidad": 0.0,
            "x": x, "y": y, "w": w, "h": h,
            "aspecto": w / h,
            "tipo": "Dado",
            "centros": pts,
            "valor": len(pts)
        })

        resultados_dados.append(len(pts))

print("Dados detectados:", len(dados))
print("Valores de la cara superior:", resultados_dados)

plt.figure(figsize=(12, 7))
plt.imshow(mascara_dados, cmap="gray")
plt.title("Máscara de dados")
plt.axis("off")
plt.show()

# Se retiran los dados (dilatados, para eliminar también su sombra)
mascara_dados_dil = cv2.dilate(
    mascara_dados,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (impar(61),) * 2)
)


# =====================================================================
# 6. SEGMENTACIÓN DE MONEDAS (sin los dados)
# =====================================================================
# Ninguna cue sola alcanza:
#   - diferencia con el fondo: falla en monedas de intensidad similar al fondo
#   - gradiente (bordes):      falla en monedas doradas lisas
#   - saturación:              solo detecta las monedas doradas
# Se combinan las tres.

# Cue 1: diferencia respecto al fondo
dif = cv2.absdiff(
    cv2.GaussianBlur(gris, (impar(7),) * 2, 0),
    fondo
)
_, cue_fondo = cv2.threshold(dif, 25, 255, cv2.THRESH_BINARY)

# Cue 2: magnitud del gradiente + Otsu
suave = cv2.GaussianBlur(gris, (impar(5),) * 2, 0)
mag = cv2.magnitude(
    cv2.Sobel(suave, cv2.CV_32F, 1, 0),
    cv2.Sobel(suave, cv2.CV_32F, 0, 1)
)
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
umbral_otsu, cue_bordes = cv2.threshold(
    mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Cue 3: saturación (monedas doradas)
_, cue_color = cv2.threshold(
    cv2.GaussianBlur(S_canal, (impar(7),) * 2, 0),
    60, 255, cv2.THRESH_BINARY
)

print("Umbral Otsu (gradiente):", umbral_otsu)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))

ax[0].imshow(cue_fondo, cmap="gray")
ax[0].set_title("Cue 1: diferencia con fondo")

ax[1].imshow(cue_bordes, cmap="gray")
ax[1].set_title("Cue 2: bordes (gradiente + Otsu)")

ax[2].imshow(cue_color, cmap="gray")
ax[2].set_title("Cue 3: saturación")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

# Unión de cues y eliminación de los dados
mascara = cue_fondo | cue_bordes | cue_color
mascara[mascara_dados_dil > 0] = 0

# Cierre: une los bordes de cada moneda
mascara = cv2.morphologyEx(
    mascara,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (impar(21),) * 2)
)

# Relleno de huecos (interior de las monedas)
mascara_rellena = ndi.binary_fill_holes(mascara > 0).astype(np.uint8) * 255

# Apertura fuerte: elimina ruido y restos finos de sombra
mascara_rellena = cv2.morphologyEx(
    mascara_rellena,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (impar(41),) * 2)
)

plt.figure(figsize=(12, 7))
plt.imshow(mascara_rellena, cmap="gray")
plt.title("Máscara de monedas (sin dados)")
plt.axis("off")
plt.show()


# =====================================================================
# 7. DETECCIÓN DE CONTORNOS DE MONEDAS
# =====================================================================

contornos_monedas, _ = cv2.findContours(
    mascara_rellena,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

AREA_MINIMA = 8000 * esc ** 2
UMBRAL_CIRCULARIDAD = 0.80   # filtro de seguridad contra restos de sombra

monedas = []

for contorno in contornos_monedas:

    area = cv2.contourArea(contorno)

    if area < AREA_MINIMA:
        continue

    perimetro = cv2.arcLength(contorno, True)

    if perimetro == 0:
        continue

    circularidad = (4 * np.pi * area) / (perimetro ** 2)

    if circularidad < UMBRAL_CIRCULARIDAD:
        print(
            f"Descartado (circularidad {circularidad:.2f}, "
            f"area {area:.0f})"
        )
        continue

    x, y, w, h = cv2.boundingRect(contorno)

    monedas.append({
        "contorno": contorno,
        "area": area,
        "perimetro": perimetro,
        "circularidad": circularidad,
        "x": x, "y": y, "w": w, "h": h,
        "aspecto": w / h,
        "tipo": "Moneda"
    })

objetos = monedas + dados

print("Monedas detectadas:", len(monedas))
print("Dados detectados:", len(dados))
print("Objetos detectados:", len(objetos))


# Visualización de la segmentación

imagen_clasificada = imagen_rgb.copy()

for objeto in monedas:

    cv2.drawContours(
        imagen_clasificada, [objeto["contorno"]], -1, (0, 255, 0), 3
    )

    cv2.putText(
        imagen_clasificada,
        "Moneda",
        (objeto["x"], objeto["y"] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

for objeto in dados:

    cv2.drawContours(
        imagen_clasificada, [objeto["contorno"]], -1, (255, 0, 0), 3
    )

    cv2.putText(
        imagen_clasificada,
        "Dado",
        (objeto["x"], objeto["y"] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

plt.figure(figsize=(14, 8))
plt.imshow(imagen_clasificada)
plt.title("Segmentación: monedas (verde) y dados (azul)")
plt.axis("off")
plt.show()


# =====================================================================
# 8. CLASIFICACIÓN DE MONEDAS
# =====================================================================

if len(monedas) < 1:
    raise ValueError("No se detectaron monedas")

caracteristicas = []

for i, moneda in enumerate(monedas):

    x = moneda["x"]
    y = moneda["y"]
    w = moneda["w"]
    h = moneda["h"]

    contorno = moneda["contorno"]
    area = moneda["area"]

    # Diámetro equivalente
    diametro = np.sqrt((4 * area) / np.pi)

    # Recorte de la moneda
    roi_hsv = hsv[y:y + h, x:x + w]

    # Máscara para usar solamente los píxeles de la moneda
    mascara_roi = np.zeros((h, w), dtype=np.uint8)

    contorno_local = contorno - np.array([[[x, y]]])

    cv2.drawContours(
        mascara_roi, [contorno_local], -1, 255, cv2.FILLED
    )

    pixeles = roi_hsv[mascara_roi > 0]

    # Características de color
    media_h = np.mean(pixeles[:, 0])
    media_s = np.mean(pixeles[:, 1])

    # Píxeles dorados
    dorados = (
        (pixeles[:, 0] >= 5) &
        (pixeles[:, 0] <= 35) &
        (pixeles[:, 1] > 45)
    )

    proporcion_dorado = np.mean(dorados)

    caracteristicas.append([
        diametro,
        proporcion_dorado,
        media_h,
        media_s
    ])


caracteristicas = np.array(caracteristicas)

print("Características extraídas (diametro, prop. dorado, H medio, S medio):")
print(caracteristicas)


# Normalización de características
escalador = StandardScaler()
X = escalador.fit_transform(caracteristicas)

# K-Means
N_GRUPOS = min(4, len(monedas))

kmeans = KMeans(
    n_clusters=N_GRUPOS,
    random_state=42,
    n_init=10
)

grupos = kmeans.fit_predict(X)

# Ordenamos los grupos por diámetro promedio
diametros_por_grupo = {}

for grupo in range(N_GRUPOS):
    indices = np.where(grupos == grupo)[0]
    diametros_por_grupo[grupo] = np.mean(caracteristicas[indices, 0])

orden_grupos = sorted(diametros_por_grupo, key=diametros_por_grupo.get)

# Etiquetas estables: Grupo 1 (más chico), Grupo 2, ...
mapa_grupos = {
    grupo_original: posicion + 1
    for posicion, grupo_original in enumerate(orden_grupos)
}

grupos_ordenados = np.array([mapa_grupos[g] for g in grupos])

for i, moneda in enumerate(monedas):
    moneda["grupo"] = int(grupos_ordenados[i])
    print(f"Moneda {i + 1}: Grupo {moneda['grupo']}")


# Dibujamos los grupos sobre la imagen

imagen_monedas = imagen_rgb.copy()

for i, moneda in enumerate(monedas):

    cv2.drawContours(
        imagen_monedas, [moneda["contorno"]], -1, (0, 255, 0), 3
    )

    cv2.putText(
        imagen_monedas,
        f"G{moneda['grupo']}",
        (moneda["x"], moneda["y"] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        3
    )

plt.figure(figsize=(14, 8))
plt.imshow(imagen_monedas)
plt.title("Clasificación de monedas mediante K-Means")
plt.axis("off")
plt.show()


# Conteo de cada grupo
conteo_grupos = Counter(grupos_ordenados)

print("\nConteo de monedas por grupo:")

for grupo, cantidad in sorted(conteo_grupos.items()):
    print(f"Grupo {grupo}: {cantidad} monedas")


# =====================================================================
# 9. RECONOCIMIENTO DE DADOS (cara superior)
# =====================================================================
# El valor de cada dado es la cantidad de puntos negros con anillo blanco
# contenidos en su región (los de las caras laterales no cumplen esa
# condición porque están en sombra).

imagen_dados = imagen_rgb.copy()

for i, dado in enumerate(dados):

    cv2.drawContours(
        imagen_dados, [dado["contorno"]], -1, (255, 0, 0), 3
    )

    for cx, cy in dado["centros"]:
        cv2.circle(
            imagen_dados, (int(cx), int(cy)), 14, (0, 255, 0), 3
        )

    cv2.putText(
        imagen_dados,
        f"Dado {i + 1}: {dado['valor']}",
        (dado["x"], dado["y"] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        3
    )

    print(f"Dado {i + 1}: {dado['valor']} puntos")

plt.figure(figsize=(14, 8))
plt.imshow(imagen_dados)
plt.title("Reconocimiento de las caras superiores de los dados")
plt.axis("off")
plt.show()


# =====================================================================
# 10. RESULTADOS FINALES
# =====================================================================

print("\n========== RESULTADOS FINALES ==========\n")

print("Objetos detectados:", len(objetos))
print("Monedas:", len(monedas))
print("Dados:", len(dados))


# Tabla de monedas

datos_monedas = []

for i, moneda in enumerate(monedas):
    datos_monedas.append({
        "Moneda": i + 1,
        "Grupo": moneda["grupo"],
        "Area": round(moneda["area"], 2),
        "Circularidad": round(moneda["circularidad"], 3)
    })

df_monedas = pd.DataFrame(datos_monedas)

print("\nCLASIFICACIÓN DE MONEDAS")
print(df_monedas)


# Conteo de monedas por grupo

conteo_monedas = df_monedas["Grupo"].value_counts().sort_index()

print("\nCONTEO DE MONEDAS")
print(conteo_monedas)


# Tabla de dados

df_dados = pd.DataFrame({
    "Dado": range(1, len(resultados_dados) + 1),
    "Valor": resultados_dados
})

print("\nVALORES DE LOS DADOS")
print(df_dados)


# Conteo de cada valor de dado

conteo_dados = Counter(resultados_dados)

print("\nCONTEO DE VALORES DE DADOS")

for valor, cantidad in sorted(conteo_dados.items()):
    print(f"Valor {valor}: {cantidad} dados")


# Gráfico de monedas

plt.figure(figsize=(8, 5))
conteo_monedas.plot(kind="bar")
plt.title("Cantidad de monedas por grupo")
plt.xlabel("Grupo")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()


# Gráfico de dados

plt.figure(figsize=(8, 5))
pd.Series(conteo_dados).sort_index().plot(kind="bar")
plt.title("Cantidad de dados por valor")
plt.xlabel("Valor")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()