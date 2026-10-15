from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ======================================================
# PASO 1: PREPARAR EL ENTORNO
# ======================================================
# Instalar previamente desde la terminal:
# python -m pip install pandas numpy

carpeta = Path(__file__).resolve().parent

# ======================================================
# PASO 2: CARGAR EL DATASET
# ======================================================

df = pd.read_csv(carpeta / "dataset_con_errores.csv")

print("\n========== PASO 2: CARGA DEL DATASET ==========")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

print("\nPrimeras 5 filas:")
print(df.head())

# Describir las variables.
variables = {
    "id": "Identificador del registro",
    "nombre": "Nombre de la persona",
    "edad": "Edad en años",
    "genero": "Género registrado",
    "pais": "País registrado",
    "ingresos": "Ingreso registrado",
    "fecha_nacimiento": "Fecha de nacimiento",
    "email": "Correo electrónico",
    "telefono": "Número de teléfono",
    "categoria": "Categoría asignada"
}

print("\nDescripción de las variables:")
for columna, descripcion in variables.items():
    print(f"{columna}: {descripcion}")

# ======================================================
# FUNCIONES PARA INTERPRETAR INGRESOS Y FECHAS
# ======================================================

def convertir_ingresos(serie):
    # Según la guía, las comas son separadores de miles.
    texto = serie.astype("string").str.strip()

    for simbolo in ["USD", "$", "€", ","]:
        texto = texto.str.replace(
            simbolo, "", regex=False
        )

    return pd.to_numeric(texto.str.strip(), errors="coerce")


def convertir_fecha(valor):
    if pd.isna(valor):
        return pd.NaT

    texto = str(valor).strip()
    partes = texto.replace("/", "-").split("-")

    if len(partes) != 3:
        return pd.NaT

    if len(partes[0]) == 4:
        formatos = ["%Y-%m-%d", "%Y/%m/%d"]
    else:
        # Para fechas ambiguas se asume día primero.
        formatos = [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y"
        ]

    for formato in formatos:
        try:
            return pd.Timestamp(
                datetime.strptime(texto, formato)
            )
        except ValueError:
            continue

    return pd.NaT


# ======================================================
# PASO 3: EXPLORAR Y DETECTAR PROBLEMAS
# ======================================================

# Reconocer celdas que solamente contienen espacios.
base = df.replace(r"^\s*$", np.nan, regex=True)

print("\n========== PASO 3: DIAGNÓSTICO ==========")

print("\nInformación general:")
df.info()

nulos_antes = base.isna().sum()
duplicados_antes = int(base.duplicated().sum())

print("\nValores faltantes por columna:")
print(nulos_antes)

print("\nTotal de valores faltantes:", int(nulos_antes.sum()))
print("Filas duplicadas:", duplicados_antes)

print("\nEstadísticas de edad:")
edades = pd.to_numeric(base["edad"], errors="coerce")
print(edades.describe())

edades_invalidas = (edades < 18) | (edades > 100)
edades_no_numericas = base["edad"].notna() & edades.isna()

print("\nEdades fuera del rango 18–100:")
print("Cantidad:", int(edades_invalidas.sum()))
print(base.loc[edades_invalidas, ["id", "nombre", "edad"]])
print("Edades no numéricas:", int(edades_no_numericas.sum()))

print("\nFormatos de género:")
print(base["genero"].value_counts(dropna=False))

print("\nFormatos de país:")
print(base["pais"].value_counts(dropna=False))

ingresos = convertir_ingresos(base["ingresos"])
ingresos_invalidos = (ingresos < 0) | (ingresos > 200000)
ingresos_no_numericos = (
    base["ingresos"].notna() & ingresos.isna()
)

print("\nIngresos fuera del rango 0–200000:")
print("Cantidad:", int(ingresos_invalidos.sum()))
print(base.loc[ingresos_invalidos, ["id", "ingresos"]])
print("Ingresos no numéricos:", int(ingresos_no_numericos.sum()))

fechas = base["fecha_nacimiento"].apply(convertir_fecha)
fechas_invalidas = (
    base["fecha_nacimiento"].notna() & fechas.isna()
)

print("\nFechas no interpretables:", int(fechas_invalidas.sum()))

# Diagnóstico de columnas adicionales.
correos_validos = (
    base["email"]
    .astype("string")
    .str.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", na=False)
)

correos_invalidos = base["email"].notna() & ~correos_validos

print("Correos con formato incorrecto:", int(correos_invalidos.sum()))

# Explicar el tratamiento antes de limpiar.
print("\nTratamientos que se aplicarán:")
print("- Duplicados: eliminar filas completamente iguales.")
print("- Nombres: quitar espacios y aplicar formato título.")
print("- Género: unificar formatos y completar con la moda.")
print("- País: unificar su escritura.")
print("- Edad: invalidar valores fuera de 18–100 y usar la mediana.")
print("- Ingresos: limpiar formatos, aplicar rango y usar la mediana.")
print("- Fechas: convertir formatos y dejar vacías las inválidas.")
print("- Correos y teléfonos: conservar para revisión, sin inventar datos.")

# ======================================================
# PASO 4: LIMPIAR LOS DATOS
# ======================================================

print("\n========== PASO 4: LIMPIEZA ==========")

# 4.1. Crear una copia.
df_limpio = base.copy()

# 4.2. Eliminar filas duplicadas.
filas_antes = len(df_limpio)
df_limpio = df_limpio.drop_duplicates().copy()

print("\n4.2. Duplicados eliminados:",
      filas_antes - len(df_limpio))

# 4.3. Limpiar nombres.
df_limpio["nombre"] = (
    df_limpio["nombre"]
    .astype("string")
    .str.strip()
    .str.title()
)

print("\n4.3. Nombres limpios:")
print(df_limpio["nombre"].head(10))

# 4.4. Estandarizar género.
genero = (
    df_limpio["genero"]
    .astype("string")
    .str.strip()
    .str.lower()
)

mapa_genero = {
    "m": "Masculino",
    "masculino": "Masculino",
    "f": "Femenino",
    "femenino": "Femenino"
}

df_limpio["genero"] = genero.map(mapa_genero)

modas = df_limpio["genero"].mode()

if not modas.empty:
    moda_genero = modas.iloc[0]
    cantidad = int(df_limpio["genero"].isna().sum())

    df_limpio["genero"] = df_limpio["genero"].fillna(
        moda_genero
    )

    print("\n4.4. Moda de género:", moda_genero)
    print("Géneros completados:", cantidad)
else:
    print("\n4.4. No hay géneros válidos para calcular la moda.")

# 4.5. Estandarizar países.
pais = (
    df_limpio["pais"]
    .astype("string")
    .str.strip()
    .str.lower()
)

mapa_paises = {
    "mexico": "México",
    "méxico": "México",
    "colombia": "Colombia",
    "argentina": "Argentina",
    "españa": "España"
}

df_limpio["pais"] = pais.map(mapa_paises)

# Conservar otros países si existieran.
df_limpio["pais"] = df_limpio["pais"].fillna(
    pais.str.title()
)

print("\n4.5. Países estandarizados:")
print(df_limpio["pais"].value_counts(dropna=False))

# 4.6. Corregir edades.
df_limpio["edad"] = pd.to_numeric(
    df_limpio["edad"], errors="coerce"
)

df_limpio.loc[
    (df_limpio["edad"] < 18) | (df_limpio["edad"] > 100),
    "edad"
] = np.nan

mediana_edad = df_limpio["edad"].median()

if pd.notna(mediana_edad):
    cantidad = int(df_limpio["edad"].isna().sum())

    df_limpio["edad"] = df_limpio["edad"].fillna(
        mediana_edad
    )

    print("\n4.6. Mediana de edad:", mediana_edad)
    print("Edades completadas:", cantidad)
else:
    print("\n4.6. No hay edades válidas para calcular la mediana.")

df_limpio["edad"] = df_limpio["edad"].round().astype("Int64")

# 4.7. Limpiar ingresos.
df_limpio["ingresos"] = convertir_ingresos(
    df_limpio["ingresos"]
)

df_limpio.loc[
    (df_limpio["ingresos"] < 0)
    | (df_limpio["ingresos"] > 200000),
    "ingresos"
] = np.nan

mediana_ingresos = df_limpio["ingresos"].median()

if pd.notna(mediana_ingresos):
    cantidad = int(df_limpio["ingresos"].isna().sum())

    df_limpio["ingresos"] = df_limpio["ingresos"].fillna(
        mediana_ingresos
    )

    print("\n4.7. Mediana de ingresos:", mediana_ingresos)
    print("Ingresos completados:", cantidad)
else:
    print("\n4.7. No hay ingresos válidos para calcular la mediana.")

# Nota: quitar símbolos monetarios no convierte monedas.
# Se sigue el tratamiento de formatos indicado por la guía.

# 4.8. Convertir fechas.
df_limpio["fecha_nacimiento"] = (
    df_limpio["fecha_nacimiento"].apply(convertir_fecha)
)

print("\n4.8. Fechas convertidas:")
print(df_limpio["fecha_nacimiento"].head(10))

# ======================================================
# PASO 5: VALIDAR Y COMPARAR RESULTADOS
# ======================================================

print("\n========== PASO 5: VALIDACIÓN FINAL ==========")

print("\nDimensiones:")
print("Filas antes:", len(df))
print("Filas después:", len(df_limpio))
print("Columnas:", df_limpio.shape[1])

nulos_despues = df_limpio.isna().sum()

comparacion = pd.DataFrame({
    "Nulos antes": nulos_antes,
    "Nulos después": nulos_despues
})

print("\nComparación de valores faltantes:")
print(comparacion)

print("\nTotal de nulos antes:", int(nulos_antes.sum()))
print("Total de nulos después:", int(nulos_despues.sum()))

print("\nDuplicados después de limpiar:")
print(int(df_limpio.duplicated().sum()))

print("\nGéneros finales:")
print(df_limpio["genero"].value_counts(dropna=False))

print("\nPaíses finales:")
print(df_limpio["pais"].value_counts(dropna=False))

print("\nEstadísticas finales de edad:")
print(df_limpio["edad"].describe())

print("\nEstadísticas finales de ingresos:")
print(df_limpio["ingresos"].describe())

print("\nEdades fuera del rango después de limpiar:")
print(int((
    (df_limpio["edad"] < 18) | (df_limpio["edad"] > 100)
).sum()))

print("\nIngresos fuera del rango después de limpiar:")
print(int((
    (df_limpio["ingresos"] < 0)
    | (df_limpio["ingresos"] > 200000)
).sum()))

print("\nObservaciones:")
print("- Las fechas inválidas o desconocidas permanecen vacías.")
print("- Las fechas ambiguas se interpretaron con el día primero.")
print("- Los correos incorrectos y teléfonos faltantes requieren revisión.")
print("- No se verificó la correspondencia entre edad y fecha de nacimiento.")

# ======================================================
# PASO 6: GUARDAR EL DATASET LIMPIO
# ======================================================

ruta_salida = carpeta / "dataset_limpio.csv"

df_limpio.to_csv(
    ruta_salida,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)

print("\n========== PASO 6: ARCHIVO GUARDADO ==========")
print("Archivo: dataset_limpio.csv")
print("Ubicación:", ruta_salida)
print("\n¡Práctica terminada!")