# Regresión lineal del dataset Automobile con MLflow

Proyecto de la cuarta semana de **Operaciones ML**. Se entrena un modelo `LinearRegression` para estimar el precio de automóviles, se comparan cinco configuraciones y se registran parámetros, métricas, predicciones y modelo en **MLflow Tracking**.

## 1. Archivos y estructura

Se recomienda mantener esta estructura dentro del repositorio:

```text
Operaciones-ML/
├── Cuarta Semana/
│   ├── train_mlflow.py
│   ├── requirements.txt
│   └── README.md
├── mlflow_tracking/             # Se crea localmente; no se versiona
│   ├── mlflow.db
│   └── artifacts/
└── start_mlflow.sh             # Script opcional y portable
```

> **Importante:** `train_mlflow.py` espera que la base de datos esté en `../mlflow_tracking/mlflow.db`, respecto a la ubicación del propio archivo. Si se cambia esta estructura, habrá que ajustar la ruta en el código.

No subir `mlflow.db`, los artefactos locales ni `.venv/` a GitHub. Son datos generados durante la ejecución. El archivo `start_mlflow.sh` **sí puede subirse**, siempre que utilice rutas relativas al repositorio y no rutas personales del equipo.

## 2. Requisitos previos

- Python **3.11.9** (versión utilizada en las pruebas originales).
- `pip` y acceso a Internet para instalar dependencias y descargar el dataset de UCI, si no se proporciona localmente.
- Terminal de macOS o Linux. En Windows, cambian los comandos de activación del entorno y del script `.sh`.

Las dependencias directas y sus versiones están en `requirements.txt`.

## 3. Crear un entorno virtual independiente

Desde la raíz del repositorio:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 'Cuarta Semana/requirements.txt'
python -m pip check
```

Si el comando `python3.11` no está disponible, primero debe instalarse Python 3.11.9 o usarse un intérprete Python 3.11 compatible. Para comprobarlo:

```bash
python --version
```

Para salir del entorno virtual, ejecutar `deactivate`.

En Windows (PowerShell), la activación es `.venv\Scripts\Activate.ps1`.

## 4. Inicializar y ejecutar MLflow

**MLflow debe iniciarse antes de ejecutar el entrenamiento por primera vez**, porque `train_mlflow.py` comprueba que ya exista `mlflow_tracking/mlflow.db`.

### Opción recomendada: script portable

Desde la raíz del repositorio:

```bash
chmod +x start_mlflow.sh
./start_mlflow.sh
```

El script crea `mlflow_tracking/` si no existe, inicia el servidor y utiliza rutas calculadas a partir de la ubicación del propio `.sh`. Mantener esta terminal abierta mientras se ejecutan los experimentos.

Abrir **http://127.0.0.1:5000** para consultar MLflow.

### Opción manual (sin `.sh`)

Desde la raíz del repositorio:

```bash
mkdir -p mlflow_tracking/artifacts
mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow_tracking/mlflow.db" \
  --default-artifact-root "file://$PWD/mlflow_tracking/artifacts" \
  --host 127.0.0.1 \
  --port 5000
```

> Las rutas de este ejemplo suponen que se está ejecutando el comando desde la raíz del repositorio. MLflow crea o actualiza su esquema de base de datos al arrancar, según la versión instalada. El directorio `mlflow_tracking/` debe existir previamente.

## 5. Ejecutar los cinco experimentos

En **otra terminal**, activar el mismo entorno virtual y entrar a la carpeta de trabajo:

```bash
source .venv/bin/activate
cd 'Cuarta Semana'
```

Ejecutar las cinco corridas:

| Prueba | Configuración | Comando |
|---|---|---|
| 1 | Baseline | `python train_mlflow.py` |
| 2 | Sin intercepto | `python train_mlflow.py --fit-intercept false --positive false --tol 0.0001 --random-state 42` |
| 3 | Con intercepto explícito | `python train_mlflow.py --fit-intercept true --positive false --tol 0.0001 --random-state 42` |
| 4 | Coeficientes positivos | `python train_mlflow.py --fit-intercept true --positive true --tol 0.0001 --random-state 42` |
| 5 | Semilla diferente | `python train_mlflow.py --fit-intercept true --positive false --tol 0.0001 --random-state 21` |

**Nota metodológica:** las pruebas 1 y 3 tienen exactamente los mismos parámetros, por lo que sirven para comprobar la reproducibilidad de la configuración inicial.

El script utiliza `test_size=0.20` salvo que se indique otro valor. Al modificar `random_state`, cambia la partición de entrenamiento/prueba, por lo que las métricas no se comparan sobre el mismo conjunto de prueba.

## 6. Origen de los datos

El dataset utilizado es **Automobile**, publicado por UCI Machine Learning Repository. Su ficha oficial está disponible en:

**https://archive.ics.uci.edu/dataset/10/automobile**

El programa primero busca el archivo local:

```text
Cuarta Semana/automobile/imports-85.data
```

Si no lo encuentra, utiliza `fetch_ucirepo(id=10)` de la librería `ucimlrepo` para obtener el mismo dataset desde UCI. **Por eso el entrenamiento pudo ejecutarse aunque el archivo de datos estuviera guardado en otro directorio**: al no encontrarlo en la ruta predeterminada, el programa recurrió a la descarga automática.

Por tanto, **no es obligatorio subir el dataset a GitHub**. La ejecución sin archivo local requiere conexión a Internet y disponibilidad del servicio de UCI. La ruta predeterminada se interpreta respecto al directorio de trabajo desde el que se ejecuta Python; por eso las instrucciones indican ejecutar el script desde `Cuarta Semana/`.

También se puede proporcionar una ruta alternativa:

```bash
python train_mlflow.py --ruta '/ruta/al/archivo/imports-85.data'
```

## 7. Revisar y comparar los resultados

En la interfaz de MLflow, localizar el experimento **Automobile - Regresion Lineal - Cuarta Semana**. Cada corrida registra:

- **Parámetros:** `fit_intercept`, `positive`, `tol`, `random_state`, `test_size` y opciones de preparación de datos.
- **Métricas:** MAE, RMSE, R² y MAPE, además de tamaños de entrenamiento y prueba.
- **Artefacto:** `evaluacion/predicciones.csv` con precios reales, predichos y errores.
- **Modelo:** pipeline entrenado con preprocesamiento y regresión lineal.

Comparar MAE y RMSE (**menor es mejor**), R² (**mayor es mejor**) y MAPE (**menor es mejor**). Para comparar configuraciones de forma más justa, mantener la misma partición de prueba.

## 8. Consideraciones para otro equipo

1. Clonar el repositorio y situarse en su raíz.
2. Instalar Python 3.11, crear y activar `.venv`.
3. Instalar `requirements.txt`.
4. Iniciar MLflow con `start_mlflow.sh` o con el comando manual, **antes del entrenamiento**.
5. Ejecutar `train_mlflow.py` desde `Cuarta Semana/`.
6. Consultar los resultados en `http://127.0.0.1:5000`.

La base `mlflow.db` de la computadora original **no se transfiere** mediante este procedimiento: se genera una base nueva y las corridas se ejecutan de nuevo. Si se requiere conservar el historial original, se necesita un procedimiento adicional de respaldo o exportación de MLflow.

**Compatibilidad:** el script fue preparado para las versiones de `requirements.txt`. Su registro del modelo utiliza `skops_trusted_types=["numpy.dtype"]` con MLflow y skops. Esta confianza debe limitarse a modelos generados por código y fuentes confiables.

## 9. Archivos recomendados para GitHub

Versionar `train_mlflow.py`, `requirements.txt`, este `README.md` y, opcionalmente, `start_mlflow.sh`. Añadir un `.gitignore` para excluir entornos virtuales, cachés y resultados locales.
