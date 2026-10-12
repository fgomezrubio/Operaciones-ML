"""Automobile LinearRegression: argparse + MLflow Tracking."""
import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

COLUMNAS = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
MAPAS = {'num-of-cylinders': {'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8,'twelve':12}, 'num-of-doors': {'two':2,'four':4}}

def booleano(v):
    if v.lower() in ('true','1','yes','si'): return True
    if v.lower() in ('false','0','no'): return False
    raise argparse.ArgumentTypeError('Usa true o false')

def proporcion(v):
    try: n = float(v)
    except ValueError as e: raise argparse.ArgumentTypeError('Debe ser un número') from e
    if not 0 < n < 1: raise argparse.ArgumentTypeError('Debe estar entre 0 y 1')
    return n

def tolerancia(v):
    try: n = float(v)
    except ValueError as e: raise argparse.ArgumentTypeError('Debe ser un número') from e
    if not np.isfinite(n) or n <= 0: raise argparse.ArgumentTypeError('Debe ser positiva y finita')
    return n

def cargar_datos(ruta):
    if ruta.exists():
        return pd.read_csv(ruta, header=None, names=COLUMNAS, na_values=['?'])
    from ucimlrepo import fetch_ucirepo
    uci = fetch_ucirepo(id=10)
    df = pd.concat([uci.data.features, uci.data.targets], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df[COLUMNAS]

def main():
    parser = argparse.ArgumentParser(description='Regresión Lineal Automobile + MLflow')
    parser.add_argument('--fit-intercept', type=booleano, default=True)
    parser.add_argument('--positive', type=booleano, default=False)
    parser.add_argument('--tol', type=tolerancia, default=0.0001)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--test-size', type=proporcion, default=0.20)
    parser.add_argument('--ruta', default='automobile/imports-85.data')
    parser.add_argument('--experiment', default='Automobile - Regresion Lineal - Cuarta Semana')
    args = parser.parse_args()

    df = cargar_datos(Path(args.ruta))
    for col, mapa in MAPAS.items():
        texto = df[col].astype('string').str.strip().str.lower()
        df[col] = texto.map(mapa).fillna(pd.to_numeric(texto, errors='coerce')).astype('Int64')
    df = df.dropna(subset=['price']).copy()
    X, y = df.drop(columns=['price']), df['price']
    numericas = X.select_dtypes(include='number').columns.tolist()
    categoricas = X.select_dtypes(include=['object','string','category']).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    prep = ColumnTransformer([
        ('numericas', Pipeline([('imputacion', SimpleImputer(strategy='median'))]), numericas),
        ('categoricas', Pipeline([('imputacion', SimpleImputer(strategy='most_frequent')),
                                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categoricas),
    ])
    modelo = Pipeline([('preprocesamiento', prep), ('modelo', LinearRegression(
        fit_intercept=args.fit_intercept, positive=args.positive, tol=args.tol))])

    # Desde Cuarta Semana: usar la base de datos ya configurada en el directorio padre.
    tracking_db = (Path(__file__).resolve().parent.parent / 'mlflow_tracking' / 'mlflow.db')
    if not tracking_db.is_file():
        raise FileNotFoundError(f'No existe la base MLflow esperada: {tracking_db}')
    mlflow.set_tracking_uri(f'sqlite:///{tracking_db.as_posix()}')
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f'LinearRegression_intercept_{args.fit_intercept}_positive_{args.positive}_seed_{args.random_state}') as run:
        for clave, valor in {
            'modelo': 'LinearRegression', 'fit_intercept': args.fit_intercept,
            'positive': args.positive, 'tol': args.tol,
            'random_state': args.random_state, 'test_size': args.test_size,
            'imputacion_numerica': 'median', 'imputacion_categorica': 'most_frequent',
            'encoding': 'OneHotEncoder',
        }.items():
            mlflow.log_param(clave, valor)

        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        metricas = {
            'MAE': mean_absolute_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'R2': r2_score(y_test, pred),
            'MAPE': mean_absolute_percentage_error(y_test, pred) * 100,
        }
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, float(valor))
        mlflow.log_metric('n_train', len(X_train))
        mlflow.log_metric('n_test', len(X_test))

        with tempfile.TemporaryDirectory() as carpeta:
            csv = Path(carpeta) / 'predicciones.csv'
            pd.DataFrame({'precio_real': y_test.to_numpy(), 'precio_predicho': pred,
                          'error': y_test.to_numpy() - pred}).to_csv(csv, index=False)
            mlflow.log_artifact(str(csv), artifact_path='evaluacion')
        mlflow.sklearn.log_model(sk_model=modelo, name='modelo', skops_trusted_types=["numpy.dtype"])

        print('\n=== Parámetros ===')
        print(f'fit_intercept={args.fit_intercept}, positive={args.positive}, tol={args.tol}, random_state={args.random_state}, test_size={args.test_size}')
        print(f'Registros: {len(df)} | Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}')
        print('\n=== Métricas ===')
        for nombre, valor in metricas.items():
            print(f'{nombre}: {valor:.4f}' + ('%' if nombre == 'MAPE' else ''))
        print(f'\nRun ID: {run.info.run_id}')
        print(f'Tracking URI: {mlflow.get_tracking_uri()}')

if __name__ == '__main__':
    main()
