import mlflow

mlflow.set_experiment("Hello MLflow")

with mlflow.start_run():

    mlflow.log_param("mensaje", "Hello MLflow")
    mlflow.log_metric("valor", 100)

    with open("resultado.txt", "w") as archivo:
        archivo.write("Mi primer experimento con MLflow")

    mlflow.log_artifact("resultado.txt")
