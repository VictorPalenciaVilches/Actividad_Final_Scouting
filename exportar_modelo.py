"""
exportar_modelo.py
Fase 1 y 2 del ecosistema analítico: ingesta, preprocesamiento, diagnóstico de
overfitting, optimización de Random Forest con GridSearchCV, evaluación (ROC-AUC,
importancia de variables) y serialización del paquete de despliegue con joblib.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # solo guardar imágenes, sin abrir ventanas
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

from src.database.connection2 import MySQLConnector

CARPETA_MODELOS = "models"
CARPETA_RESULTADOS = "resultados"


def evaluar(modelo, X_te, y_te):
    """Devuelve las métricas de un modelo sobre el conjunto de Test."""
    y_pred = modelo.predict(X_te)
    y_proba = modelo.predict_proba(X_te)[:, 1]
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred, pos_label=1),
        "auc": roc_auc_score(y_te, y_proba),
    }


def main():
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    print("=================================================================")
    print("  FASE 1: ENTRENAMIENTO Y PERSISTENCIA DEL MODELO (DEPLOYMENT)")
    print("=================================================================\n")

    # ---------------------------------------------------------------
    # 1. INGESTA DESDE MYSQL
    # ---------------------------------------------------------------
    db = MySQLConnector()
    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            p.position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """
    print("[1/6] Extrayendo datos desde MySQL...")
    df = db.extract_data(query)
    if df.empty:
        print("[ERROR] No se obtuvieron datos de la base de datos.")
        return
    print(f"      Registros obtenidos: {len(df)}\n")

    # ---------------------------------------------------------------
    # 2. PREPROCESAMIENTO (se guardan las medianas para producción)
    # ---------------------------------------------------------------
    print("[2/6] Preprocesamiento e imputación por mediana agrupada...")
    df["estatura_cm"] = pd.to_numeric(df["estatura_cm"], errors="coerce")
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["valor_mercado"] = pd.to_numeric(df["valor_mercado"], errors="coerce")
    df["posicion"] = df["posicion"].fillna("Sin Posicion")

    medianas_estatura = df.groupby("posicion")["estatura_cm"].median().to_dict()
    mediana_global_estatura = df["estatura_cm"].median()
    mediana_global_edad = df["edad"].median()

    df["estatura_cm"] = df.groupby("posicion")["estatura_cm"].transform(lambda x: x.fillna(x.median()))
    df["estatura_cm"] = df["estatura_cm"].fillna(mediana_global_estatura)
    df["edad"] = df["edad"].fillna(mediana_global_edad)

    umbral_elite = df["valor_mercado"].quantile(0.75)
    df["es_elite"] = (df["valor_mercado"] >= umbral_elite).astype(int)

    df_dummies = pd.get_dummies(df[["posicion"]], drop_first=True, dtype=int)
    X = pd.concat([df[["edad", "estatura_cm"]], df_dummies], axis=1)
    y = df["es_elite"]

    filas_nan = X.isna().any(axis=1)
    if filas_nan.sum() > 0:
        X, y = X[~filas_nan], y[~filas_nan]

    print(f"      Umbral Élite (percentil 75): {umbral_elite:,.0f} EUR")
    print(f"      Clase Estándar: {(y == 0).sum()} | Clase Élite: {(y == 1).sum()}\n")

    # ---------------------------------------------------------------
    # 3. DIVISIÓN ESTRATIFICADA 80/20
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ---------------------------------------------------------------
    # 4. DIAGNÓSTICO DE OVERFITTING: ÁRBOL LIBRE VS ÁRBOL PODADO
    # ---------------------------------------------------------------
    print("[3/6] Diagnóstico de overfitting (árbol libre vs podado)...")
    tree_libre = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X_train, y_train)
    tree_podado = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42).fit(X_train, y_train)

    acc = {}
    for nombre, m in [("Árbol Sin Límite", tree_libre), ("Árbol Podado", tree_podado)]:
        acc[nombre] = (accuracy_score(y_train, m.predict(X_train)),
                       accuracy_score(y_test, m.predict(X_test)))
        print(f"      {nombre:<18} Train: {acc[nombre][0]:.4f} | Test: {acc[nombre][1]:.4f} "
              f"| Brecha: {acc[nombre][0] - acc[nombre][1]:.4f}")
    print()

    # Gráfico comparativo de overfitting
    fig, ax = plt.subplots(figsize=(7, 5))
    etapas = list(acc.keys())
    x = np.arange(len(etapas))
    w = 0.35
    b1 = ax.bar(x - w / 2, [acc[e][0] for e in etapas], w, label="Train Accuracy", color="#e74c3c", edgecolor="black")
    b2 = ax.bar(x + w / 2, [acc[e][1] for e in etapas], w, label="Test Accuracy", color="#3498db", edgecolor="black")
    ax.bar_label(b1, fmt="%.4f", padding=2)
    ax.bar_label(b2, fmt="%.4f", padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(etapas)
    ax.set_ylim(0, 1.1)
    ax.set_title("Diagnóstico de Overfitting: Train vs Test", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/fig_overfitting.png", dpi=200)
    plt.close()

    # ---------------------------------------------------------------
    # 5. ÁRBOL ÚNICO OPTIMIZADO (referencia para la curva ROC)
    # ---------------------------------------------------------------
    print("[4/6] GridSearchCV sobre el Árbol de Decisión (modelo de referencia)...")
    grid_tree = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={
            "max_depth": [2, 3, 4, 5, 6],
            "min_samples_leaf": [3, 5, 10, 15],
            "criterion": ["gini", "entropy"],
            "class_weight": ["balanced", None],
        },
        cv=5, scoring="f1", n_jobs=-1,
    )
    grid_tree.fit(X_train, y_train)
    tree_opt = grid_tree.best_estimator_
    m_tree = evaluar(tree_opt, X_test, y_test)
    print(f"      Mejores hiperparámetros: {grid_tree.best_params_}")
    print(f"      Test -> F1: {m_tree['f1']:.4f} | AUC: {m_tree['auc']:.4f}\n")

    # ---------------------------------------------------------------
    # 6. RANDOM FOREST + GRIDSEARCHCV
    # ---------------------------------------------------------------
    print("[5/6] GridSearchCV sobre Random Forest (puede tardar unos minutos)...")
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [5, 10],
            "class_weight": ["balanced", None],
        },
        cv=5, scoring="f1", n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)
    rf_opt = grid_rf.best_estimator_
    m_rf = evaluar(rf_opt, X_test, y_test)
    print(f"      Mejores hiperparámetros: {grid_rf.best_params_}")
    print(f"      F1 medio en validación cruzada: {grid_rf.best_score_:.4f}")
    print("\n      --- Comparación sobre el conjunto de Test ---")
    print(f"      {'Modelo':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'AUC':>9}")
    for nombre, m in [("Árbol optimizado", m_tree), ("Random Forest", m_rf)]:
        print(f"      {nombre:<22}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['auc']:>9.4f}")

    cm = confusion_matrix(y_test, rf_opt.predict(X_test))
    print(f"\n      Matriz de confusión Random Forest [[VN, FP], [FN, VP]]: {cm.tolist()}\n")

    # Curvas ROC comparativas
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, modelo, color in [("Árbol Único (optimizado)", tree_opt, "#e67e22"),
                                  ("Random Forest", rf_opt, "#27ae60")]:
        fpr, tpr, _ = roc_curve(y_test, modelo.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{nombre} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.5)")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    ax.set_title("Curvas ROC: Árbol Único vs Random Forest", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/fig_roc.png", dpi=200)
    plt.close()

    # Importancia de variables (Gini)
    imp = pd.Series(rf_opt.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("      --- Importancia de variables (Gini) ---")
    for nombre, valor in imp.items():
        print(f"      {nombre:<28}{valor:.4f}")
    print()
    top = imp.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.barh(top.index, top.values, color="#8e44ad", edgecolor="black")
    ax.bar_label(barras, fmt="%.3f", padding=3)
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Importancia de Variables: Random Forest", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/fig_importancias.png", dpi=200)
    plt.close()

    # Matriz de confusión del Random Forest
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Estándar", "Élite"]).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Matriz de Confusión: Random Forest", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/fig_matriz_rf.png", dpi=200)
    plt.close()

    # ---------------------------------------------------------------
    # 7. MODELO FINAL (reentrenado con todos los datos) Y SERIALIZACIÓN
    # ---------------------------------------------------------------
    print("[6/6] Entrenando modelo final con todos los datos y exportando...")
    rf_final = RandomForestClassifier(**grid_rf.best_params_, random_state=42, n_jobs=-1)
    rf_final.fit(X, y)

    paquete_despliegue = {
        "modelo": rf_final,
        "columnas_X": X.columns.tolist(),
        "medianas_estatura": medianas_estatura,
        "mediana_global_estatura": mediana_global_estatura,
        "mediana_global_edad": mediana_global_edad,
        "umbral_elite_eur": umbral_elite,
        "posiciones_validas": sorted(df["posicion"].unique().tolist()),
    }
    ruta_modelo = f"{CARPETA_MODELOS}/modelo_scouting_rf.joblib"
    joblib.dump(paquete_despliegue, ruta_modelo)
    print(f"      Paquete exportado en: '{ruta_modelo}'")

    # Resumen en JSON (para el informe)
    resumen = {
        "overfitting": {k: {"train": v[0], "test": v[1], "brecha": v[0] - v[1]} for k, v in acc.items()},
        "arbol_optimizado": {"params": grid_tree.best_params_, **m_tree},
        "random_forest": {"params": grid_rf.best_params_, "f1_cv": grid_rf.best_score_, **m_rf},
        "matriz_confusion_rf": cm.tolist(),
        "importancias": imp.to_dict(),
        "umbral_elite_eur": float(umbral_elite),
    }
    with open(f"{CARPETA_RESULTADOS}/resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=float)
    print(f"      Resumen guardado en '{CARPETA_RESULTADOS}/resumen.json'")
    print(f"      Gráficas guardadas en '{CARPETA_RESULTADOS}/'\n")
    print("[OK] Proceso completado.")


if __name__ == "__main__":
    main()