# app_mitiempo.py
# Streamlit app for MiTiempo - Prototipo
# Original Colab / script reference: /mnt/data/simulador_completo1.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths to models folder (adjust if needed)
MODELS_DIR = Path("./models")

st.set_page_config(page_title="MiTiempo - Prototipo IA", layout="centered")

st.title("MiTiempo — Prototipo de Autogestión de Permisos")
st.write("Interfaz de demostración que usa pipelines entrenados para predecir resultados de solicitudes de permisos.")

# Helper: load model safely
def load_model(name):
    path = MODELS_DIR / name
    if not path.exists():
        st.error(f"Modelo no encontrado: {path} — ejecuta primero el script de entrenamiento para generar los .pkl en ./models/")
        return None
    return joblib.load(path)

# Load models (lazy load on first use)
nb_pipe = load_model('nb_text2tipo.pkl')
log_pipe = load_model('log_resultado.pkl')
tree_pipe = load_model('tree_resultado.pkl')
reg_pipe = load_model('reg_impacto.pkl')
knn_pipe = load_model('knn_dias.pkl')
ocsvm = load_model('ocsvm_anomalia.pkl')
svm_scaler = load_model('svm_scaler.pkl')
kmeans = load_model('kmeans_segmentacion.pkl')
kmeans_scaler = load_model('kmeans_scaler.pkl')

st.sidebar.header("Nuevo pedido de permiso")
with st.sidebar.form("form_solicitud"):
    motivo = st.text_area("Motivo del permiso", value="Cita médica")
    edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
    antiguedad = st.number_input("Años de antigüedad", min_value=0.0, max_value=60.0, value=1.0)
    dias_ult_ano = st.number_input("Días usados en el último año", min_value=0, max_value=365, value=0)
    dias_solicitados = st.number_input("Días solicitados", min_value=0, max_value=365, value=1)
    area = st.selectbox("Área", ["Administrativa","Operativa","Comercial","Financiera","Tecnologia","Talento Humano"]) 
    submit = st.form_submit_button("Evaluar solicitud")

if submit:
    st.subheader("Resumen de la solicitud")
    st.write(f"**Motivo:** {motivo}")
    st.write(f"**Edad:** {edad} — **Antigüedad:** {antiguedad} años")
    st.write(f"**Días solicitados:** {dias_solicitados} — **Días usados último año:** {dias_ult_ano}")
    st.write(f"**Área:** {area}")

    # 1) Tipo de permiso (Naive Bayes sobre texto)
    if nb_pipe is not None:
        try:
            tipo_pred = nb_pipe.predict([motivo])[0]
            st.success(f"Tipo de permiso detectado por IA: {tipo_pred}")
        except Exception as e:
            st.error(f"Error en predicción tipo de permiso: {e}")
    else:
        st.info("Pipeline NB no cargado.")

    # 2) Detección de anomalías (One-Class SVM)
    if ocsvm is not None and svm_scaler is not None:
        try:
            X_svm = np.array([[dias_solicitados, dias_ult_ano, antiguedad]])
            X_svm_scaled = svm_scaler.transform(X_svm)
            anom = ocsvm.predict(X_svm_scaled)[0]
            if anom == -1:
                st.warning("⚠ Solicitud atípica detectada (anómala)")
            else:
                st.info("✔ Solicitud dentro de los patrones normales")
        except Exception as e:
            st.error(f"Error en detección de anomalías: {e}")
    else:
        st.info("Modelo de anomalías no disponible.")

    # 3) Impacto estimado (Regresión)
    impacto_estimado = None
    if reg_pipe is not None:
        try:
            X_reg = pd.DataFrame([[dias_solicitados, dias_ult_ano, antiguedad]], columns=["dias_solicitados","dias_ult_ano","antiguedad_anios"])
            impacto_estimado = float(reg_pipe.predict(X_reg)[0])
            st.write(f"🔎 Impacto estimado sobre el área: {impacto_estimado:.2f}")
        except Exception as e:
            st.error(f"Error en predicción de impacto: {e}")
    else:
        st.info("Modelo de regresión no disponible.")

    # 4) Probabilidad de aprobación (Logistic Regression)
    if log_pipe is not None:
        try:
            X_log = pd.DataFrame([{"motivo_texto": motivo, "impacto_area": impacto_estimado if impacto_estimado is not None else 0.0, "dias_solicitados": dias_solicitados, "antiguedad_anios": antiguedad, "area": area}])
            probs = log_pipe.predict_proba(X_log)[0]
            classes = log_pipe.classes_
            proba_df = pd.DataFrame({"clase": classes, "prob": probs})
            st.write("**Probabilidades (Regresión Logística):**")
            st.table(proba_df)
        except Exception as e:
            st.error(f"Error en predicción de probabilidades: {e}")
    else:
        st.info("Modelo de probabilidad no disponible.")

    # 5) Decisión del árbol (simulación de decisión final)
    if tree_pipe is not None:
        try:
            X_tree = pd.DataFrame([{"motivo_texto": motivo, "impacto_area": impacto_estimado if impacto_estimado is not None else 0.0, "dias_solicitados": dias_solicitados, "antiguedad_anios": antiguedad, "area": area}])
            pred_tree = tree_pipe.predict(X_tree)[0]
            st.write(f"🧭 Decisión (Árbol): **{pred_tree}**")
        except Exception as e:
            st.error(f"Error en predicción del árbol: {e}")
    else:
        st.info("Árbol de decisión no disponible.")

    # 6) Segmento del empleado (KMeans)
    if kmeans is not None and kmeans_scaler is not None:
        try:
            X_cluster = np.array([[edad, antiguedad, dias_ult_ano]])
            X_cluster_scaled = kmeans_scaler.transform(X_cluster)
            cluster = int(kmeans.predict(X_cluster_scaled)[0])
            st.write(f"👥 Segmento del empleado (KMeans): Grupo {cluster}")
        except Exception as e:
            st.error(f"Error en segmentación KMeans: {e}")
    else:
        st.info("KMeans no disponible.")

    # 7) Sugerencia de días (KNN regressor)
    if knn_pipe is not None:
        try:
            X_knn = pd.DataFrame([[dias_ult_ano, antiguedad, edad]], columns=["dias_ult_ano","antiguedad_anios","edad"])
            sugg = float(knn_pipe.predict(X_knn)[0])
            st.write(f"💡 Sugerencia IA de días: {sugg:.1f} días")
        except Exception as e:
            st.error(f"Error en predicción KNN: {e}")
    else:
        st.info("KNN no disponible.")

    st.success("Evaluación completada. Usa los modelos entrenados localmente para mejorar los resultados.")

st.markdown("---")
st.caption("Si los modelos no están cargando, ejecuta primero el script de entrenamiento que genera ./models/*.pkl y luego recarga esta app.")

# Footer: referencia
st.markdown("**Nota:** Esta app usa pipelines entrenados y guardados en ./models/. El notebook original que generó el dataset/modelos es: `/mnt/data/simulador_completo1.py`.")
