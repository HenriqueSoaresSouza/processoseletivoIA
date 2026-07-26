import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

custom_objects = {
    "GlorotUniform": tf.keras.initializers.GlorotUniform,
    'Zeros': tf.keras.initializers.Zeros
}

model = tf.keras.models.load_model("model.h5")

model.summary()

export_path = "./saved_model_temp"
model.export(export_path)

# Usando Dynamic Range Quantization

conversao = tf.lite.TFLiteConverter.from_saved_model(export_path)

conversao.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_dynamic = conversao.convert()
tfl_dyn_path = "./model.tflite"
with open(tfl_dyn_path, "wb") as f:
    f.write(tflite_dynamic)

import shutil
shutil.rmtree(export_path)

print("Modelo otimizado e salvo com sucesso em:", tfl_dyn_path)