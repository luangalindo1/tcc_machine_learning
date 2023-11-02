# -*- coding: utf-8 -*
#Created on Wed May 3 10:15:15 2023

#@author: luang

#%% bibliotecas
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#%% datasets e diretório de trabalho 
os.chdir('/...PATH...')

custo_treino = pd.read_csv("custo_treino.csv", sep=",", encoding='utf-8') 
custo_teste = pd.read_csv("custo_teste.csv", sep=",", encoding='utf-8') 
ibov_treino = pd.read_csv("ibov_treino.csv", sep=",", encoding='utf-8') 
ibov_teste = pd.read_csv("ibov_teste.csv", sep=",", encoding='utf-8') 
demanda_treino = pd.read_csv("demanda_treino.csv", sep=",", encoding='utf-8') 
demanda_teste = pd.read_csv("demanda_teste.csv", sep=",", encoding='utf-8')

#%% arranjo dos conjuntos de treinamento e teste
train_dataset = pd.concat([custo_treino, ibov_treino, demanda_treino], axis=1, ignore_index=True) 
train_dataset.dropna(axis=0, how='any', inplace=True)
test_dataset = pd.concat([custo_teste, ibov_teste, demanda_teste], axis=1, ignore_index=True) 
test_dataset.dropna(axis=0, how='any', inplace=True)

#%% removendo as datas e dividindo o dataset 
train_dataset.drop(columns=[0, 2, 4], inplace=True) 
train_dataset.columns = ["Custo (R$/MWh)", "Ibov", "Demanda (MW)"]

test_dataset.drop(columns=[0, 2, 4], inplace=True) 
test_dataset.columns = ["Custo (R$/MWh)", "Ibov", "Demanda (MW)"] 

#%% inspeção dos dados
sns.pairplot(train_dataset, diag_kind="kde")

#%% análise estatística
train_stats = train_dataset.describe() # análise estatística dos dados train_stats.pop("Demanda (MW)") #remoção dos alvos
train_stats = train_stats.transpose()

#%% separação das etiquetas (labels)
#criando novos datasets para não modificar os originais
 
train_features = train_dataset.copy() 
test_features = test_dataset.copy()

train_labels = train_features.pop('Demanda (MW)') 
test_labels = test_features.pop('Demanda (MW)')

# os dados serão normalizados por questão de boas práticas, # e para aumentar a eficiência do algoritmo
norm = tf.keras.layers.Normalization(axis=-1) norm.adapt(np.array(train_features))
#%% construção do modelo de aprendizagem profunda (DNN)
def build_and_compile_model(normalizer): 
    model = keras.Sequential([normalizer, 
                              layers.Dense(64, activation='relu'), 
                              layers.Dense(64, activation='relu'), 
                              layers.Dense(1)])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#%% treinamento do modelo
dnn_model = build_and_compile_model(norm) 
dnn_model.summary()

history = dnn_model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=500)

#%% gráfico temporal
def plot_loss(history): plt.figure()
plt.plot(history.history['loss'], label='Erro') 
plt.plot(history.history['val_loss'], label='Erro de validação') 
plt.xlabel('Época')
plt.ylabel('Erro [MW]') 
plt.legend() 
plt.grid(True)

plot_loss(history) 
#%% teste do modelo 
test_results = {}
test_results['Modelo_DNN'] = dnn_model.evaluate(test_features,
test_labels, verbose=0)

#%% predição da demanda (MW) usando o conjunto de teste 
test_predictions = dnn_model.predict(test_features).flatten()
plt.figure()
a = plt.axes(aspect='equal') 
plt.scatter(test_labels, test_predictions) 
plt.xlabel('Valores Verdadeiros [MW]') 
plt.ylabel('Previsões [MW]')
lims = [70, 90] 
plt.xlim(lims) 
plt.ylim(lims)
_ = plt.plot(lims, lims)

#%% distribuição de erros
error = test_predictions - test_labels

plt.figure() 
plt.hist(error, bins = 25)
plt.xlabel("Erro de Predição [MW]")
_ = plt.ylabel("Quantidade") 
plt.xticks(np.arange(-7, 8, 1))

#%% erros percentuais
error_p = (error/test_labels)*100

plt.figure() 
plt.hist(error_p, bins=10) 
plt.title('Erro de Previsão')
plt.ylabel('Erro percentual (%)') 
plt.xticks(labels=[])

error_p_stats = error_p.describe() 
error_stats = error.describe()

#%% salvando o modelo 
dnn_model.save('Modelo_DNN')

#%% carregar o modelo
reloaded = tf.keras.models.load_model('Modelo_DNN')

test_results['reloaded'] = reloaded.evaluate( test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Erro Médio Absoluto [MW]']).T
