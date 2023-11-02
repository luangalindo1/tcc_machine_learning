# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:52:28 2023

@author: luanfabiomg
"""

#%%
import os
import pandas as pd
import matplotlib.pyplot as plt

#%%
#setando o diretório de trabalho
#os.chdir('//home//luan//Área de Trabalho//data//demanda')
os.chdir('H:\\Meu Drive\\_GÁS INFINITO\\ENGENHARIA ELÉTRICA\\TCC- Estágio\\data\\demanda')
#Backslash \ is an escape character, so you have to use \\

demanda_treino = pd.read_csv("demanda_2020a2022.csv", sep="\t", encoding= 'utf-16', header=None)
demanda_teste = pd.read_csv("demanda_2023.csv", sep="\t", encoding='utf-16', header=None)

#%%
#eliminando a primeira linha
demanda_treino.drop(0, inplace=True) 

#eliminando os valores na das colunas e criando uma nova coluna
demanda_treino["Demanda (MW)"] = demanda_treino.iloc[1:, 5:].apply(lambda x: x.dropna().values, axis=1)

#eliminando as colunas com valores na
demanda_treino.drop(range(5,162), axis=1, inplace=True)

#%%
#eliminando demais colunas que não serão usadas
demanda_treino.drop(range(2,4), axis=1, inplace=True)
demanda_treino.drop(0, axis=1, inplace=True)
demanda_treino.drop(4, axis=1, inplace=True)

#eliminando a primeira linha
demanda_treino.drop(1, inplace=True)

#resetando os índices das linhas
demanda_treino.reset_index(drop=True, inplace=True)

#renomeando as colunas
demanda_treino.columns=["Data", "Demanda (MW)"]

#%%
#convertendo os dados para float
demanda_treino['Demanda (MW)'] = demanda_treino['Demanda (MW)'].astype(float)

#convertendo as datas para um formato reconhecido pelo pandas
demanda_treino['Data'] = pd.to_datetime(demanda_treino['Data'], format='%d/%m/%Y')
demanda_treino['Data'] = demanda_treino['Data'].dt.strftime('%d/%m/%Y')

#%%
#plot exploratório
plt.figure()
plt.title("Demanda entre 2020 e 2022", fontsize=14)
plt.plot(demanda_treino['Data'], demanda_treino['Demanda (MW)'])
plt.xlabel('Semana Operativa', fontsize=14)
plt.xticks(range(0, len(demanda_treino), 10), rotation=45, fontsize=10)
plt.ylabel('Demanda (MW)', fontsize=14)
plt.savefig('exp_demanda.png')

#%%
#realizando os mesmos procedimentos para os dados de teste
demanda_teste.drop(0, inplace=True)
demanda_teste["Demanda (MW)"] = demanda_teste.iloc[1:, 5:].apply(lambda x: x.dropna().values, axis=1)
demanda_teste.drop(range(4,16), axis=1, inplace=True)
demanda_teste.drop(range(2,4), axis=1, inplace=True)
demanda_teste.drop(0, axis=1, inplace=True)
demanda_teste.drop(1, inplace=True)
demanda_teste['Demanda (MW)'] = demanda_treino['Demanda (MW)'].astype(float)
demanda_teste.columns=["Data", "Demanda (MW)"]
demanda_teste['Data'] = pd.to_datetime(demanda_teste['Data'], format='%d/%m/%Y')
demanda_teste['Data'] = demanda_teste['Data'].dt.strftime('%d/%m/%Y')
plt.figure()
plt.title("Demanda em 2023 de janeiro a março", fontsize=14)
plt.plot(demanda_teste['Data'], demanda_teste['Demanda (MW)'])
plt.xlabel('Semana Operativa', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Demanda (MW)', fontsize=14)
plt.savefig('test_demanda.png')

#%%
#salvando os arquivos tratados
demanda_treino.to_csv('demanda_treino.csv', index=False)
demanda_teste.to_csv('demanda_teste.csv', index=False)