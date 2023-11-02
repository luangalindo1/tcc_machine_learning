# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:49:26 2023

@author: luang
"""
#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
#diretório de trabalho
#os.chdir('//home//luan//Área de Trabalho//data//custos')
os.chdir('H:\\Meu Drive\\_GÁS INFINITO\\ENGENHARIA ELÉTRICA\\TCC- Estágio\\data\\custos')

#dados sem tratamento
custo1 = pd.read_csv("1 preco_semanal JAN a JUN 2020.csv", sep=";", encoding='utf-8', decimal=',')
custo2 = pd.read_csv("2 preco_semanal JUN a DEZ 2020.csv", sep=";", encoding='utf-8', decimal=',')
custo3 = pd.read_csv("3 preco_semanal DEZ 2020 a MAI 2021.csv", sep=";", encoding='utf-8', decimal=',')
custo4 = pd.read_csv("4 preco_semanal MAI A OUT 2021.csv", sep=";", encoding='utf-8', decimal=',')
custo5 = pd.read_csv("5 preco_semanal OUT 2021 a ABR 2022.csv", sep=";", encoding='utf-8', decimal=',')
custo6 = pd.read_csv("6 preco_semanal ABR a OUT 2022.csv", sep=";", encoding='utf-8', decimal=',')
custo7 = pd.read_csv("7 preco_semanal OUT a DEZ 2022.csv", sep=";", encoding='utf-8', decimal=',')
custo_teste = pd.read_csv("preco_semanal JAN a MAR 2023.csv", sep=";", encoding='utf-8', decimal=',')

#%%
#concatenando os dados
custo_treino = pd.concat([custo1, custo2, custo3, custo4, custo5, custo6, custo7], 
                         ignore_index=True, levels=None)

#%%
#tomando o custo médio nacional
custo_treino['CMN (R$/MWh)'] = np.mean(custo_treino.loc[:, "SUDESTE":"NORTE"], axis=1)
custo_teste['CMN (R$/MWh)'] = np.mean(custo_teste.loc[:, "SUDESTE":"NORTE"], axis=1)

#%%
#eliminando as colunas desnecessárias
custo_treino.drop(['ANO', 'MES', 'SEMANA', 'DATA_INICIO', 'SUDESTE', 'SUL', 'NORDESTE',
                      'NORTE'], axis=1, inplace=True)
custo_teste.drop(['ANO', 'MES', 'SEMANA', 'DATA_INICIO', 'SUDESTE', 'SUL', 'NORDESTE',
                      'NORTE'], axis=1, inplace=True)
#%%
#renomeando as colunas
custo_treino.columns=["Data", "CMN (R$/MWh)"]
custo_teste.columns=["Data", "CMN (R$/MWh)"]

#%%
#plot exploratório dos custos de treino
plt.figure()
plt.title("Custo Médio Nacional de 2020 a 2022", fontsize=14)
plt.plot(custo_treino['Data'], custo_treino['CMN (R$/MWh)'])
plt.xlabel('Semana', fontsize=14)
plt.xticks(range(0, len(custo_treino), 10), rotation=45, fontsize=10)
plt.ylabel('Custo Médio Nacional (R$/MWh)', fontsize=14)
plt.savefig('cmn_treino.png')

#%%
#plot exploratório dos custos de teste
plt.figure()
plt.title("Custo Médio Nacional em 2023 de janeiro a março", fontsize=14)
plt.plot(custo_teste['Data'], custo_teste['CMN (R$/MWh)'])
plt.xlabel('Semana', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Custo Médio Nacional (R$/MWh)', fontsize=14)
plt.savefig('cmn_teste.png')

#%%
#formatando as datas
custo_treino['Data'] = pd.to_datetime(custo_treino['Data'], format='%d/%m/%Y')
custo_treino['Data'] = custo_treino['Data'].dt.strftime('%d/%m/%Y')
custo_teste['Data'] = pd.to_datetime(custo_teste['Data'], format='%d/%m/%Y')
custo_teste['Data'] = custo_teste['Data'].dt.strftime('%d/%m/%Y')

#%%
#por fim, salvando os dados filtrados
custo_treino.to_csv('custo_treino.csv', index=False)
custo_teste.to_csv('custo_teste.csv', index=False)
