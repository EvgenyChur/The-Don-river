# -*- coding: utf-8 -*-
"""
@author: Evgeny Churiulin
Скрипт предназначен для создания на основе рядов метеорологических данных макетов для работы архивной версии
модели снежного покрова SnoWE.

Отмечу, что скрипт не является полностью универсальным и в случае длинных рядов данных (например с 1960 года), может работать очень долго, поскольку цикл сначала пробегает по 
всем строкам каждого файла, после чего из-за неправильного зацикливания начинает делать данную процедуру n-раз в зависимости от длины ряда. На основе многочисленных 
экспериментов было установлено, что можно останавливать работу после 1 итерации (результат не измениться). В идеале - надо переделать скрипт
"""

import pandas as pd
import os
import numpy as np

#Функция для считывания данных из файлов
def meteodata(iPath):
    df = pd.read_csv(iPath_csv, sep=';')
    return df

#Для бассейна реки Дон
csv_data = 'D:/Don/data_for_maket/for_makets/'
path_exit = 'D:/Don/data_for_maket/makets/'

dirs_csv = os.listdir(csv_data)

#Очистка результатов предыдущей работы
dirs_result = os.listdir(path_exit)
for file in dirs_result:
    os.remove(path_exit + file)
    
#Объявление и назначение массива
maket_data = ''

for file in dirs_csv:
    fileName_csv = file
    iPath_csv = (csv_data + fileName_csv)
    print (iPath_csv)
    
    df_temp = meteodata(iPath_csv)
    
    if len(maket_data)>0:
        maket_data = pd.concat([maket_data, df_temp])
    else:
        maket_data = df_temp
print ('Columns:',maket_data.columns)

#Создание массива со значениями равными -9999 - константе отсутствия       
data_for_maket = np.full((len(maket_data),2),-9999)
df_zero_values = pd.DataFrame(data=data_for_maket) 

#Получение данные из массивов:
# Первые две строи из массива с -9999 Остальные из метеомассива       
defSwe =  df_zero_values.iloc[:,0]
defRho =  df_zero_values.iloc[:,1]
date_array = maket_data['Date'].values
#meteo_dates = pd.date_range('01-01-2000','31-12-2017', freq = '1d')

# В зависимости от количества доступных рядов метеорологических данных следует выбрать один из нескольких вариантов дальнейшей работы
"""
# Вариант 1 - максимальный набор с исходными данными + корректно задана дата для данных
id_code = maket_data['index       '].values
lat = maket_data['lat         '].values
lon = maket_data['lon         '].values
height = maket_data['height      '].values
snowDepth = maket_data['hSnow       '].values
averageT = maket_data['t2m         '].values
maxT = maket_data['tMax2m      '].values                  
p24Sum = maket_data['R24         '].values  
"""

"""
#Вариант 2 - дата задана не корректно и нужно задать дату в нужно формате
date_array = pd.Series(meteo_dates, name='Date')

id_code = pd.Series(maket_data['index'].values,index=meteo_dates, name='id')
lat = pd.Series(maket_data['lat'].values, index=meteo_dates, name='lat')
lon = pd.Series(maket_data['lon'].values, index=meteo_dates, name='lon')
height = pd.Series(maket_data['height'].values, index=meteo_dates, name='height')
snowDepth = pd.Series(maket_data['hSnow'].values, index=meteo_dates, name='snowDepth')
averageT = pd.Series(maket_data['t2m'].values, index=meteo_dates, name='averageT')
maxT = pd.Series(maket_data['tMax2m'].values, index=meteo_dates, name='maxT')               
p24Sum = pd.Series(maket_data['R24'].values, index=meteo_dates, name='p24Sum')
"""

# Вариант 3 - Корректно задана дата, но исходных данных минимальный набор
# Данный вариант использовался для р. Дон
id_code = maket_data['id_st'].values
lat = maket_data['lat'].values
lon = maket_data['lon'].values
height = maket_data['height'].values
snowDepth = maket_data['snowDepth'].values
averageT = maket_data['averageT'].values
maxT = maket_data['maxT'].values                  
p24Sum = maket_data['p24Sum'].values                  

#Преобразование данных к типу series                    
date_array = pd.Series(date_array, name='Date')
id_code = pd.Series(id_code, name='id')
lat = pd.Series(lat, name='lat')
lon = pd.Series(lon, name='lon')
height = pd.Series(height, name='height')
snowDepth = pd.Series(snowDepth, name='snowDepth')
averageT = pd.Series(averageT, name='averageT')
maxT = pd.Series(maxT, name='maxT') 
p24Sum = pd.Series(p24Sum, name='p24Sum')
defSwe = pd.Series(defSwe, name='defSwe')
defRho = pd.Series(defRho, name='defRho') 

#Слияние данных
date_array.index = id_code.index = lon.index = lat.index = height.index = snowDepth.index = maxT.index = averageT.index = p24Sum.index = defSwe.index = defRho.index 
maket = pd.concat([date_array, id_code, lon, lat, height, snowDepth, maxT, averageT,p24Sum,defSwe,defRho], axis = 1 )

#Сортировка данных
for i in maket['Date']:
    result_data = maket[maket['Date'].isin([i])]
    result_data = result_data.drop(['Date'], axis = 1)
    result_data = result_data.replace(np.nan, 0)
           
    if (i[2] == "." ):
        dstr = i[6:10]+"-"+i[3:5]+"-"+i[0:2]
    else:
        dstr = i
    result_data.to_csv(path_exit + 'smfeIn_'+ dstr+'.txt', sep='\t', encoding='utf-8' ,float_format='%9.3f', index = False)#, mode = 'a') 
    print (result_data.head())
    
 
