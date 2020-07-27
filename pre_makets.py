# -*- coding: utf-8 -*-
"""
@author: Evgenii Churiulin
Скрипт предназначен для подготовки предварительных таблиц с метеоданными (таблица 1) для последующих преобразований и создании на их основе макетов для модели SnoWE.
Отмечу, что исходные данные использованные в данной работе брались с meteo.ru. В результате чего, удалось получить данные только с 13 метеостанций, поэтому было принято 
решение задать их широты и долготы в ручном режиме. В случае, если будет больше станций, то эти данные следует задавать автоматически.

"""
import pandas as pd
import numpy as np

fileName = '34163'
main_path = 'D:/Don/data_for_maket/'
result_exit = main_path + 'for_makets/'

#Создаем датафрейм с исходными данными для станций
id_station = pd.Series([27935, 34056, 34059, 34110, 34123, 34139, 34152,
                        34163, 34240, 34274, 34321, 34356, 34432, 34646])
id_lat = pd.Series([52.88, 52.25, 51.87, 51.17, 51.7, 51.05, 51.55, 
                    51.63, 50.80, 50.42, 50.22, 49.8, 49.38, 47.63])
id_lon = pd.Series([40.48, 43.78, 43.58, 37.35, 39.22, 40.70, 43.15,
                    45.45, 42.00, 41.05, 38.10, 43.67, 40.17, 42.12])
id_height = pd.Series([158, 213, 163, 226, 149, 194, 159,
                       201, 110, 92, 112, 119, 145, 65])

df_initial_date = pd.concat([id_station, id_lat, id_lon, id_height],axis = 1)
df_initial_date.columns = ['id', 'lat', 'lon', 'height']

#Выбор метеоданных для замены значений
for i in range(len(df_initial_date)): 
    if (df_initial_date['id'][i] == int(fileName)):
        id_st = df_initial_date['id'][i]                                       # Выбор станции
        latitude = df_initial_date['lat'][i]                                   # Выбор широты
        longitude = df_initial_date['lon'][i]                                  # Выбор долготы
        st_height = df_initial_date['height'][i]                               # Выбор высоты метеостанции

# Загрузка путей                     
iPath_temp_prec = main_path + '/temp_prec/{}'.format(fileName + '.xlsx')
iPath_temp_max  = main_path + '/max_temp/{}'.format(fileName  + '.xlsx')
iPath_snow      = main_path + '/h_snow/{}'.format(fileName + '.csv')

#Загружаем данные о средней температуре и осадках
df_temp_prec = pd.read_excel(iPath_temp_prec, skiprows = 0, sep=';', skipinitialspace = True, na_values= ['9990','********'])
date_index_1 = df_temp_prec.iloc[:,0]                                            # Индекс по дате
averageT = pd.Series(df_temp_prec['mean_temp'].values, index=date_index_1)       # Средние температуры
p24Sum = pd.Series(df_temp_prec['prec'].values, index=date_index_1)              # Осадки

#Загружаем данные о максимальной температуре
df_temp_max = pd.read_excel(iPath_temp_max, skiprows = 0, sep=';', skipinitialspace = True, na_values= ['9990','********'])
date_index_2 = df_temp_max.iloc[:,0]                                             # Индекс по дате
maxT = pd.Series(df_temp_max['max_temp'].values, index=date_index_2)             # Максимальные температуры
maxT = maxT.replace(np.nan, -9999)

#Загружаем данные о снеге
df_snow = pd.read_csv(iPath_snow, skiprows = 0, sep=';', header=None, skipinitialspace = True, na_values= ['9990','********'])
df_snow.columns = ['Id', 'Year', 'Month', 'Day','SD','6','7','8','9']
year = df_snow.iloc[:,1]
mon = df_snow.iloc[:,2]
day = df_snow.iloc[:,3]
meteo_dates = [pd.to_datetime('{}-{}-{}'.format(i, j, z), format='%Y-%m-%d') for i,j,z in zip(year, mon,day)] 
sd = pd.Series(df_snow['SD'].values, index=meteo_dates)

#Создаем новый пустой массив длины df_temp_prec и шириной равной отсутствующим столбцам требующихся для макета
dummyarray = np.empty((len(df_temp_prec),4))
dummyarray[:] = np.nan
df_zero = pd.DataFrame(dummyarray)


id_code = pd.Series(df_zero[0].values, index=date_index_1) 
id_code = id_code.replace(np.nan, id_st)
lat = pd.Series(df_zero[1].values, index=date_index_1) 
lat = lat.replace(np.nan, latitude)
lon = pd.Series(df_zero[2].values, index=date_index_1) 
lon = lon.replace(np.nan, longitude)
height = pd.Series(df_zero[3].values, index=date_index_1) 
height = height.replace(np.nan, st_height)

#Объединяем пустой дата фрейм с данными за устойчивый морозный период            #Примечания - новые индексы 
df_data_maket = pd.concat([id_code, lat, lon, height, sd, averageT, maxT, p24Sum], axis = 1)      
df_data_maket = df_data_maket.replace(np.nan, -9999)
df_data_maket.to_csv(result_exit + fileName[0:5] +'.csv', sep=';', float_format='%.3f',
                   header = ['id_st','lat','lon','height','snowDepth','averageT','maxT','p24Sum'],
                             index_label = 'Date') 
