# -*- coding: utf-8 -*-
"""
@author: Evgenii Churiulin
Данный скрипт предназначен для выполнениния нескольких задач:
1) Коррекция метеорологических данных, полученных в ходе работы с базой данных ФГБУ "Гидрометцентр России". Данная задача возникла из-за наличия грубых ошибок в исходных данных.
Данная задача относится к фазе подготовительной работы и включена в раздел 1 - Подготовительная стадия работы.
2) Расчет и статистическая оценка периодов с оттепелями и устойчивыми морозными периодами.
Следует отметить, что данный скрипт может быть запущен как для целого ряда станций (70 в моем примере), так и для одной уникальной станции (см. код)
"""

import pandas as pd
import os
import datetime
import numpy as np
from pandas.tseries.offsets import DateOffset
import logging



#Версия для работы по одной станции
"""
fileName = '26825.csv'
iPath = 'D:/Churyulin/msu_cosmo/Data_Natalia_Leonidovna/Data/{}'.format(fileName)
"""

#Версия для работы по станциям
path_main = 'D:/Don/' #Путь к папке, где хранится проект с рекой Дон


# Фаза работы 1: Подготовительная. 
"""
#Входные данные
data_1 = path_main + 'meteo_2000_2011/'
data_2 = path_main + 'meteo_2011_2020/'
data_snowe = path_main + 'snowe_data/'

dirs_csv_1 = sorted(os.listdir(data_1))
dirs_csv_2 = sorted(os.listdir(data_2))
dirs_csv_3 = sorted(os.listdir(data_snowe))

#Выходные данные
result_exit = path_main + 'meteo_2000_2020/'

#Очистка предыдуших результатов работы
dirs_exit = os.listdir(result_exit)
for file in dirs_exit:
    os.remove(result_exit + file)


#Работа с временным рядом по метеостанции
for data_file in dirs_csv_1:
    fileName_csv = data_file
    fileName_snowe = '000' + str(data_file[0:5]) + '.txt'
    
    iPath_1 = (data_1 + fileName_csv)
    iPath_2 = (data_2 + fileName_csv)
    iPath_3 = (data_snowe + fileName_snowe)
    
    # Считываем данные из модели SnoWE
    #df_snowe = pd.read_csv(iPath_3, skiprows = 0, sep=' ', dayfirst = True, parse_dates = True, index_col = [0],
                           #skipinitialspace = True, na_values= ['9990','********'])   
    widths = [4,2,2,8,10, 10, 8, 12, 10, 10, 12] 
    df_snowe = pd.read_fwf(iPath_3, widths=widths, skiprows=0, skip_footer=0, header = 0) 
    df_snowe.columns = ['YYYY', 'MM', 'DD', 'route','lon','lat','maxT','aveT','depth','swe','rho']
    
    # На основе данных создаем индекс по дате, и приводим его к удобоваримому варианту для дальнейшей работы
    year = df_snowe.iloc[:,0]
    month = df_snowe.iloc[:,1]
    day = df_snowe.iloc[:,2]
    meteo_dates = [pd.to_datetime('{}-{}-{}'.format(i, j, z), format='%Y-%m-%d') for i,j,z in zip(year, month,day)] 
    dtime = pd.Series(meteo_dates)
    dtime = dtime - DateOffset(days=1)
        
    # Считываем данные по метеостанциям за период с 2000 по 2011 год
    df_2000_2011 = pd.read_csv(iPath_1, skiprows = 0, sep=';', dayfirst = True,
                               parse_dates = True, header = None, skipinitialspace = True,
                               na_values= ['******','********'])
    df_2000_2011 = df_2000_2011.drop_duplicates(keep = False)
    df_2000_2011 = df_2000_2011.drop([5,6,13,14,15,16,17,18,19,20,21,22,23,24,28,33,34,35,
                                      36,37,38,39,40,41,42,43,44,45,46], axis=1)
    
    # Считываем данные по метеостанциям за период с 2011 по 2020 год 
    df_2011_2020 = pd.read_csv(iPath_2, skiprows = 0, sep=';', dayfirst = True,
                               parse_dates = True, header = None, skipinitialspace = True,
                               na_values= ['***','******','********'])
    df_2011_2020 = df_2011_2020.drop_duplicates(keep = False)
    df_2011_2020 = df_2011_2020.drop([5,6,13,14,15,16,17,18,19,20,21,22,23,24,28,33,34,35,
                                      36,37,38,39,40,41,42,43,44,45,46], axis=1)
         
    #Формируем общий массив с данными 
    df_2000_2020 = pd.concat([df_2000_2011, df_2011_2020])
    df_2000_2020 = df_2000_2020.drop_duplicates()
    df_2000_2020.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    
    
    #Считываем данные для работы с ними
    index_date = pd.to_datetime(df_2000_2020[0])    # time step
    index_meteo = pd.Series(df_2000_2020[1].values, index = index_date, dtype = 'float')  # index meteostation
    lat = pd.Series(df_2000_2020[2].values, index = index_date, dtype = 'float')          # latitude of meteostation
    lon = pd.Series(df_2000_2020[3].values, index = index_date, dtype = 'float')          # longitude of meteostation    
    h_station = pd.Series(df_2000_2020[4].values, index = index_date, dtype = 'float')    # higth of meteostation    
    ps = pd.Series(df_2000_2020[5].values, index = index_date, dtype = 'float')           # air pressure at meteostation
    pmsl = pd.Series(df_2000_2020[6].values, index = index_date, dtype = 'float')         # air pressure at meteostation in according to see level    
    t2m = pd.Series(df_2000_2020[7].values, index = index_date, dtype = 'float')          # 2m air temperature
    t2m_negative = pd.Series(df_2000_2020[7].values, index = index_date, dtype = 'float') # 2m air temperature special series for negative temp
    td2m = pd.Series(df_2000_2020[8].values, index = index_date, dtype = 'float')         # 2m dew point
    dd10m = pd.Series(df_2000_2020[9].values, index = index_date, dtype = 'float')        # direction of wind
    ff10m = pd.Series(df_2000_2020[10].values, index = index_date, dtype = 'float')       # speed of wind 
    #ff10max = pd.Series(df_2000_2020[10].values, index = index_date, dtype = 'float')    # speed of wind (max for day)   
    tmin2m = pd.Series(df_2000_2020[11].values, index = index_date, dtype = 'float')      # 2m min air temperature
    tmax2m = pd.Series(df_2000_2020[12].values, index = index_date, dtype = 'float')      # 2m max air temperature   
    tming = pd.Series(df_2000_2020[13].values, index = index_date, dtype = 'float')       # min soil temperature for night 
    R12 = pd.Series(df_2000_2020[14].values, index = index_date, dtype = 'float')         # Precipitation for 12 hours
    R12_liquid = pd.Series(df_2000_2020[14].values, index = index_date, dtype = 'float')  # Precipitation for 12 hours special liquid
    R12_solid = pd.Series(df_2000_2020[14].values, index = index_date, dtype = 'float')   # Precipitation for 12 hours special soil
    R24 = pd.Series(df_2000_2020[15].values, index = index_date, dtype = 'float')         # Precipitation for 24 hours
    t_g = pd.Series(df_2000_2020[16].values, index = index_date, dtype = 'float')         # temperatura of soil
    hsnow = pd.Series(df_2000_2020[17].values, index = index_date, dtype = 'float')       # Depth of snow
    snowe_depth = pd.Series(df_snowe['depth'].values, index = dtime, dtype = 'float')     # Depth of snow (SnoWE)
    snowe_rho = pd.Series(df_snowe['rho'].values, index = dtime, dtype = 'float')         # Snow density of snow (SnoWE)
    snowe_swe = pd.Series(df_snowe['swe'].values, index = dtime, dtype = 'float')         # Snow water equivalent of snow (SnoWE)
    
    #Грубая система коррекции исходных метеорологических данных
    
    # Коррекция данных о высоте метеостанции
    for height_st, a in enumerate(h_station):
        if a > 1000:                                                           # 1000 м.б.с - взято условно, поскольку основные станции расположены на равнине
            h_station[height_st] = h_station[height_st - 1]
        elif (h_station[height_st] - h_station[height_st-1]) > 5:
            h_station[height_st] = (h_station[height_st-1]+ h_station[height_st+1])/2
    
    # Коррекция данных по атмосферному давлению на станции           
    for ps_st, b in enumerate(ps):
        if b > 1060:                                                           # 1060 гПа - max значение, без учета экстремальных случаев
            ps[ps_st] = (ps[ps_st+1]+ps[ps_st-1])/2
        elif b < 950:                                                          # 950 гПа - min значение, без учета экстремальных случаев   
            ps[ps_st] = (ps[ps_st+1]+ps[ps_st-1])/2
    
    # Коррекция данных по атмосферному давлению приведенного к уровню моря
    for ps_pmsl, c in enumerate(pmsl):
        if c > 1060:                                                           # 1060 гПа - max значение, без учета экстремальных случаев
            pmsl[ps_pmsl] = (pmsl[ps_pmsl+1]+pmsl[ps_pmsl-1])/2
        elif c < 950:                                                          # 950 гПа - min значение, без учета экстремальных случаев   
            pmsl[ps_pmsl] = (pmsl[ps_pmsl+1]+pmsl[ps_pmsl-1])/2 

    #Коррекция данных по приземной температуре воздуха
    for t2m_st,d in enumerate(t2m):
        if d > 35:
            t2m[t2m_st] = t2m[t2m_st-1]
        elif d < -30:
            t2m[t2m_st] = t2m[t2m_st-1]
        elif (t2m[t2m_st] - t2m[t2m_st-1]) > 20:
            t2m[t2m_st] = (t2m[t2m_st-1]+ t2m[t2m_st+1])/2        
           
    #Коррекция данных по приземной температуре воздуха с целью создания только отрицательных температур
    for t2m_neg_st, e in enumerate(t2m_negative):
        if e > 0:
            t2m_negative[t2m_neg_st] = np.nan

    #Коррекция данных по температуре точке росы        
    for td2m_st, f in enumerate(td2m):
        if (td2m[td2m_st]-td2m[td2m_st-1]) > 35:
            td2m[td2m_st] = t2m[td2m_st-1]
        elif (td2m[td2m_st]-td2m[td2m_st-1]) < -35:
            td2m[td2m_st] = td2m[td2m_st-1]    
           
    #Коррекция данных по направлению ветра
    for dd10m_st, g in enumerate(dd10m):
        if g > 360:
            dd10m[dd10m_st] = 0
        elif g < 0:
            dd10m[dd10m_st] = 0
            
    #Коррекция данных по скорости ветра
    for ff10m_st, h in enumerate(ff10m):
        if h > 50:
            ff10m[ff10m_st] = ff10m[ff10m_st-1]
        elif h < 0:
            ff10m[ff10m_st] = ff10m[ff10m_st-1]         

    #Коррекция данных по минимальной температуре воздуха
    for tmin2m_st, i in enumerate(tmin2m):
        if i > 50:
            tmin2m[tmin2m_st] = np.nan
        elif i < -50:
            tmin2m[tmin2m_st] = np.nan
        elif (tmin2m[tmin2m_st]-tmin2m[tmin2m_st-1]) > 35:
            tmin2m[tmin2m_st] = tmin2m[tmin2m_st-1]
        elif (tmin2m[tmin2m_st]-tmin2m[tmin2m_st-1]) < -35:
            tmin2m[tmin2m_st] = tmin2m[tmin2m_st-1]

    #Коррекция данных по максимальной температуре воздуха
    for tmax2m_st, j in enumerate(tmax2m):
        if j > 50:
            tmax2m[tmax2m_st] = np.nan
        elif j < -50:
            tmax2m[tmax2m_st] = np.nan        
        elif (tmax2m[tmax2m_st]-tmax2m[tmax2m_st-1]) > 35:
            tmax2m[tmax2m_st] = tmax2m[tmax2m_st-1]
        elif (tmax2m[tmax2m_st]-tmax2m[tmax2m_st-1]) < -35:
            tmax2m[tmax2m_st] = tmax2m[tmax2m_st-1]    

    #Коррекция данных по минимальной температуре почвы за ночь
    for tming_st, k in enumerate(tming):
        if k > 40:
            tming[tming_st] = tming[tming_st-1]
        elif k < -35:
            tming[tming_st] = tming[tming_st-1]                    
        elif (tming[tming_st]-tming[tming_st-1]) > 35:
            tming[tming_st] = tming[tming_st-1]

    #Коррекция данных по температуре почвы
    for t_g_st, l in enumerate(t_g):
        if l > 40:
            t_g[t_g_st] = t_g[t_g_st-1]
        elif l < -35:
            t_g[t_g_st] = t_g[t_g_st-1]
        elif (t_g[t_g_st]-t_g[t_g_st-1]) > 35:
            t_g[t_g_st] = t_g[t_g_st-1]
            
    #Коррекция данных по высоте снега 
    for hsnow_st, m in enumerate(hsnow):
        if m > 200:
            hsnow[hsnow_st] = np.nan        
        elif (hsnow[hsnow_st]-hsnow[hsnow_st-1]) > 75:
            hsnow[hsnow_st] = hsnow[hsnow_st-1]    
        if np.isnan(hsnow[hsnow_st]) and not np.isnan(hsnow[hsnow_st-1]) and not np.isnan(hsnow[hsnow_st+1]) :
            hsnow[hsnow_st] = (hsnow[hsnow_st-1]+hsnow[hsnow_st+1])/2
    
    for snowe_depth_st, kk in enumerate(snowe_depth):    
        if kk == np.isnan(snowe_depth[snowe_depth_st]) and snowe_depth[snowe_depth_st-1] != 0 and snowe_depth[snowe_depth_st+1] != 0:
            
            snowe_depth[snowe_depth_st] = (snowe_depth[snowe_depth_st - 1] + snowe_depth[snowe_depth_st + 1])/2
        #if np.isnan(hsnow[hsnow_st]) and np.isnan(hsnow[hsnow_st-1]) and np.isnan(hsnow[hsnow_st+1]) :
            #hsnow[hsnow_st] = np.nan

    #Коррекция данных по сумме осадков за 12 часов
    for R12_st, n in enumerate(R12):
        if n > 99:
            #print (R12[R12_st].index, 'axtung')
            R12[R12_st] = np.nan
        elif n < 0:
            R12[R12_st] = np.nan        
        elif (R12[R12_st]-R12[R12_st-1]) > 50:
        #elif (n-(n-1)) > 50:    
            R12[R12_st] = (R12[R12_st+1]+R12[R12_st-1])/2 

    #Коррекция данных по сумме осадков за 12 часов liquid
    for R12_liq, liq in enumerate(R12_liquid):
        if liq > 99:
            R12_liquid[R12_liq] = 0
        elif liq < 0:
            R12_liquid[R12_liq] = 0
        elif (R12_liquid[R12_liq] - R12_liquid[R12_liq-1]) > 50:
            R12_liquid[R12_liq] = (R12_liquid[R12_liq+1]+R12_liquid[R12_liq-1])/2
            
        if t2m[R12_liq] < -1.2:
            R12_liquid[R12_liq] = np.nan      

    #Коррекция данных по сумме осадков за 12 часов solid
    for R_12_sol, sol in enumerate(R12_solid):
        if sol > 99:
            R12_solid[R_12_sol] = 0
        elif sol < 0:
            R12_solid[R_12_sol] = 0
        elif (R12_solid[R_12_sol] - R12_solid[R_12_sol-1]) > 50:
            R12_solid[R_12_sol] = (R12[R_12_sol+1]+R12[R_12_sol-1])/2        
        
        if t2m[R_12_sol] > -1.2:
            R12_solid[R_12_sol] = np.nan
      
        #Коррекция данных по сумме осадков за 24 часа
    for R24_st, o in enumerate(R24):
        if o > 99:
            R24[R24_st] = 0
        elif o < 0:
            R24[R24_st] = 0        
        elif (R24[R24_st]-R24[R24_st-1]) > 50:
            R24[R24_st] = (R24[R24_st+1]+R24[R24_st-1])/2         

    #Выполняем процедуру усреднения данных к суточной дискретности по времени
    id_st = index_meteo.resample('d').mean()  
    lat = lat.resample('d').mean()    
    lon = lon.resample('d').mean()
    h_station = h_station.resample('d').mean()
    ps = ps.resample('d').mean()
    pmsl = pmsl.resample('d').mean()
    t2m = t2m.resample('d').mean()
    t2m_negative = t2m_negative.resample('d').mean()
    tmin2m = tmin2m.resample('d').mean()
    tmax2m = tmax2m.resample('d').mean()
    tming = tming.resample('d').mean()    
    t_g = t_g.resample('d').mean()
    td2m = td2m.resample('d').mean()
    dd10m = dd10m.resample('d').mean()
    ff10mean = ff10m.resample('d').mean()
    ff10max = ff10m.resample('d').max()
    hsnow = hsnow.resample('d').mean()    
    R12 = R12.resample('d').sum()
    R12_liquid = R12_liquid.resample('d').sum()
    R12_solid = R12_solid.resample('d').sum()   
    R24 = R24.resample('d').sum()
    
    #Создаем новый массив с данными
    df_data = pd.concat([id_st, lat, lon, h_station, ps, pmsl, t2m, t2m_negative,
                         tmin2m, tmax2m, tming, t_g, td2m, dd10m, ff10mean, ff10max,
                         hsnow, snowe_depth, snowe_rho, snowe_swe, R12, R12_liquid, 
                         R12_solid, R24], axis = 1)
    
    # Экспортируем новый массив данных в формате csv
    df_data.to_csv(result_exit + fileName_csv[0:5] +'.csv', sep=';', float_format='%.3f',
                   header = ['id_st','lat','lon','h_station','ps','pmsl','t2m','t2m_negative',
                             'tmin2m','tmax2m','tming','t_g','td2m','dd10m','ff10meam','ff10max',
                             'hsnow','hsnow_snowe','rho','swe','R12','R12_liquid','R12_solid',
                             'R24'], index_label = 'Date')    

# На данном шаге, подготовительный этап можно считать завершенным. В результате его работы были сформированы временные ряды с данными и полученны все интересующие для 
# дальнейшей работы параметры. Следует также отметить, что данный этап можно выполнить только один раз перед началом новой работы, после чего использовать полученные 
# материалы для дальнейшей работы.
"""    

# Фаза работы 2: Описательная статистика. 

#Входные данные
data_stat = path_main + 'meteo_2000_2020/'
dirs_csv_stat = sorted(os.listdir(data_stat))

#Выходные данные
result_exit = path_main + 'Statistica/'

#Очистка предыдуших результатов работы
dirs_exit = os.listdir(result_exit)
for file in dirs_exit:
    os.remove(result_exit + file)

#Работа с временным рядом по метеостанции
for data_file in dirs_csv_stat:
    fileName_csv = data_file
    iPath_stat = (data_stat + fileName_csv)
    
    # Считываем данные из модели SnoWE
    df_stat = pd.read_csv(iPath_stat, skiprows = 0, sep=';', dayfirst = True, parse_dates = True, index_col = [0],
                          skipinitialspace = True, na_values= ['9990','********'])  
    w = 20   
    periods_winter = [['2000-10-01','2001-04-30'],
                      ['2001-10-01','2002-04-30'],
                      ['2002-10-01','2003-04-30'],
                      ['2003-10-01','2004-04-30'],
                      ['2004-10-01','2005-04-30'],
                      ['2005-10-01','2006-04-30'],
                      ['2006-10-01','2007-04-30'],
                      ['2007-10-01','2008-04-30'],
                      ['2008-10-01','2009-04-30'],
                      ['2009-10-01','2010-04-30'],
                      ['2010-10-01','2011-04-30'],
                      ['2011-10-01','2012-04-30'],
                      ['2012-10-01','2013-04-30'],
                      ['2013-10-01','2014-04-30'],
                      ['2014-10-01','2015-04-30'],
                      ['2015-10-01','2016-04-30'],
                      ['2016-10-01','2017-04-30'],
                      ['2017-10-01','2018-04-30'],
                      ['2018-10-01','2019-04-30'],
                      ['2019-10-01','2020-04-30']]                                            
    periods_winter = np.array(periods_winter)
    
    for tr in range(w):
        # Создаем лог для вывода ошибки. В частности, в данном логе будет фиксироваться случаи когда данные в рассматриваемом году были, но при этом они не соответствуют
        # условиям отбора, то формируются текст в log файл с ошибками. Данная информация будет представлять для нас повышенный интерес, поскольку в ней будут фиксироваться
        # года, когда устойчивых морозных периодов в течении зимы не наблюдалось, в результате чего и как таковые оттепели полностью отсутствуют. Либо если данные за период 
        # отсутствую, то формируется текст в log файл с ошибками. 
        
        logging.basicConfig(filename='D:/Don/logs.log', level=logging.DEBUG, 
                            format='%(levelname)s %(name)s %(message)s')
        
        try:
            y_w_1 = periods_winter[tr][0]
            y_w_2 = periods_winter[tr][1]
            ts_id_st = df_stat['id_st'][y_w_1:y_w_2]
            ts_lat = df_stat['lat'][y_w_1:y_w_2]
            ts_lon = df_stat['lon'][y_w_1:y_w_2]
            ts_h_station = df_stat['h_station'][y_w_1:y_w_2]
            ts_ps = df_stat['ps'][y_w_1:y_w_2]
            ts_pmsl = df_stat['pmsl'][y_w_1:y_w_2]
            ts_t2m = df_stat['t2m'][y_w_1:y_w_2].interpolate()
            ts_t2m_negative = df_stat['t2m_negative'][y_w_1:y_w_2]
            ts_t2m_min = df_stat['tmin2m'][y_w_1:y_w_2]
            ts_t2m_max = df_stat['tmax2m'][y_w_1:y_w_2].interpolate()
            ts_tming = df_stat['tming'][y_w_1:y_w_2]  
            ts_t_g = df_stat['t_g'][y_w_1:y_w_2]
            ts_td2m = df_stat['td2m'][y_w_1:y_w_2]
            ts_dd10m = df_stat['dd10m'][y_w_1:y_w_2]
            ts_ff10mean = df_stat['ff10meam'][y_w_1:y_w_2]
            ts_ff10max = df_stat['ff10max'][y_w_1:y_w_2]
            ts_hsnow = df_stat['hsnow'][y_w_1:y_w_2] 
            ts_hsnow_snowe_sd = df_stat['hsnow_snowe'][y_w_1:y_w_2].interpolate()
            ts_hsnow_snowe_rho = df_stat['rho'][y_w_1:y_w_2]
            ts_hsnow_snowe_swe = df_stat['swe'][y_w_1:y_w_2].interpolate()
            ts_R12 = df_stat['R12'][y_w_1:y_w_2]
            ts_R12_liquid = df_stat['R12_liquid'][y_w_1:y_w_2]
            ts_R12_solid = df_stat['R12_solid'][y_w_1:y_w_2]  
            ts_R24 = df_stat['R24'][y_w_1:y_w_2]
            
            # Алгоритм 1 - Определяем границы устойчивого морозного периода
            
            # Определяем дату начала устойчивого морозного периода по максимальной среднесуточной температуре воздуха
            # по условию считается, что за начало морозного периода принимается 5 последовательных дней с отрицательной температурой воздуха
            # также в условии говорится, что окончанием морозного периода является устойчивый переход среднесуточной температуры через 0
            
            # Важно по Т2m_max - определяем только начало устойчивого морозного периода
            EMPTY=-1
            COLD_START = 5                                                     # 5 дней холода, после чего начинаеся устойчивый морозный период
            HOT_STOP = 20                                                      # 30 дней с положительной температурой воздуха, после которой устойчивый морозный период заканчиваеся           
            plusCount=0                                                        # счетчик положительных температур
            count=0                                                            # общий счетчик случая
            negativStart = -1                                                  # начало холодного периода 
            negativEnd = -1                                                    # конец холодного периода    
            neg_t_start = []                                                   # список с датами начала холодного периода
            neg_t_stop = []                                                    # список с датами конца холодного периода
            for j in range(len(ts_t2m_max)):
                if ts_t2m_max[j] < 0:
                    plusCount = 0
                    count += 1
                    if (count == COLD_START):
                        negativStart = j - COLD_START + 1 
                        neg_t_start.append(negativStart)
                else:
                    if (negativStart != EMPTY):
                        count += 1
                        plusCount += 1
                        if (plusCount >= HOT_STOP):    #month                                
                            negativEnd = j - HOT_STOP+1
                            neg_t_stop.append(negativEnd)                       
                            break  
                    else:
                        count=0                
                if (j == (len(ts_t2m_max) - 1)):
                    if (negativStart != EMPTY):
                        negativEnd = j                     
                        neg_t_stop.append(negativEnd)
            
            # Важно по Т2m - определяем конец устойчивого морозного периода
            EMPTY_T2M = -1
            COLD_START_T2M = 5                                                     # 5 дней холода, после чего начинаеся устойчивый морозный период
            HOT_STOP_T2M = 20                                                      # 25 дней с положительной температурой воздуха, после которой устойчивый морозный период заканчиваеся           
            plusCount_t2m = 0                                                      # счетчик положительных температур
            count_t2m = 0                                                          # общий счетчик случая
            negativStart_t2m = -1                                                  # начало холодного периода 
            negativEnd_t2m = -1                                                    # конец холодного периода    
            neg_t_start_t2m = []                                                   # список с датами начала холодного периода
            neg_t_stop_t2m = []                                                    # список с датами конца холодного периода
            
            for jj in range(len(ts_t2m)):
                if ts_t2m[jj] < 0:
                    plusCount_t2m = 0
                    count_t2m += 1
                    if (count_t2m == COLD_START_T2M):
                        negativStart_t2m = jj - COLD_START_T2M + 1 
                        neg_t_start_t2m.append(negativStart_t2m)
                else:
                    if (negativStart_t2m != EMPTY_T2M):
                        count_t2m += 1
                        plusCount_t2m += 1
                        if (plusCount_t2m >= HOT_STOP_T2M):    #month                                
                            negativEnd_t2m = jj - HOT_STOP_T2M + 1
                            neg_t_stop_t2m.append(negativEnd_t2m)                       
                            break  
                    else:
                        count_t2m=0                
                if (jj == (len(ts_t2m) - 1)):
                    if (negativStart_t2m != EMPTY_T2M):
                        negativEnd_t2m = jj                                                
                        neg_t_stop_t2m.append(negativEnd_t2m)
                        
            
            # Создаем условие при котором проверяем даты окончания случая
            if negativEnd_t2m >= negativEnd:
                negativEnd_t2m = negativEnd_t2m
            else:
                negativEnd_t2m = negativEnd            
            
            # Создаем счетчик для новых индексов
            time_period =  negativEnd_t2m - negativStart
            new_index = []
            for k in range(time_period):
                new_index.append(k)

            #Извлекаем из исходных рядов данные за устойчивый морозный период
            stat_cold_period = pd.Series(ts_t2m[negativStart:negativEnd_t2m])               # данные о температуре воздуха T2m_mean
            cold_period_sd = pd.Series(ts_hsnow_snowe_sd[negativStart:negativEnd_t2m])      # данные о высоте снега 
            cold_period_swe = pd.Series(ts_hsnow_snowe_swe[negativStart:negativEnd_t2m])    # данные о запасе воды в снеге 
            cold_period_r12_liquid = pd.Series(ts_R12_liquid[negativStart:negativEnd_t2m])  # данные о жидких осадках 
            cold_period_r12_solid = pd.Series(ts_R12_solid[negativStart:negativEnd_t2m])    # данные о твердых осадках 
            
            
            #Извлекаем индексы устойчивого холодного периода
            date_stat_cold_period = stat_cold_period.index        
            date_stat_cold_period = date_stat_cold_period.to_series()                       # Переводим из TimeDateIndex to Series
            
            
            #Создаем новый дата фрейм для переиндексации                                    #Датафрейм с интересующими переменными за устойчивый морозный период
            df_data_cold_period = pd.concat([stat_cold_period, date_stat_cold_period,
                                             cold_period_sd, cold_period_swe, cold_period_r12_liquid,
                                             cold_period_r12_solid],axis = 1) 
                       
            #Извлекаем из нового датафрейма значения и переиндексируем
            t2m_new_index = pd.Series(df_data_cold_period['t2m'].values, index = new_index, dtype = 'float')
            date_new_index = pd.Series(df_data_cold_period['Date'].values.astype('datetime64[D]'))
            sd_new_index = pd.Series(df_data_cold_period['hsnow_snowe'].values, index = new_index, dtype = 'float')
            swe_new_index = pd.Series(df_data_cold_period['swe'].values, index = new_index, dtype = 'float')
            r12_liq_new_index = pd.Series(df_data_cold_period['R12_liquid'].values, index = new_index, dtype = 'float')
            r12_sol_new_index = pd.Series(df_data_cold_period['R12_solid'].values, index = new_index, dtype = 'float')
                                                                   
            #Создаем новый пустой массив длины stat_cold_period, массив пустой. Нужно для определения признака оттепели или ее отсутствия
            dummyarray = np.empty((len(stat_cold_period),1))
            dummyarray[:] = np.nan
            df_zero = pd.DataFrame(dummyarray)
            
            #Объединяем пустой дата фрейм с данными за устойчивый морозный период            #Примечания - новые индексы 
            df_cold_period = pd.concat([t2m_new_index, date_new_index, sd_new_index,
                                        swe_new_index, r12_liq_new_index, r12_sol_new_index,
                                        df_zero], axis = 1)      
            df_cold_period.columns = ['t2m','Date','sd','swe','R12_liq','R12_sol','status']

            # Алгоритм 2 - Заполнение признака оттепели
            
            #Заполняем столбец статуса для создания новых индексов
            for pp in range(len(df_cold_period)):                                                    # Перебираем строки в датафрейме
                if (df_cold_period['t2m'][pp] >= 0):
                    df_cold_period['status'][pp] = 1
                else:
                    df_cold_period['status'][pp] = 0
                    if (pp == 0):
                        if (df_cold_period['t2m'][pp + 1] >= 0):
                            df_cold_period['status'][pp] = 1
                    elif (pp == (len(df_cold_period) - 1)):
                        if (df_cold_period['t2m'][pp - 1] >= 0):
                            df_cold_period['status'][pp] = 1
                    else:
                        if ((df_cold_period['t2m'][pp - 1] >= 0) and (df_cold_period['t2m'][pp + 1] >= 0)):
                            df_cold_period['status'][pp] = 1
                
            # Извлекаем столбец с датами с целью создания нового индекса по времени
            date_index_cold = pd.to_datetime(df_cold_period['Date'])
            y6 = pd.Series(df_cold_period['t2m'].values, index = date_index_cold, dtype = 'float')         # T2m с новым индексом            
            y7 = pd.Series(df_cold_period['sd'].values, index = date_index_cold, dtype = 'float')          # sd с новым индексом
            y8 = pd.Series(df_cold_period['swe'].values, index = date_index_cold, dtype = 'float')         # swe с новым индексом
            y9 = pd.Series(df_cold_period['R12_liq'].values, index = date_index_cold, dtype = 'float')     # r12_liq с новым индексом
            y10 = pd.Series(df_cold_period['R12_sol'].values, index = date_index_cold, dtype = 'float')    # r12_sol с новым индексом
            y11 = pd.Series(df_cold_period['status'].values, index = date_index_cold, dtype = 'float')     # Дата с новым индексом
            
            # Создаем второй датафрейм с индексом по реальным датам           
            df_cold_period_1 = pd.concat([y6, y7, y8, y9, y10, y11], axis = 1)                             
            df_cold_period_1.columns = ['t2m','sd', 'swe','R12_liq', 'R12_sol', 'status']
                      
            # Алгоритм 3. Определение количества теплых дней внутри устойчивого морозного периода
            
            period_positive = False                                                 # Положительный период
            start_positive = []                                                     # Индекс начала оттепелей
            step_positive = []                                                      # Период оттепели
            duration_positive = 1                                                   # Счетчик дней в оттепели
            
                      
            for i in range(len(df_cold_period_1)):
                if df_cold_period_1['status'][i] >= 1:
                    if (period_positive == False):
                        period_positive = True
                        start_positive.append(i)
                        duration_positive = 1
                    else:
                        duration_positive = duration_positive + 1                                                                    
                    if i == (len(df_cold_period_1)-1):
                        #print ('POS 1')
                        if duration_positive < 30:
                            step_positive.append(duration_positive)
                        else:
                            del (start_positive[-1])                
                        
                        #test_snow = np.asarray(start_positive)
                        #for l in range(len(test_snow)):
                           # print ('text - snowe', df_cold_period_1['t2m'][l], 'index', df_cold_period_1.index[l])
                        #for s in range(len(aa)):
                            #print (sum(df_cold_period[aa[s]:(ac[s]+1)]))
                elif df_cold_period_1['status'][i] == 0:
                    if (period_positive == True):
                        if  duration_positive < 30:
                            step_positive.append(duration_positive)                
                        else:
                            del(start_positive[-1])            
                        period_positive = False  

            t2m_t_start = []                                                        # Температура на момент начала случая
            t2m_t_stop = []                                                         # Температура на конец случая
            t2m_mean = []                                                           # Средняя температура 
            
            sd_t_start = []                                                         # Высота снега на момент начала случая
            sd_t_stop = []                                                          # Высота снега на конец случая          
            sd_mean = []                                                            # Средняя высота снега 
            
            swe_t_start = []                                                        # запас воды в снеге на момент начала случая
            swe_t_stop = []                                                         # запас воды в снеге на конец случая          
            swe_mean = []                                                           # Средний запас воды в снеге 
            
            R12_liq_t_start = []                                                    # Жидкие осадки на момент начала случая
            R12_liq_t_stop = []                                                     # Жидкие осадки на конец случая    
            R12_liq_mean = []                                                       # Жидкие осадки за случай

            R12_sol_t_start = []                                                    # Твердые осадки на момент начала случая
            R12_sol_t_stop = []                                                      # Твердые осадки на конец случая
            R12_sol_mean = []                                                       # Твердые осадки за случай
                        
            #Создаем вывод для устойчивого морозного периода (УМП) и оттепелей
            try:
                start_cold_period = np.asarray(neg_t_start)                         # создаем массив из индексов начала УМП из T2m_max                
                stop_cold_period = np.asarray(neg_t_stop_t2m)                       # создаем массив из индексов конца УМП из T2m               
                cold_time_period = negativEnd_t2m - negativStart                    # находим продолжительность случая по разности между началом и концом
                                
                first_pos_day = np.asarray(start_positive)                          # создаем массив из индексов начала оттепели из T2m                                             
                time_step_pos = np.asarray(step_positive)                           # создаем массив из продолжительности случая
                last_pos_day = first_pos_day + (time_step_pos - 1)                  # вычисляем индексы конца оттепели по T2m
                last_pos_day_list = last_pos_day.tolist()                           # создаем массив индексов конца оттепелей
                
                
                # Создаем счетчик суммы количества дней оттепелей, которые наблюдались в УМП 
                numbers_thaw = 0
                summa_thaw = []
                for l in time_step_pos:
                    numbers_thaw = numbers_thaw + l
                summa_thaw.append(numbers_thaw)                
                summe_thaw = np.asarray(summa_thaw)                                 # Создаем  массив суммы количества дней оттепели               
                                                 
                # Создаем серии и извлекаем данные из первоначальных данных
                date_cold_start = pd.Series(ts_t2m_max.index[start_cold_period])    # дата начала УМП - используем ряд T2m_max             
                date_cold_stop = pd.Series(ts_t2m.index[stop_cold_period])          # дата конца УМП  - используем ряд T2m                     
                date_cold_period = pd.Series(cold_time_period)                      # сумма дней устойчивого холодного периода                
                days_thaw = pd.Series(summe_thaw)                                   # сумма дней с оттепелью
                
                k = ((date_cold_period - days_thaw)/date_cold_period)*100           # рассчитываем индекс устойчивости морозного периода
                
                # Вывод информации по УМП
                df_data_ump = pd.concat([date_cold_start, date_cold_stop, date_cold_period,days_thaw, k],axis = 1)            
                df_data_ump.to_csv(result_exit + fileName_csv[0:5] + '_cold_period' + '.csv', mode = 'a', sep=';',float_format='%.0f',
                                    header = ['Start_cold',u'End_cold','Cold_period','Days_thaw','K, %'])
                                
                
                #Рассчитываем оттепели и изменение характеристик снега за оттепели
                start_warm_period = pd.Series(first_pos_day)                                  # даты начала оттпели
                stop_warm_period = pd.Series(last_pos_day_list)                               # даты конца оттпели
                
                date_warm_start = pd.Series(df_cold_period_1.index[start_warm_period])        # индексы начала оттепели        
                date_warm_stop = pd.Series(df_cold_period_1.index[stop_warm_period])          # индексы конца оттепели                             
                date_warm_period = pd.Series(time_step_pos)                                   # продолжительность оттепелей

                # Определение метео характеристик на момент начала оттепели 
                for limbo in start_warm_period:                            
                    t2m_t_start.append(df_cold_period_1['t2m'][limbo])                       # T2m - на момент случая
                    sd_t_start.append(df_cold_period_1['sd'][limbo])                          # sd - на момент случая
                    swe_t_start.append(df_cold_period_1['swe'][limbo])                        # swe - на момент случая
                    R12_liq_t_start.append(df_cold_period_1['R12_liq'][limbo])                # R12_liq - на момент случая
                    R12_sol_t_start.append(df_cold_period_1['R12_sol'][limbo])                  # R12_sol - на момент случая 

                    
                for stimbo in stop_warm_period:
                    t2m_t_stop.append(df_cold_period_1['t2m'][stimbo])                       # T2m - на конец случая
                    sd_t_stop.append(df_cold_period_1['sd'][stimbo])                          # sd - на конец случая
                    swe_t_stop.append(df_cold_period_1['swe'][stimbo])                        # swe - на конец случая
                    R12_liq_t_stop.append(df_cold_period_1['R12_liq'][stimbo])                # R12_liq - на конец случая
                    R12_sol_t_stop.append(df_cold_period_1['R12_sol'][stimbo])                  # R12_sol - на конец случая 
                            
                                                                                
                for numbers_periods in range(len(start_warm_period)):
                    t2m_means = np.mean(df_cold_period_1['t2m'][start_warm_period[numbers_periods]:(stop_warm_period[numbers_periods] + 1)])                                                                        
                    t2m_mean.append(t2m_means)                                                    
                    sd_means = np.mean(df_cold_period_1['sd'][start_warm_period[numbers_periods]:(stop_warm_period[numbers_periods] + 1)])                                 
                    sd_mean.append(sd_means)                                                   
                    swe_means = np.mean(df_cold_period_1['swe'][start_warm_period[numbers_periods]:(stop_warm_period[numbers_periods] + 1)])                                 
                    swe_mean.append(swe_means)                      
                    R12_liq_means = np.sum(df_cold_period_1['R12_liq'][start_warm_period[numbers_periods]:(stop_warm_period[numbers_periods] + 1)])                                 
                    R12_liq_mean.append(R12_liq_means)                     
                    R12_sol_means = np.sum(df_cold_period_1['R12_sol'][start_warm_period[numbers_periods]:(stop_warm_period[numbers_periods] + 1)])                                 
                    R12_sol_mean.append(R12_sol_means) 

                # Создание серий для загрузки их в датафрейм (рассматриваем случае для оттепелей)
                
                thaw_t2m_start = pd.Series(t2m_t_start)
                thaw_t2m_stop = pd.Series(t2m_t_stop)
                thaw_t2m_mean = pd.Series(t2m_mean)
                
                thaw_sd_start = pd.Series(sd_t_start)
                thaw_sd_stop = pd.Series(sd_t_stop)
                thaw_sd_mean = pd.Series(sd_mean)
                h_snow_sd_delta = thaw_sd_start - thaw_sd_stop
                
                thaw_swe_start = pd.Series(swe_t_start)
                thaw_swe_stop = pd.Series(swe_t_stop)
                thaw_swe_mean = pd.Series(swe_mean)
                h_snow_swe_delta = thaw_swe_start - thaw_swe_stop

                thaw_R12_liq_start = pd.Series(R12_liq_t_start)
                thaw_R12_liq_stop = pd.Series(R12_liq_t_stop)
                thaw_R12_liq_mean = pd.Series(R12_liq_mean)                
                
                thaw_R12_sol_start = pd.Series(R12_sol_t_start)
                thaw_R12_sol_stop = pd.Series(R12_sol_t_stop)
                thaw_R12_sol_mean = pd.Series(R12_sol_mean)                 
                
         
                
                # Вывод информации по оттепелям внутри каждого УМП
                df_data_thaw = pd.concat([date_warm_start, date_warm_stop, date_warm_period,
                                          thaw_t2m_start, thaw_t2m_stop, thaw_t2m_mean,
                                          thaw_sd_start,  thaw_sd_stop,  thaw_sd_mean,  h_snow_sd_delta,
                                          thaw_swe_start, thaw_swe_stop, thaw_swe_mean, h_snow_swe_delta,
                                          thaw_R12_liq_start, thaw_R12_liq_stop, thaw_R12_liq_mean,
                                          thaw_R12_sol_start, thaw_R12_sol_start, thaw_R12_sol_mean],axis = 1)                                                      #Датафрейм с информацией об оттепелях


                df_data_thaw.to_csv(result_exit + fileName_csv[0:5] + '_thaw' + '.csv', mode = 'a', sep=';', float_format='%.2f',
                                    header = ['Start_thaw',u'End_thaw','Duration',
                                              'T2m_start','T2m_end','T2m_mean',
                                              'SD_start', 'SD_end', 'SD_mean', 'SD_delta',
                                              'SWE_start', 'SWE_end','SWE_mean','SWE_delta',
                                              'R12_liq_start', 'R12_liq_end', 'R12_liq_mean',
                                              'R12_sol_start', 'R12_liq_end', 'R12_sol_mean'])            

            #Если данные за период отсутствую, то формируется текст в log файл с ошибками            
            except IndexError as error:
                logging.warning('No data for ' + fileName_csv[0:5] + ' from ' + y_w_1 + ' to ' + y_w_2)
                logger=logging.getLogger(__name__)        
                logger.error(error)

                
        #Если данные в рассматриваемом году были, но при этом они не соответствуют условиям отбора, то формируются текст в log файл с ошибками. Важно
        except ValueError as error:
            logging.info('No negative 5 days temperature ' + fileName_csv[0:5])
            logger=logging.getLogger(__name__)
            logger.error(error)
