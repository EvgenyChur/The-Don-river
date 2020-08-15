"""
@author: Churiulin Evgenii
Скрипт предназначен для запуска алгоритма машинного обучения Random Forest, метода главных компонент и факторного анализа
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis


#########################
# Функция 1 для коррекции пустых значений с учетом предыдущего и следующего значения
#########################

# colums_df_stat - имя переменной (тип - объект Series), полученной на основе выгрузки данных из исходного массива данных
def nan_values_correction(colums_df_stat):
    for tmin2m_st, k in enumerate(colums_df_stat):
        if (tmin2m_st < (len(colums_df_stat)-1)):       
            if np.isnan(colums_df_stat[tmin2m_st]) and colums_df_stat[tmin2m_st-1] != 0 and colums_df_stat[tmin2m_st+1] != 0:
                colums_df_stat[tmin2m_st] = (colums_df_stat[tmin2m_st - 1] + colums_df_stat[tmin2m_st + 1])/2  
            elif np.isnan(colums_df_stat[tmin2m_st]) and colums_df_stat[tmin2m_st-1] != 0 and colums_df_stat[tmin2m_st+2] != 0:
                colums_df_stat[tmin2m_st] = (colums_df_stat[tmin2m_st - 1] + colums_df_stat[tmin2m_st + 2])/2
            elif np.isnan(colums_df_stat[tmin2m_st]) and colums_df_stat[tmin2m_st-1] != 0 and colums_df_stat[tmin2m_st+3] != 0:
                colums_df_stat[tmin2m_st] = (colums_df_stat[tmin2m_st - 1] + colums_df_stat[tmin2m_st + 3])/2                
        if tmin2m_st == (len(colums_df_stat)-1):            
            if np.isnan(colums_df_stat[tmin2m_st]):
                colums_df_stat[tmin2m_st] = (colums_df_stat[tmin2m_st - 1])
    return (colums_df_stat) 

#########################
# Функция 2 для коррекции пустых значений с учетом предыдущего, следующего, 2 следующих, 3 следующих значения
#########################

# colums_df_stat2 - имя переменной (тип - объект Series), полученной на основе выгрузки данных из исходного массива данных         
def nan_values_correction_2(colums_df_stat_2):
    for tt, kkk in enumerate(colums_df_stat_2):
        if (tt < (len(colums_df_stat_2)-1)):
            if np.isnan(colums_df_stat_2[tt]) and not np.isnan(colums_df_stat_2[tt-1]) and not np.isnan(colums_df_stat_2[tt+1]):
                colums_df_stat_2[tt] = (colums_df_stat_2[tt-1] + colums_df_stat_2[tt+1])/2   
            elif np.isnan(colums_df_stat_2[tt]) and not np.isnan(colums_df_stat_2[tt-1]) and not np.isnan(colums_df_stat_2[tt+2]):
                colums_df_stat_2[tt] = (colums_df_stat_2[tt-1] + colums_df_stat_2[tt+2])/2             
            elif np.isnan(colums_df_stat_2[tt]) and not np.isnan(colums_df_stat_2[tt-1]) and not np.isnan(colums_df_stat_2[tt+3]):
                colums_df_stat_2[tt] = (colums_df_stat_2[tt-1] + colums_df_stat_2[tt+3])/2
        if tt == (len(colums_df_stat_2)-1):
            if np.isnan(colums_df_stat_2[tt]):
                colums_df_stat_2[tt] = colums_df_stat_2[tt-1]
    return (colums_df_stat_2)            

# Примечание функции 1 и 2 можно заменить одной, но требуется доработка + дополнительно следует учесть когда пропуск стоит на 1 месте


#########################
# Функция 3 для коррекции пустых значения за период с мая по октябрь (не включая). Использовал для заполнения 
#########################

# пустых значений снежного покрова
# colums_snow - имя переменной (тип - объект Series), полученной на основе выгрузки данных из исходного массива данных    
def snow_values_correction(colums_snow):
    for h_st, kk in enumerate(colums_snow):
        month = colums_snow.index[h_st].month
        if month >= 5 and month <= 9:
            colums_snow[h_st] = 0
    return (colums_snow)

#########################
# Функция 4 для загрузки исходных метеорологических данных
#########################
    
# iPath - путь к данным   
def initial_data(iPath):
    # Считываем данные 
    df = pd.read_csv(iPath, skiprows = 0, sep=';', dayfirst = True, parse_dates = True)
    #print ('Columns:', df.columns)
    #Удаляем дубликаты и столбцы, которые не представляют интереса для данного исследования   
    df = df.drop_duplicates(keep = False)
    df = df.drop(['lat','lon','h_station','id_st','t2m_negative','hsnow'], axis=1)    
    #Создаем серии для заполнения пропусков в даннных
    index_date = pd.to_datetime(df['Date'])    # time step
    ps = pd.Series(df['ps'].values, index = index_date, dtype = 'float')                # air pressure at meteostation
    pmsl = pd.Series(df['pmsl'].values, index = index_date, dtype = 'float')            # air pressure at meteostation in according to see level
    t2m = pd.Series(df['t2m'].values, index = index_date, dtype = 'float')              # 2m air temperature
    tmin2m = pd.Series(df['tmin2m'].values, index = index_date, dtype = 'float')        # 2m min air temperature
    tmax2m = pd.Series(df['tmax2m'].values, index = index_date, dtype = 'float')        # 2m max air temperature   
    tming = pd.Series(df['tming'].values, index = index_date, dtype = 'float')          # min soil temperature for night 
    td2m = pd.Series(df['td2m'].values, index = index_date, dtype = 'float')            # 2m dew point
    t_g = pd.Series(df['t_g'].values, index = index_date, dtype = 'float')              # temperatura of soil
    dd10m = pd.Series(df['dd10m'].values, index = index_date, dtype = 'float')          # direction of wind
    ff10mean = pd.Series(df['ff10meam'].values, index = index_date, dtype = 'float')    # speed of wind
    ff10max = pd.Series(df['ff10max'].values, index = index_date, dtype = 'float')      # speed of wind
    hsnow = pd.Series(df['hsnow_snowe'].values, index = index_date, dtype = 'float')    # Depth of snow
    swe = pd.Series(df['swe'].values, index = index_date, dtype = 'float')              # SWE of snow
    rho = pd.Series(df['rho'].values, index = index_date, dtype = 'float')              # RHO of snow    
    # Запускаем функции коррекции данных
    ps = nan_values_correction(ps)
    pmsl = nan_values_correction(pmsl)
    t2m = nan_values_correction(t2m)
    tmin2m = nan_values_correction(tmin2m)
    tmax2m = nan_values_correction(tmax2m)
    tming = nan_values_correction(tming)
    tming = nan_values_correction_2(tming)
    td2m = nan_values_correction(td2m)
    t_g = nan_values_correction(t_g)
    t_g = nan_values_correction_2(t_g)
    for t_g_s, kk in enumerate(t_g):
        if (t_g_s < (len(t_g)-1)):
            if np.isnan(t_g[t_g_s]) and not np.isnan(tming[t_g_s]):
                t_g[t_g_s] = tming[t_g_s]           
        if t_g_s == (len(t_g)-1):
            if np.isnan(t_g[t_g_s]):
                t_g[t_g_s] = tming[t_g_s]
    dd10m = nan_values_correction(dd10m)
    ff10mean = nan_values_correction(ff10mean)
    ff10max = nan_values_correction(ff10max)
    hsnow = snow_values_correction(hsnow)
    swe = snow_values_correction(swe)
    rho = snow_values_correction(rho)    
    # Отбрасываем строки с пустыми значениям, где заполненных параметров меньше 15
    df = df.dropna(axis='rows', thresh=15)  
    # Выполняем переиндексирование по дате
    df['Date'] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')    
    # Устанавливаем новый индекс для массива данных
    df = df.set_index('Date')    
    # Заполняем оставшиеся пустые значения в столбцах средним значением по столбцу    
    #df = df.fillna(method='ffill')
    #df = df.fillna(df.mean())
    #df = df.fillna(0)        
    return (df)           

#########################
# Функция для выборки только зимних значений метеопараметров
#########################

# data_maket  - пустой массив данных, куда будут записываться данные
# df_data - массив с метеоданными
# name - текстовый параметр с именем столбца из df_data
# time_1 - переменная начала периода
# time_2 - переменная конца периода  
def winter_data(data_maket, df_data, name, time_1, time_2):
    if len(data_maket)>0:               
        data_maket = pd.concat([data_maket, df_data[name][time_1:time_2]])
    else:
        data_maket = df_data[name][time_1:time_2]
    return (data_maket)

#########################
# Функция выполняющая простейшее машинное обучение на основе метода RandomForest
#########################

# df_main_year или df_main_winter = df - массив по которому выполняем описание
# Data_path_exit - путь, где будет храниться результат работы    
def simple_machine_learning(df_data, Data_path_exit):
    # Выбрасываем интересующий нас столбец из подготовленного массива данных и создаем 2 новых переменных
    X = df_data.drop('Discharge', axis=1)
    y = df_data['Discharge']

    # Разделяем наши данные на тестовый набор данных и независимый набор данных для провеки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=2020)

    # Стандартизируем данные
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)

    # Переводим из объекта Series в объект numpy
    y_train = np.array(y_train)
    # Переводим из float 64 в int 32, нужно для корректной работы методов дальше
    y_train = y_train.astype(np.int32)

    # Подключаем алгоритм машинного обучения и выполняем обучение базовой модели 
    rfc = RandomForestClassifier()
    rfc.fit(X_train_scaled, y_train)
    #display(rfc.score(X_train_scaled, y_train))

    # Отображаем самые важные признаки для обучения 
    feats = {}
    # Требуется указать какие данные используются
    for feature, importance in zip(df_data.columns, rfc.feature_importances_):       
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale = 5)
    sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30,15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Значимость переменной', fontsize=25, weight = 'bold')
    plt.ylabel('Переменные', fontsize=25, weight = 'bold')
    #plt.title('Feature Importance', fontsize=25, weight = 'bold')
    plt.savefig(Data_path_exit + 'Importance factor' + '.png', format='png', dpi = 300)               
    # Отображаем графически важность признаков
    display(plt.show())
    # Делаем визуализацию важности признаков
    display(importances)
       
    ###### Запускаем метод главных компонент 
    pca_test = PCA(n_components=13)
    pca_test.fit(X_train_scaled)
    sns.set(style='whitegrid')
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.xlabel('Число компонент')
    plt.ylabel('Суммарно объясненная дисперсия')
    plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
    plt.savefig(Data_path_exit + 'PCA' + '.png', format='png', dpi = 300)    
    display(plt.show())
    evr = pca_test.explained_variance_ratio_
    cvr = np.cumsum(pca_test.explained_variance_ratio_)
    pca_df = pd.DataFrame()
    pca_df['Cumulative Variance Ratio'] = cvr
    pca_df['Explained Variance Ratio'] = evr
    #display(pca_df.head(10))    
    # Понижаем размерность исходного массива данных до n-компонент описывающих 95% дисперсии
    pca = PCA(n_components=10)
    pca.fit(X_train_scaled)
    X_train_scaled_pca = pca.transform(X_train_scaled)
    X_test_scaled_pca = pca.transform(X_test_scaled)
    
    # Создаем новый массив данных с распределением дисперсии по компонентам
    pca_dims = []
    for x in range(0, len(pca_df)):
        pca_dims.append('PCA Component {}'.format(x))
    pca_test_df = pd.DataFrame(pca_test.components_, columns=X.columns, index=pca_dims)
    pca_test_df.to_csv(Data_path_exit + 'PCA_result.csv', sep=';', float_format='%.3f')   # массив данных с распределением дисперсии по компонентам
    #print (pca_test_df.head(10).T)

#########################
# Функция для стандартизации переменных
#########################
    
# df - исходный массив с данными
def scale_features(df):
    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)
    return scaled

#########################
# Функция для выполнения факторного анализа
#########################
    
# df_main_year или df_main_winter = df - массив по которому выполняем описание
# Data_path_exit - путь к папкам, где будут храниться данные
# df2_index - массив с исходными индексами по дате (может быть 2 варианта либо df_main (4 сезона) или df_main1 (холодный сезон))
def factor_analysis(df, Data_path_exit, df2_index):
    # Выполняем расчет описательной статистики и корреляционной матрицы
    df_des_stat = df.describe()
    df_cor_stat = df.corr()
    # Делаем вывод описательной статистики и корреляционной матрицы
    df_des_stat.to_csv(Data_path_exit + 'des_stat.csv', sep=';', float_format='%.3f')
    df_cor_stat.to_csv(Data_path_exit + 'cor_stat.csv', sep=';', float_format='%.3f')
    # Строим диаграммы рассеивания и гистограммы
    matrix = scatter_matrix(df, figsize=[20,20], alpha=0.2)
    # Импортируем данные
    plt.savefig(Data_path_exit + 'Scatter_matrix' +'.png', format='png', dpi = 300)

    df_scaled = preprocessing.scale(df)                                        # массив со стандартизированными данными
    # Проецируем с метода главных компонент переменнные на плоскость. Выделяем 4 главных фактора (можно больше)
    pca = PCA(n_components=4)
    pca1 = pca.fit(df_scaled)
    print('Доля разброса, которую объясняют факторы: ', pca.explained_variance_ratio_)
       
    # Рассчитываем значения основных факторов
    zzz = pca.transform(df_scaled)
    values_factors = pd.DataFrame(zzz)
    values_factors.to_csv(Data_path_exit + 'factor_values.csv', sep=';', float_format='%.3f')    
    #print (zzz)

    # Факторный анализ
    fa = FactorAnalysis(n_components=4)                                        # Количество факторов
    fac_1 = fa.fit(df_scaled)
    df_fa = pd.DataFrame(fa.components_, columns=df.columns)                   
    df_fa.to_csv(Data_path_exit + 'factor_result.csv', sep=';', float_format='%.3f')   # Координаты факторов в пространстве исходных значений
    # Уникальность значений в смысле дисперсии, объяснённой факторами (чем больше, тем хуже объясняется факторами) содержится в атрибуте
    fac_2 = pd.Series(fa.noise_variance_, df.columns)
    fac_2.to_csv(Data_path_exit + 'Unic_values.csv', sep=';', float_format='%.3f')  # Координаты факторов. Основной результат    
    print ('Уникальность значений:\n', fac_2)
    scores = pd.DataFrame(fa.transform(df_scaled), columns=['factor1', 'factor2','factor3', 'factor4'])   
    scores = scores.set_index(df2_index.index)
    scores.to_csv(Data_path_exit + 'factor_vectors.csv', sep=';', float_format='%.3f')  # Координаты факторов. Основной результат


###### Этап 1. Подготовка начальных данных для проведения дальнейших вычислений

# Создание массивов метеоданных для водосборов

# Путь к папке, где хранится проект с рекой Дон
path_main = 'D:/Don/'
# Путь к папкам, куда записываются результирующие данные
iPath_result = path_main +'Main_data/'                                         #Результирующие массивы с метео данными
iPath_exit = path_main +'PCA/'                                                 #Результаты машинного обучения, PCA и факторного анализа

###### Река Сосна - г.п. Елец
iPath_stat_exit1 = iPath_exit + 'Sosna_river/annual_data/'                     #Результаты для всех сезонов
iPath_stat_exit2 = iPath_exit + 'Sosna_river/cold_data/'                       #Результаты для холодного сезона

###### Река Битюг - г.п. Бобров
iPath_stat_exit3 = iPath_exit + 'Bitug_river/annual_data/'                     #Результаты для всех сезонов
iPath_stat_exit4 = iPath_exit + 'Bitug_river/cold_data/'                       #Результаты для холодного сезона

###### Река Тихая Сосна - г.п. Алексеевка
iPath_stat_exit5 = iPath_exit + 'M_Sosna_river/annual_data/'                   #Результаты для всех сезонов
iPath_stat_exit6 = iPath_exit + 'M_Sosna_river/cold_data/'                     #Результаты для холодного сезона

###### Река Медведица - г.п. Лысые горы
iPath_stat_exit7 = iPath_exit + 'Medveditsa_river/annual_data/'                #Результаты для всех сезонов
iPath_stat_exit8 = iPath_exit + 'Medveditsa_river/cold_data/'                  #Результаты для холодного сезона


######
#Версия для р. Сосна - г.п. Елец (метеостанции 27928, 27915, 34013, 34112)
######
"""
fileName_1 = '27928.csv'
fileName_2 = '27915.csv'
fileName_3 = '34013.csv'
fileName_4 = '34112.csv'

iPath_1 = path_main + 'meteo_2000_2020/{}'.format(fileName_1)
iPath_2 = path_main + 'meteo_2000_2020/{}'.format(fileName_2)
iPath_3 = path_main + 'meteo_2000_2020/{}'.format(fileName_3)
iPath_4 = path_main + 'meteo_2000_2020/{}'.format(fileName_4)

# Загружаем массивы с метеоданными
df_27928 = initial_data(iPath_1)
df_27915 = initial_data(iPath_2)
df_34013 = initial_data(iPath_3)
df_34112 = initial_data(iPath_4)

# Создаем общий массив и усредняем значения метеопараметров
df_data = pd.concat((df_27928, df_27915,df_34013,df_34112)).groupby(level=0).mean()

# Подгружаем данные гидрологических наблюдений
fileName_hydro = 'Rivers_discharges.xlsx'
iPath_hydro = path_main + 'hydro_data/{}'.format(fileName_hydro)

df_hydro = pd.read_excel(iPath_hydro, skiprows = 0, sep=';', dayfirst = True, parse_dates = True, index_col = [0], 
                         skipinitialspace = True, na_values= ['9990','********'])
print ('Columns:', df_hydro.columns)
data_rivers = df_hydro['Sosna']
"""
######
#Версия для р. Битюг - г.п. Бобров (метеостанции 34036, 34238)
######
"""
fileName_1 = '34036.csv'
fileName_2 = '34238.csv'

iPath_1 = path_main + 'meteo_2000_2020/{}'.format(fileName_1)
iPath_2 = path_main + 'meteo_2000_2020/{}'.format(fileName_2)

# Загружаем массивы с метеоданными
df_34036 = initial_data(iPath_1)
df_34238 = initial_data(iPath_2)

# Создаем общий массив и усредняем значения метеопараметров
df_data = pd.concat((df_34036, df_34238)).groupby(level=0).mean()

# Подгружаем данные гидрологических наблюдений
fileName_hydro = 'Rivers_discharges.xlsx'
iPath_hydro = path_main + 'hydro_data/{}'.format(fileName_hydro)

df_hydro = pd.read_excel(iPath_hydro, skiprows = 0, sep=';', dayfirst = True, parse_dates = True, index_col = [0], 
                         skipinitialspace = True, na_values= ['9990','********'])
print ('Columns:', df_hydro.columns)
data_rivers = df_hydro['Bitug']
"""
######
#Версия для р. Тихая Сосна - г.п. Алексеевка (метеостанции 34213, 34321)
######
"""
fileName_1 = '34213.csv'
fileName_2 = '34321.csv'

iPath_1 = path_main + 'meteo_2000_2020/{}'.format(fileName_1)
iPath_2 = path_main + 'meteo_2000_2020/{}'.format(fileName_2)

# Загружаем массивы с метеоданными
df_34213 = initial_data(iPath_1)
df_34321 = initial_data(iPath_2)

# Создаем общий массив и усредняем значения метеопараметров
df_data = pd.concat((df_34213, df_34321)).groupby(level=0).mean()

# Подгружаем данные гидрологических наблюдений
fileName_hydro = 'Rivers_discharges.xlsx'
iPath_hydro = path_main + 'hydro_data/{}'.format(fileName_hydro)

df_hydro = pd.read_excel(iPath_hydro, skiprows = 0, sep=';', dayfirst = True, parse_dates = True, index_col = [0], 
                         skipinitialspace = True, na_values= ['9990','********'])
print ('Columns:', df_hydro.columns)
data_rivers = df_hydro['Tixay Sosna']
"""
######
#Версия для р. Медведица - г.п. Лысые Горы (метеостанции 34063, 34069, 34163)
######

fileName_1 = '34063.csv'
fileName_2 = '34069.csv'
fileName_3 = '34163.csv'

iPath_1 = path_main + 'meteo_2000_2020/{}'.format(fileName_1)
iPath_2 = path_main + 'meteo_2000_2020/{}'.format(fileName_2)
iPath_3 = path_main + 'meteo_2000_2020/{}'.format(fileName_3)

# Загружаем массивы с метеоданными
df_34063 = initial_data(iPath_1)
df_34069 = initial_data(iPath_2)
df_34163 = initial_data(iPath_3)

# Создаем общий массив и усредняем значения метеопараметров
df_data = pd.concat((df_34063, df_34069, df_34163)).groupby(level=0).mean()

# Подгружаем данные гидрологических наблюдений
fileName_hydro = 'Rivers_discharges.xlsx'
iPath_hydro = path_main + 'hydro_data/{}'.format(fileName_hydro)

df_hydro = pd.read_excel(iPath_hydro, skiprows = 0, sep=';', dayfirst = True, parse_dates = True, index_col = [0], 
                         skipinitialspace = True, na_values= ['9990','********'])
print ('Columns:', df_hydro.columns)
data_rivers = df_hydro['Medveditsa']





# Объединяем массив с метеоданными с данными о расходах воды
df_main = pd.concat((df_data, data_rivers), axis = 1)

# Отбрасываем строки с пустыми значениям, где заполненных параметров меньше 15. Для того чтобы отфльтровать лишние расходы воды
df_main = df_main.dropna(axis='rows', thresh=15) 

# Отбрасываем "ненужные" или дублирующие столбцы данных 
df_main = df_main.drop(['tming','pmsl','dd10m','R12','R24'], axis=1)            # Основной массив с метеоданными

# Заполняем оставшиеся пустые значения в столбцах средним значением по столбцу
# Формируем датафрейм с информацией о всех метеоданных за весь год (с января по декабрь)
df_main = df_main.fillna(df_main.mean())

# Нужно правильно указывать столбец из которого были взяты расход воды
#df_main = df_main.rename(columns={'Sosna': 'Discharge'})
#df_main = df_main.rename(columns={'Bitug': 'Discharge'})
#df_main = df_main.rename(columns={'Tixay Sosna': 'Discharge'})
df_main = df_main.rename(columns={'Medveditsa': 'Discharge'})



###### Этап 2. Готовим данные для всех сезонов
# Делаем переиндексацию и отбрасываем дату
count = []  
count_numbers = 0
for jj in range(len(df_main)):
    count_numbers += 1 
    count.append(count_numbers)
t = pd.Series(count, index = df_main.index)    
df_main_year = df_main.set_index(t)                                            # Итоговый массив данных для все 4 сезонов



###### Этап 3. Готовим данные для зимнего сезона
# Создаем специальный массив с данными только за зимний период годы, чтобы посмотреть влияние снега на весеннее половодье
# Создаем пустые списки для переменных
ps_winter = ''
t2m_winter = ''
tmin2m_winter = ''
tmax2m_winter = ''
t_g_winter = ''
td2m_winter = ''
ff10meam_winter = ''
ff10max_winter = ''
hsnow_snowe_winter = ''
rho_winter = ''
swe_winter = ''
R12_liquid_winter = ''
R12_solid_winter = ''
Discharge_winter = ''

# Задаем количество периодов = количеству зимних сезонов
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
    try:
        y_w_1 = periods_winter[tr][0]
        y_w_2 = periods_winter[tr][1]
                
        ps_winter = winter_data(ps_winter, df_main, 'ps', y_w_1, y_w_2)       
        t2m_winter = winter_data(t2m_winter, df_main, 't2m', y_w_1, y_w_2)        
        tmin2m_winter = winter_data(tmin2m_winter, df_main, 'tmin2m', y_w_1, y_w_2)        
        tmax2m_winter = winter_data(tmax2m_winter, df_main, 'tmax2m', y_w_1, y_w_2)        
        t_g_winter = winter_data(t_g_winter, df_main, 't_g', y_w_1, y_w_2)        
        td2m_winter = winter_data(td2m_winter, df_main, 'td2m', y_w_1, y_w_2)
        ff10meam_winter = winter_data(ff10meam_winter, df_main, 'ff10meam', y_w_1, y_w_2)
        ff10max_winter = winter_data(ff10max_winter, df_main, 'ff10max', y_w_1, y_w_2)        
        hsnow_snowe_winter = winter_data(hsnow_snowe_winter, df_main, 'hsnow_snowe', y_w_1, y_w_2)
        rho_winter = winter_data(rho_winter, df_main, 'rho', y_w_1, y_w_2)
        swe_winter = winter_data(swe_winter, df_main, 'swe', y_w_1, y_w_2)
        R12_liquid_winter = winter_data(R12_liquid_winter, df_main, 'R12_liquid', y_w_1, y_w_2)
        R12_solid_winter = winter_data(R12_solid_winter, df_main, 'R12_solid', y_w_1, y_w_2)
        Discharge_winter = winter_data(Discharge_winter, df_main, 'Discharge', y_w_1, y_w_2)                      
    except:
        print ('No data')

df_main_winter = pd.concat([ps_winter, t2m_winter, tmin2m_winter, tmax2m_winter, t_g_winter,
                            td2m_winter, ff10meam_winter, ff10max_winter, hsnow_snowe_winter,
                            rho_winter, swe_winter, R12_liquid_winter, R12_solid_winter,
                            Discharge_winter], axis = 1)

df_main1 = df_main_winter                                                       # Создаем специальный массив для сохранения индекса с датой для факторного анализа
# Делаем переиндексацию и отбрасываем дату
count2 = []  
count_numbers_2 = 0
for jjj in range(len(df_main_winter)):
    count_numbers_2 += 1 
    count2.append(count_numbers_2)

t2 = pd.Series(count2, index = df_main_winter.index)    
df_main_winter = df_main_winter.set_index(t2)                                       # Итоговый массив данных для зимних сезонов


###### Река Сосна - г.п. Елец
"""
###### Этап 4. Машинное обучение и Факторный анализ
print ('4 сезона')
annual_data_m = simple_machine_learning(df_main_year, iPath_stat_exit1)
annual_data_f = factor_analysis(df_main_year, iPath_stat_exit1, df_main)
print ('Холодный сезон')
cold_data_m = simple_machine_learning(df_main_winter, iPath_stat_exit2)
cold_data_f = factor_analysis(df_main_winter, iPath_stat_exit2, df_main1)
"""

###### Река Битюг - г.п. Бобров
"""
###### Этап 4. Машинное обучение и Факторный анализ
print ('4 сезона')
annual_data_m = simple_machine_learning(df_main_year, iPath_stat_exit3)
annual_data_f = factor_analysis(df_main_year, iPath_stat_exit3, df_main)
print ('Холодный сезон')
cold_data_m = simple_machine_learning(df_main_winter, iPath_stat_exit4)
cold_data_f = factor_analysis(df_main_winter, iPath_stat_exit4, df_main1)
"""


###### Река Тихая Сосна - г.п. Алексеевка
"""
###### Этап 4. Машинное обучение и Факторный анализ
print ('4 сезона')
annual_data_m = simple_machine_learning(df_main_year, iPath_stat_exit5)
annual_data_f = factor_analysis(df_main_year, iPath_stat_exit5, df_main)
print ('Холодный сезон')
cold_data_m = simple_machine_learning(df_main_winter, iPath_stat_exit6)
cold_data_f = factor_analysis(df_main_winter, iPath_stat_exit6, df_main1)
"""

###### Река Медведица - г.п. Лысые Горы

###### Этап 4. Машинное обучение и Факторный анализ
print ('4 сезона')
annual_data_m = simple_machine_learning(df_main_year, iPath_stat_exit7)
annual_data_f = factor_analysis(df_main_year, iPath_stat_exit7, df_main)
print ('Холодный сезон')
cold_data_m = simple_machine_learning(df_main_winter, iPath_stat_exit8)
cold_data_f = factor_analysis(df_main_winter, iPath_stat_exit8, df_main1)







