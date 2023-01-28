import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew
import seaborn as sb

# 3 
# Učitati bazu u DataFrame. Proveriti kako izgleda prvih nekoliko vrsta u bazi.
df = pd.read_csv('GuangzhouPM20100101_20151231.csv')
print(df.head())

# %% 4

# Koliko ima obeležja?
print(df.shape[1])
# Koliko ima uzoraka?
print(df.shape[0])
# Šta predstavlja jedan uzorak baze?
print(df.sample())
# Kojim obeležjima raspolažemo? 
print(df.head(0))
# Koja obeležja su kategorička, a koja numerička?
print(df.info())
#  Postoje li nedostajući podaci?
print(df.isnull().values.any())
# Gde se javljaju i koliko ih je?
print(df.isnull())
print(df.isnull().sum())

# Postoje li nelogične/nevalidne vrednosti?
def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   return outliers

outliers = find_outliers_IQR(df['HUMI'])
print('invalid values: HUMI: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #-9999 je nelogicna vrednost

outliers = find_outliers_IQR(df['DEWP'])
print('invalid values: DEWP: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #-9999 je nelogicna vrednost

outliers = find_outliers_IQR(df['PRES'])
print('invalid values: PRES: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['Iws'])
print('invalid values: Iws: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['precipitation'])
print('invalid values: precipitation: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nije nevalidna, vec outlier
print(sum(i > 50 for i in df['Iprec']))
print(sum(i > 5 for i in df['Iprec']))

outliers = find_outliers_IQR(df['Iprec'])
print('invalid values: Iprec: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nije nevalidna, vec outlier
print(sum(i > 200 for i in df['Iprec']))
print(sum(i > 10 for i in df['Iprec']))

outliers = find_outliers_IQR(df['year'])
print('invalid values: Iprec: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['season'])
print('invalid values: Iprec: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['month'])
print('invalid values: Iprec: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['day'])
print('invalid values: Iprec: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['hour'])
print('invalid values: Iprec: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['PM_City Station'])
print('invalid values: PM_City Station: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['PM_5th Middle School'])
print('invalid values: PM_5th Middle School: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

outliers = find_outliers_IQR(df['PM_US Post'])
print('invalid values: PM_US Post: ')
print('max outlier value: '+ str(outliers.max()))
print('min outlier value: '+ str(outliers.min())) #nema

# %% 5
# Izbaciti obeležja koja se odnose na sve lokacije merenja koncentracije PM čestica osim US Post.

data = df.drop(columns=['No', 'PM_City Station', 'PM_5th Middle School'])
print(data.columns)
print(data.head()) 

# Dodatno je izbacena kolona: 'No',zato sto predstavlja redni broj merenja i nije od značaja za analizu

# %% 6
# Ukoliko postoje nedostajući podaci, rešiti taj problem na odgovarajući način. 
# Objasniti zašto je rešeno na odabrani način.


print(data.isnull().sum())
data.dropna(subset = ['season'], inplace=True)
print(data.isnull().sum())

# Uzorak u kom je sezona 0 je uklonjen jer ima nedostajuce vrednosti i za druga obelezja

data.dropna(subset = ['PM_US Post'], inplace=True)
print(data.isnull().sum())

# Uklonjene su vrednosti gde je PM2.5 (’PM_US Post’) null, zato sto je to bitno obelezje za dalju analizu

# %% 7. 
# Analizirati obeležja (statističke veličine, raspodela, …)

from fitter import Fitter

print(data.describe())

# %% TEMP outlirers

print(data['TEMP'].describe())

sb.displot(data=data, x="TEMP", kind="hist", bins = 5, aspect = 1.5)
height = data["TEMP"].values

print('koef.asimetrije:  %.2f' % skew(data['TEMP'])) # Raspodela je negativno asimetrična 
print('koef.spljoštenosti:  %.2f' % kurtosis(data['TEMP'])) # Raspodela spljoštena u odnosu na normalnu raspodelu

f = Fitter(height, distributions=["norm"])
f.fit()
f.summary()

plt.xlabel('Temperatura (℃)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper left')

plt.show()

plt.boxplot(data['TEMP'])
plt.xlabel('Temp')
plt.grid()

# %% DEWP

print(data['DEWP'].describe())

data.drop(data[data['DEWP'] == -9999].index, inplace = True)
print(data['DEWP'].describe())


#%% ouliers
sb.displot(data=data, x="DEWP", kind="hist", bins = 10, aspect = 1.5)
height = data["DEWP"].values

print('koef.asimetrije:  %.2f' % skew(data['DEWP'])) # Raspodela je negativno asimetrična 
print('koef.spljoštenosti:  %.2f' % kurtosis(data['DEWP'])) # Raspodela izdužena u odnosu na normalnu raspodelu

f = Fitter(height, distributions=["norm"])
f.fit()
f.summary()

plt.xlabel('Temperatura rose/kondenzacije (°C)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper left')
plt.show()

plt.boxplot(data['DEWP'])
plt.xlabel('DEWP')
plt.grid()

# %% HUMI outliers, iqr

print(data['HUMI'].describe())

data.drop(data[data['HUMI'] == -9999].index, inplace = True)
print(data['HUMI'].describe())

sb.displot(data=data, x="HUMI", kind="hist", bins = 5, aspect = 1.5)
height = data["HUMI"].values

print('koef.asimetrije:  %.2f' % skew(data['HUMI'])) # Raspodela je negativno asimetrična 
print('koef.spljoštenosti:  %.2f' % kurtosis(data['HUMI'])) # Raspodela izdužena u odnosu na normalnu raspodelu

f = Fitter(height, distributions=["norm"])
f.fit()
f.summary()

plt.xlabel('Vlažnost vazduha (%)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper left')
plt.show()

plt.boxplot(data['HUMI'])
plt.xlabel('HUMI')
plt.grid()

# %% PRES outliers

print(data['PRES'].describe())

sb.displot(data=data, x="PRES", kind="hist", bins = 5, aspect = 1.5)
height = data["PRES"].values

print('koef.asimetrije:  %.2f' % skew(data['PRES'])) # Nema asimetrije
print('koef.spljoštenosti:  %.2f' % kurtosis(data['PRES'])) # Raspodela spljoštena u odnosu na normalnu raspodelu

f = Fitter(height, distributions=[
                          "norm"])
f.fit()
f.summary()

plt.xlabel('Vazdušni pritisak (hPa)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper left')
plt.show()

plt.boxplot(data['PRES'])
plt.xlabel('PRES')
plt.grid()

# %% Iws iqr, outliers

print(data['Iws'].describe())

sb.displot(data=data, x="Iws", kind="hist", bins = 5, aspect = 1.5)
height = data["Iws"].values

print('koef.asimetrije:  %.2f' % skew(data['Iws'])) # Pozitivna asimetrija
print('koef.spljoštenosti:  %.2f' % kurtosis(data['Iws'])) # Raspodela izdužena u odnosu na normalnu raspodelu

f = Fitter(height, distributions=["norm"])
f.fit()
f.summary()

plt.xlabel('Kumulativna brzina vetra (m/s)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper left')
plt.show()

plt.boxplot(data['Iws'])
plt.xlabel('Iws')
plt.grid()

# %% precipitation iqr, outliers

print(data['precipitation'].describe())

sb.displot(data=data, x="precipitation", kind="hist", bins = 5, aspect = 1.5)
height = data["precipitation"].values

print('koef.asimetrije:  %.2f' % skew(data['precipitation'])) # Pozitivna asimetrija
print('koef.spljoštenosti:  %.2f' % kurtosis(data['precipitation'])) # Raspodela izdužena u odnosu na normalnu raspodelu

f = Fitter(height, distributions=["norm"])
f.fit()
f.summary()

plt.xlabel('Padavine na sat (mm)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper left')
plt.show()

plt.boxplot(data['precipitation'])
plt.xlabel('precipitation')
plt.grid()

# %% Iprec iqr, outliers

print(data['Iprec'].describe())

sb.displot(data=data, x="Iprec", kind="hist", bins = 5, aspect = 1.5)
height = data["precipitation"].values

print('koef.asimetrije:  %.2f' % skew(data['Iprec'])) # Pozitivna asimetrija
print('koef.spljoštenosti:  %.2f' % kurtosis(data['Iprec'])) # Raspodela izdužena u odnosu na normalnu raspodelu

f = Fitter(height, distributions=["norm"])
f.fit()
f.summary()

plt.xlabel('Kumulativne padavine (mm)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper left')
plt.show()

plt.boxplot(data['Iprec'])
plt.xlabel('Iprec')
plt.grid()


# %% cbwd

sb.countplot(x='cbwd', data=data)
plt.show()
print(data['cbwd'].describe())

# %% 11. Uraditi još nešto po sopstvenom izboru (takođe obavezna stavka).

# boxplots za obelezja sa outleier-ima

plt.figure()
plt.boxplot([data['HUMI'], data['Iws'], data['precipitation']])
plt.xticks([1, 2, 3], ["HUMI", "Iws", "precipitation"])
plt.grid()

# %%boxplots za obelezja gde je bolje koristiti iqr nego dinamcki opseg

plt.figure()
plt.boxplot([data['DEWP'], data['HUMI'], data['Iws'], data['precipitation']])
plt.xticks([1, 2, 3, 4], ["DEWP", "HUMI", "Iws", "precipitation"])
plt.grid()

# %% 8. 
# Analizirati detaljno vrednosti obeležja PM2.5 (’PM_US Post’).

print(data['PM_US Post'].describe())

plt.hist(data['PM_US Post'], bins = 20)
sb.displot(data=data, x="PM_US Post", kind="hist", bins = 100, aspect = 1.5)
height = data["PM_US Post"].values
f = Fitter(height, distributions=["norm"])
f.fit()
f.summary()

print('koef.asimetrije:  %.2f' % skew(data['PM_US Post'])) # Negativna asimetrija
print('koef.spljoštenosti:  %.2f' % kurtosis(data['PM_US Post'])) # Raspodela izdužena u odnosu na normalnu raspodelu

plt.xlabel('PM_US Post')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper right')
plt.show()

plt.boxplot(data['PM_US Post'])
plt.xlabel('PM_US Post')
plt.grid()

# %% 9
# Vizuelizovati i iskomentarisati zavisnost promene PM2.5 od preostalih obeležja u bazi.
# Analizirati međusobne korelacije obeležja.

# %% season

data_season = data[['season', 'PM_US Post']]
plt.scatter(data_season['season'], data_season['PM_US Post'])
plt.xlabel('Sezona')
plt.ylabel('PM_US Post')
plt.show()

data_season_help = data[['season', 'PM_US Post']]
print(data_season_help.corr()) # slaba korelacija

# %% year

data_year = data.set_index('year')
print(data_year.head())

plt.figure()
plt.boxplot([data_year.loc['2011','PM_US Post'], data_year.loc['2012','PM_US Post'], data_year.loc['2013','PM_US Post'], data_year.loc['2014','PM_US Post'], data_year.loc['2015','PM_US Post']]) 
plt.ylabel('PM_US Post')
plt.xlabel('Godina')
plt.xticks([1, 2, 3, 4, 5], ["2011", "2012", "2013", "2014", "2015"])
plt.grid()
plt.show()

data_year_help = data[['year', 'PM_US Post']]
print(data_year_help.corr()) # slaba, negativna korelacija


# %% month

data_month = data[['month', 'PM_US Post']]
plt.scatter(data_month['month'], data_month['PM_US Post'])
plt.ylabel('PM_US Post')
plt.xlabel('Mesec')
plt.show()

corr_mat = data_month.corr()
sb.heatmap(corr_mat, annot=True) #Slaba negativna korelacija
plt.show()

# %% Iws

data_Iws = data[['Iws', 'PM_US Post']]
sns.regplot(data = data_Iws, x=data_Iws['Iws'], y=data_Iws['PM_US Post'], scatter_kws={"color": "white", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"red", 'linewidth':1})
plt.legend()

data_Iws_help = data[['Iws', 'PM_US Post']]
print(data_Iws_help.corr()) #Slaba negativna korelacija

# %% day

data_day = data[['day', 'PM_US Post']]
plt.scatter(data_day['day'], data_day['PM_US Post'])
plt.ylabel('PM_US Post')
plt.xlabel('Dan')
plt.show()

corr_mat = data_day.corr()
sb.heatmap(corr_mat, annot=True) #Slaba pozitivna korelacija
plt.show()

# %% hour

data_hour = data[['hour', 'PM_US Post']]
plt.scatter(data_hour['hour'], data_hour['PM_US Post'])
plt.ylabel('PM_US Post')
plt.xlabel('Sat')
plt.show()

corr_mat = data_hour.corr()
sb.heatmap(corr_mat, annot=True) #Slaba pozitivna korelacija
plt.show()

# %% sesaon

data_season = data[['season', 'PM_US Post']]
plt.scatter(data_season['season'], data_season['PM_US Post'])
plt.ylabel('PM_US Post')
plt.xlabel('Sezona')
plt.show()

corr_mat = data_season.corr()
sb.heatmap(corr_mat, annot=True) #Slaba pozitivna korelacija
plt.show()

# %% temp

data_temp  = data[['TEMP', 'PM_US Post']]
sns.regplot(data = data_temp, x=data_temp['TEMP'], y=data_temp['PM_US Post'], scatter_kws={"color": "white", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"red", 'linewidth':1})
plt.ylim(0, 1200)
plt.show()

print(data_temp.corr()) #Slaba negativna korelacija

# %% dewp

data_dewp  = data[['DEWP', 'PM_US Post']]
sns.regplot(data = data_dewp, x=data_dewp['DEWP'], y=data_dewp['PM_US Post'], scatter_kws={"color": "white", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"red", 'linewidth':1})
plt.ylim(0, 1200)
plt.show()

print(data_dewp.corr())#Slaba negativna korelacija

# %% HUMI

plt.style.use('dark_background')
data_HUMI  = data[['HUMI', 'PM_US Post']]
sns.regplot(data = data_HUMI, x=data_HUMI['HUMI'], y=data_HUMI['PM_US Post'], scatter_kws={"color": "white", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"red", 'linewidth':1})
plt.ylim(0, 1200)
plt.show()

print(data_HUMI.corr()) #Slaba negativna korelacija


# %% pres

plt.style.use('dark_background')
data_pres  = data[['PRES', 'PM_US Post']]
sns.regplot(data = data_pres, x=data_pres['PRES'], y=data_pres['PM_US Post'], scatter_kws={"color": "white", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"red", 'linewidth':1})
plt.ylim(0, 1200)
plt.show()

print(data_pres.corr()) #Slaba pozitivna korelacija

# %% precipitation, Iprec

data_precipitation  = data[['Iprec', 'precipitation', 'PM_US Post']]
sns.regplot(data = data_precipitation, x=data_precipitation['precipitation'], y=data_precipitation['PM_US Post'], scatter_kws={"color": "white", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"red", 'linewidth':1})
sns.regplot(data = data_precipitation, x=data_precipitation['Iprec'], y=data_precipitation['PM_US Post'], scatter_kws={"color": "blue", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"yellow", 'linewidth':1})
plt.ylim(0, 350)
plt.show()

print(data_precipitation.corr()) # Slaba negativna korelacija za oba

# %%
# Analizirati međusobne korelacije obeležja.
corr_mat1 = data.corr()
fig, ax = plt.subplots(figsize=(12,8))
sb.heatmap(corr_mat1, linewidth=1, annot=True, ax=ax)
plt.show()

c = data.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")
print(so)

# %% sa najvecim korelacijama dewp, pres i pres, temp

data_dewp_pres  = data[['PRES', 'DEWP']]
sns.regplot(data = data_dewp_pres, x=data_dewp_pres['PRES'], y=data_dewp_pres['DEWP'], scatter_kws={"color": "white", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"red", 'linewidth':1})
plt.show()

print(data_dewp_pres.corr())

data_temp_pres  = data[['PRES', 'TEMP']]
sns.regplot(data = data_temp_pres, x=data_temp_pres['PRES'], y=data_temp_pres['TEMP'], scatter_kws={"color": "white", 'edgecolor': 'grey', 'alpha': 0.4}, line_kws={"color":"red", 'linewidth':1})
plt.show()

print(data_temp_pres.corr())

# %%
print(corr_mat1['PM_US Post'])

# %% cbwd - kvalitativno obelezje - transformacija

df_dummy = pd.get_dummies(data['cbwd'], prefix='cbwd')
data = pd.concat([data, df_dummy], axis=1)
data.drop(['cbwd'], axis=1, inplace=True)
print(data.head())

# %% cbwd i pm2.5 - korelacija

data_cbwd = data[['cbwd_NW', 'cbwd_cv', 'cbwd_SE', 'cbwd_NE', 'cbwd_SW', 'PM_US Post']]
corr_mat = data_cbwd.corr()
print(corr_mat['PM_US Post']) # slava pozitivina ili negaticne korelacija
plt.show()

# %% 11. Uraditi još nešto po sopstvenom izboru (takođe obavezna stavka).
# promena PM_US Post: po godini(najranijoj i najkasnijoj) histogramom

gb = data.groupby(by=['season', 'year']).mean()
print(gb)

# histogram
plt.hist(data_year.loc[2012, 'PM_US Post'], bins=100, density=True, alpha=0.3, label='2012')
plt.hist(data_year.loc[2015, 'PM_US Post'], bins=100, density=True, alpha=0.3, label='2015')

plt.xlabel('PM_US Post')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper left')

# u poslednjej godini, raspodela je izduzenija nego u prvoj
# obe raspoedele su pozitivno simetricne, u prvoj godini je taj koeficijent veci


# %% II DEO: LINEARNA REGRESIJA

# %% 1) Potrebno je 15% nasumično izabranih uzoraka ostaviti kao test skup, 15% kao validacioni a
# preostalih 70% koristiti za obuku modela.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

x = data.drop(['PM_US Post'], axis=1).copy()
y = data['PM_US Post'].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=data['month'])
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)       

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

x_test.drop(47939, axis='index', inplace=True)
y_test.drop(47939, axis='index', inplace=True)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# %% 2) Isprobati različite hipoteze, regularizaciju modela, selekciju obeležja (unapred ili unazad)

# Funkciju koja računa različite mere uspešnosti regresora, a koja će biti korišćena nakon svake 
# obuke i testiranja.

def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted) # np.mean((y_test-y_predicted)**2)
    mae = mean_absolute_error(y_test, y_predicted) # np.mean(np.abs(y_test-y_predicted))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))


# %% Treniranje sa train setom, testiranje na validacionom skupu

# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(x_train, y_train)

# Testiranje
y_predicted = first_regression_model.predict(x_val)

# Evaluacija
model_evaluation(y_val, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)

# Pravac vetra i godine, imaju uticaj
# Kondezacija i vlaznost vazduha nemaju uticaj
# Koeficijent determinacije je negativan - model ne prati trend podataka

# %%
# Treniranje sa train i val setom, testiranje na testu

# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(pd.concat([x_train, x_val], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True))

# Testiranje
y_predicted = first_regression_model.predict(x_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)

# Znacaj pravca vetra je manji i dalje, pored godina, ima najveci uticaj
# msr i mar su manje nego na testiranju na val skupu
# koeficijent determinacije je bio negativan, sada je pozitivan - mala korelacija (0,13)

# %% SELEKCIJA OBELEZJA UNAZAD

# %% selekcija

import statsmodels.api as sm
X = sm.add_constant(x_train)

model = sm.OLS(y_train, X.astype('float')).fit()
print(model.summary())

# %% izbacivanje obelezja gde je P>|t| najvece, prag: 0,02

X = sm.add_constant(x_train.drop('HUMI', axis=1))

model = sm.OLS(y_train, X.astype('float')).fit()
print(model.summary())

# %% izbacivanje obelezja gde je P>|t| najvece, prag: 0,02

X = sm.add_constant(x_train.drop(['HUMI', 'DEWP'], axis=1))

model = sm.OLS(y_train, X.astype('float')).fit()
print(model.summary())

# %% izbacivanje obelezja gde je P>|t| najvece, prag: 0,02

X = sm.add_constant(x_train.drop(['HUMI', 'TEMP', 'DEWP'], axis=1))

model = sm.OLS(y_train, X.astype('float')).fit()
print(model.summary())

# %% Standardizacija obelezja (svodjenje na sr.vr. 0 i varijansu 1)
# Testiranje na validacionom skupu

scaler = StandardScaler()
scaler.fit(x_train)

x_train_std = scaler.transform(x_train)
x_val_std = scaler.transform(x_val)

x_train_std = pd.DataFrame(x_train_std)
x_val_std = pd.DataFrame(x_val_std)

x_train_std_columns = list(x.columns)
x_val_std_columns = list(x.columns)

print(x_train_std.head())

# %% ponavljanje obuke na validacionom skupu

# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
regression_model_std = LinearRegression()

# Obuka modela
regression_model_std.fit(x_train_std, y_train)

# Testiranje
y_predicted = regression_model_std.predict(x_val_std)

# Evaluacija
model_evaluation(y_val, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_std.coef_)),regression_model_std.coef_)
plt.show()
print("koeficijenti: ", regression_model_std.coef_)

# nije dovelo do poboljosanja, samo do eventualne promene brzine konvergencije

# %% Standardizacija obelezja (svodjenje na sr.vr. 0 i varijansu 1)
# Testiranje na test skupu

scaler = StandardScaler()
scaler.fit(pd.concat([x_train, x_val], ignore_index=True))

x_train_std_1 = scaler.transform(pd.concat([x_train, x_val]))
x_test_std = scaler.transform(x_test)

x_train_std_1 = pd.DataFrame(x_train_std_1)
x_test_std = pd.DataFrame(x_test_std)

x_train_std_columns = list(x.columns)
x_test_std_columns = list(x.columns)

print(x_train_std.head())

# %% ponavljanje obuke na validacionom skupu

# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
regression_model_std = LinearRegression()

# Obuka modela
regression_model_std.fit(x_train_std_1, pd.concat([y_train, y_val]))

# Testiranje
y_predicted = regression_model_std.predict(x_test_std)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_std.coef_)),regression_model_std.coef_)
plt.show()
print("koeficijenti: ", regression_model_std.coef_)

# nije dovelo do poboljosanja, samo do eventualne promene brzine konvergencije

# %%  HIPOTEZE

corr_mat = x_train.corr()

plt.figure(figsize=(12, 8), linewidth=2)
sb.heatmap(corr_mat, annot=True)
plt.show()


# %% PolynomialFeatures - validacioni skup

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_val = poly.transform(x_val_std)

print(poly.get_feature_names())

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...

# Inicijalizacija
regression_model_inter = LinearRegression()

# Obuka modela
regression_model_inter.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_inter.predict(x_inter_val)

# Evaluacija
model_evaluation(y_val, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter.coef_)

# mse je veca, mae je ista
# r2 je negativniji

# %% PolynomialFeatures - test skup

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
x_inter_train_1 = poly.fit_transform(x_train_std_1)
x_inter_test = poly.transform(x_test_std)

print(poly.get_feature_names())

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...

# Inicijalizacija
regression_model_inter = LinearRegression()

# Obuka modela
regression_model_inter.fit(x_inter_train_1, pd.concat([y_train, y_val]))

# Testiranje
y_predicted = regression_model_inter.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train_1.shape[0], x_inter_train_1.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter.coef_)

# smanjile su se obe greske
# r2 score je veci - korelacija je izrazenija

# %% Linearna regresija sa drugacijom hipotezom; model sa interakcijama i kvadratima, test skup, validacioni nece biti unapredjen

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_inter_train_1 = poly.fit_transform(x_train_std_1)
x_inter_test = poly.transform(x_test_std)

print(poly.get_feature_names())

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...

# Inicijalizacija
regression_model_degree = LinearRegression()

# Obuka modela
regression_model_degree.fit(x_inter_train_1, pd.concat([y_train, y_val]))

# Testiranje
y_predicted = regression_model_degree.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train_1.shape[0], x_inter_train_1.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_degree.coef_)),regression_model_degree.coef_)
plt.show()
print("koeficijenti: ", regression_model_degree.coef_)

# smanjile su se obe greske
# r2 score je veci
# degree=3, ne dovodi do pobolsanja


# %% REGULARIZACIJA

#sprecavanje velikih vrednosti koeficijenata, zbog smanjivanja mogucnosti nadprilagodjavanja modela

# %% RIDGE

# Validacija

# Inicijalizacija
ridge_model = Ridge(alpha=5)

# Obuka modela
ridge_model.fit(x_inter_train, y_train)

# Testiranje
y_predicted = ridge_model.predict(x_inter_val)

# Evaluacija
model_evaluation(y_val, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)

# %%
# Test

# Inicijalizacija
ridge_model = Ridge(alpha=5)

# Obuka modela
ridge_model.fit(x_inter_train_1, pd.concat([y_train, y_val]))

# Testiranje
y_predicted = ridge_model.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train_1.shape[0], x_inter_train_1.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)

# razlika kod gresaka i r2 score-a nije velika, ali su koeficijenti ograniceni

# %% LASSO

# %% val

# Model initialization
lasso_model = Lasso(alpha=0.01)

# Fit the data(train the model)
lasso_model.fit(x_inter_train, y_train)

# Predict
y_predicted = lasso_model.predict(x_inter_val)

# Evaluation
model_evaluation(y_val, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


#ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.show()
print("koeficijenti: ", lasso_model.coef_)

# %% test

# Model initialization
lasso_model = Lasso(alpha=0.01)

# Fit the data(train the model)
lasso_model.fit(x_inter_train_1, pd.concat([y_train, y_val]))

# Predict
y_predicted = lasso_model.predict(x_inter_test)

# Evaluation
model_evaluation(y_test, y_predicted, x_inter_train_1.shape[0], x_inter_train_1.shape[1])


#ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.show()
print("koeficijenti: ", lasso_model.coef_)

# vrednosti koje su bile jako male, svedene su na 0 i time je izvrsena selekcija
# vrednosti mse, mar i r2 score su skoro identicne kao od ridge modela

# %% 3) Odabrati najbolji model linearne regresije i objasniti zašto je baš taj model odabran.

plt.figure(figsize=(10,5))
plt.plot(regression_model_degree.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'linear',zorder=7) # zorder for ordering the markers
plt.plot(ridge_model.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge') # alpha here is for transparency
plt.plot(lasso_model.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Lasso')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc='best')
plt.show()

# Izabran model: lasso, koeficijenti su ograniceni - spreceno je nadprilagodjavanje modela podacima,
# obelezjima s malim koeficijentima su dodeljene nule za nove vrednosti istih i time je izvrsena selekcija,
# mse i mae su najmanje i r2 score najveci.

# %% III DEO: KNN KLASIFIKATOR

# %% 1. Prvo je potrebno uzorcima iz date baze dodeliti labele: bezbedno, nebezbedno ili opasno.
# Uzorcima čija je vrednost koncentracije PM2.5 čestica do 55.4 µg/m3 dodeliti labelu bezbedno,
# onima čija je vrednost koncentracije PM2.5 čestica od 55.5 µg/m3 do 150.4 µg/m3 dodeliti labelu
# nebezbedno, dok onima sa vrednošću preko 150.5 µg/m3 dodeliti labelu opasno

def assignNewLabels(value):
    if value <= 55.4:
        return 'bezbedno'
    elif 55.5 <= value <= 150.4:
        return 'nebezbedno'
    elif value >= 150.5:
        return 'opasno'
    else:
        return -1

data['Labels'] = data['PM_US Post'].apply(assignNewLabels)

# %%

def evaluation_classifier(conf_mat):
    
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    precision = TP/(FP+TP)
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F_score = 2*precision*sensitivity/(precision*sensitivity)
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)

# %% 2. Koristiti 15% uzoraka za testiranje finalnog klasifikatora, a preostalih 85% uzoraka koristiti: 
# za metodu unakrsne validacije sa 10 podskupova. Ovom metodom odrediti optimalne parametre
# klasifikatora, oslanjajući se na željenu meru uspešnosti. Obratiti pažnju da u svakom od
# podskupova za unakrsnu validaciju, kao i u test skupu, bude dovoljan broj uzoraka svake klase.

X = data.drop(['Labels'], axis=1).copy()
y = data['Labels'].copy()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=data['Labels'])

# %% inicijalicaija i obuka klasifikatora

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# interval od 1 do 14 - n_neighbors
# StratifiedKFold - odrzanje klasnog odnosa
# metrike: 'minkowski', 'chebyshev', “euclidean”, “manhattan”, "hamming", "jaccard", "dice"

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

parameters = {'metric':['minkowski', 'chebyshev', 'euclidean', 'manhattan', "hamming"], 'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}


knn = KNeighborsClassifier()
clf = GridSearchCV(estimator=knn, param_grid=parameters, scoring='accuracy', cv=kfold, refit=True, verbose=3)
clf.fit(x_train, y_train)

print(clf.best_score_)
print(clf.best_params_)

# %%
# 3. Za konačno odabrane parametre prikazati i analizirati matricu konfuzije dobijenu akumulacijom
# matrica iz svake od 10 iteracija unakrsne validacije. Odrediti prosečnu tačnost klasifikatora, kao i
# tačnost za svaku klasu
# parameters = {'metric':['manhattan'], 'n_neighbors':[14]}

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

indexes = list(kfold.split(x_train, y_train))
fin_conf_mat = np.zeros((len(np.unique(y_train)),len(np.unique(y_train))))

accuracy = []
    
for train_index, test_index in indexes:

   Xfold_train = x_train.iloc[train_index,:]
   yfold_train = y_train.iloc[train_index]

   Xfold_test = x_train.iloc[test_index,:]
   yfold_test = y_train.iloc[test_index]

   knn = KNeighborsClassifier(n_neighbors=14, metric='manhattan')
   knn.fit(Xfold_train, yfold_train)

   yfold_pred = knn.predict(Xfold_test) 
   accuracy.append(accuracy_score(yfold_test, yfold_pred))    
      
   conf_mat_temp = confusion_matrix(y_train.iloc[test_index], yfold_pred)
   print(conf_mat_temp)
   print(accuracy_score(yfold_test, yfold_pred))
   fin_conf_mat += conf_mat_temp # usrednjavanje

print("Prosecna tacnost: ")
evaluation_classifier(conf_mat_temp)  # prosecna tacnost
print("Akumuilarana matrica konfuzije: ")
print(fin_conf_mat)
print("Tacnost za svaku klasu:")
print(fin_conf_mat.diagonal()/fin_conf_mat.sum(axis=1))

# 17652 bezbednih predvidjenih kao bezbednih
# 8868 nebezbednih predvidjenih kao nebezbednih
# 587 opasnih predvidjenih kao opasnih

# 80 bezbednih predvidjenih kao nebezbednih
# 0 bezbednih predvidjenih kao opasnih

# 279 nebezbednih predvidjenih kao bezbednih
# 8 nebezbednih predvidjenih kao opasnih

# 0 opasnih predvidjenih kao bezbednih
# 25 opasnih predvidjenih kao nebezbednih

# %%
# 4. Klasifikator sa konačno odabranim parametrima obučiti na celokupnom trening skupu, pa testirati
# na izdvojenom test skupu. Na osnovu dobijene matrice konfuzije izračunati mere uspešnosti
# klasifikatora, kao i mere uspešnosti za svaku klasu (tačnost, osetljivost, specifičnost, 
# preciznost, Fmera).

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classifier = KNeighborsClassifier(n_neighbors=14, metric='manhattan')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test) 
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,  display_labels=clf.classes_)
disp.plot(cmap="Blues")
plt.show()

print(evaluation_classifier(conf_mat))
print(classification_report(y_test, y_pred))