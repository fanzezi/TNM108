import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# World Happiness data
worldHappy = pd.read_csv("/Users/FannyBanny/Documents/TNM108/Minipro/happy-data/test.csv")
# Internet users
internet_user = pd.read_csv("/Users/FannyBanny/Documents/TNM108/Minipro/internet/internet_data.csv")

#Sort the data
internet_user = internet_user.sort_values('Country', axis = 0, ascending = True, inplace = False, na_position = 'last')
worldHappy = worldHappy.sort_values('Country', axis = 0, ascending = True, inplace = False, na_position = 'last')

#print(pd.concat([internet_user, worldHappy], keys=['Db1', 'Db2'], names=['Country or region', 'Row ID']))

internet_user['Internet Users'] = internet_user['Internet Users'].replace({',':''}, regex = True)
internet_user['Population'] = internet_user['Population'].replace({',':''}, regex = True)
internet_user['Percentage'] = internet_user['Percentage'].replace({'%':''}, regex = True)

# Remove unness. data
internet = internet_user.drop(['Internet Users', 'Population', 'Rank.1', 'Rank'], axis=1)
worldHappy = worldHappy.drop(['Overall rank'], axis = 1)

merge_data = pd.merge(internet, worldHappy, on='Country')

merge_data_country = merge_data.drop(['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'], axis = 1)
merge_data_internet = merge_data_country.drop('Country', axis = 1)
merge_data2 = merge_data.drop(['Country', 'Percentage'], axis = 1)

#Ta bort NaN- values
merge_data2.fillna(merge_data2.mean(), inplace=True)

#Training set and test set



##### FIIIIIXXXX
# 1 > 6, 0 < 6
#for Score in merge_data.columns():
merge_data2[merge_data2['Score'] < 2] = 0
merge_data2[(merge_data2['Score'] >= 2) & (merge_data['Score'] <4)] = 1
merge_data2[(merge_data2['Score'] >= 4) & (merge_data['Score'] <6)] = 2
merge_data2[merge_data2['Score'] >= 4] = 2


#merge_data2.loc[merge_data2['Score'] < 6, 'Score'] = 0
#merge_data2.loc[merge_data2['Score'] >= 6, 'Score'] = 1


# Splitting into test and traing data   
x = np.array(merge_data2.drop(['Score'], 1).astype(float)) #training
y = np.array(merge_data2['Score']) #test


#fit the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(x)

#Builing our model
#max-ter: 783 max för 0, 784 min för 1
kmeans = KMeans(algorithm = 'auto', copy_x = True, init = 'k-means++', max_iter = 300, n_clusters = 2, n_init = 10, n_jobs = 1, 
precompute_distances = 'auto', random_state = None, tol = 0.0001, verbose = 0)

#fit the data
kmeans.fit(X_scaled)
prediction_all = []
#Predicting the happiness score
correct = 0
for i in range(len(X_scaled)):
    predict_me = np.array(X_scaled[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    prediction_all.append(prediction[0])
    #print(y[i])
    #print(prediction[0])
    if prediction[0] == y[i]:
        correct += 1

#Presdiction accuracy
print(correct/len(X_scaled))

#Add back country to dataset
prediction = pd.DataFrame(prediction_all)


y2 = pd.DataFrame.join(merge_data_country, prediction)
merge_data_internet = np.array((merge_data_internet).astype(float))

# Fixa label på plot
plt.scatter(merge_data_internet, merge_data['Score'], c = prediction , s = 50, cmap='viridis')
plt.xlabel('Internet percentage users')
plt.ylabel('Happiness score')
#centers = kmeans.cluster_centers_
#plt.scatter(centers[:,0], centers[:,1], c = 'black', s = 200, alpha= 0.5)
    
plt.show()

'''
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(merge_data_internet, merge_data['Score'], c=prediction, s = 50 )
for i,j in range(len(X_scaled)):
    ax.scatter(i,j, s=50, c='red', marker ='+')
ax.set_xlabel('Internet users in percentage')
ax.set_ylabel('Happiness Score')

fig.show()
'''



