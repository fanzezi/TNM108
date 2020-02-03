import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# World Happiness data
worldHappy = pd.read_csv("/Users/FannyBanny/Documents/TNM108/Minipro/happy-data/test.csv")
worldHappyTest = pd.read_csv("/Users/FannyBanny/Documents/TNM108/Minipro/happy-data/2019.csv")
# Internet users data
internet_user = pd.read_csv("/Users/FannyBanny/Documents/TNM108/Minipro/internet/internet_data.csv")

#Sort the data
internet_user = internet_user.sort_values('Country', axis = 0, ascending = True, inplace = False, na_position = 'last')
worldHappy = worldHappy.sort_values('Country', axis = 0, ascending = True, inplace = False, na_position = 'last')
worldHappyTest = worldHappyTest.sort_values('Country', axis = 0, ascending = True, inplace = False, na_position = 'last')

#Remove unwanted characters
internet_user['Internet Users'] = internet_user['Internet Users'].replace({',':''}, regex = True)
internet_user['Population'] = internet_user['Population'].replace({',':''}, regex = True)
internet_user['Percentage'] = internet_user['Percentage'].replace({'%':''}, regex = True)

# Remove unness. data
internet = internet_user.drop(['Internet Users', 'Population', 'Rank.1', 'Rank'], axis=1)
worldHappy = worldHappy.drop(['Overall rank'], axis = 1)

merge_data = pd.merge(internet, worldHappy, on='Country')
merge_data_test = pd.merge(internet, worldHappyTest, on ='Country')

merge_data_country = merge_data.drop(['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'], axis = 1)
merge_data_internet = merge_data_country.drop('Country', axis = 1)

merge_data2 = merge_data.drop(['Country', 'Percentage'], axis = 1)
merge_data_test2 = merge_data_test.drop(['Country', 'Percentage'], axis = 1)

#Remove NaN- values
merge_data2.fillna(merge_data2.mean(), inplace=True)


#Set intervals
#Training set 
merge_data2.loc[merge_data2['Score'] < 6, 'Score'] = 0
merge_data2.loc[merge_data2['Score'] >= 6, 'Score'] = 1
#Test set
merge_data_test2.loc[merge_data_test2['Score'] < 6, 'Score'] = 0
merge_data_test2.loc[merge_data_test2['Score'] >= 6, 'Score'] = 1


# Splitting into test and traning data   
x = np.array(merge_data2.drop(['Score'], 1).astype(float)) #training data
y = np.array(merge_data_test2['Score']) #test data

#fit the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(x)

#Builing our model
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
    if prediction[0] == y[i]:
        correct += 1

#Prediction accuracy
print(correct/len(X_scaled))

#Add back country to dataset
prediction = pd.DataFrame(prediction_all)
y2 = pd.DataFrame.join(merge_data_country, prediction)
merge_data_internet = np.array((merge_data_internet).astype(float))
country_labels = merge_data['Country']

#Plot our results
x = merge_data_internet
y = merge_data['Score']
plt.xlabel('Internet percentage users')
plt.ylabel('Happiness score')
plt.scatter(x, y, c=prediction)

for i,type in enumerate(country_labels):
    x_coords = x[i]
    y_coords = y[i]
    plt.text(x_coords,y_coords,type, fontsize =5)

plt.show()



