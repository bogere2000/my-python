import pandas as pd
import numpy as np
from matplotlib import pyplot
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
dataset = pd.read_csv('diabetes.csv')
print(dataset)
dataframe = pd.read_csv('diabetes.csv')
print(dataframe)
print(dataframe.shape)
print(dataframe.describe())
print(dataframe.head(10))
print(dataframe.tail(7))
print(dataframe.dtypes)
class_counts = dataframe.groupby('Outcome').size()
print(class_counts)
from pandas import set_option
set_option('display.width', 100)
set_option('precision', 3)
correlation = dataframe.corr(method='pearson')
print(correlation)
skew = dataframe.skew()
print(skew)
dataframe.hist()
pyplot.show()
dataframe.plot(kind='density', subplots=True, layout=(5,5), sharex =False)
pyplot.show()
pyplot.figure(figsize=(30,30))
dataframe.plot(kind='box', subplots= True, layout=(5,5), sharex=False)
pyplot.show()
#correlation matrix plot
correlation = dataframe.corr()
plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 9, 1)
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()
#plotting scatterplot matrix
correlation = dataframe.corr()
plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()
array = dataframe.values
x = array[:,0:8]
y = array[:, 8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x,y)
set_printoptions(precision=3)
print(fit.scores_)
feature = fit.transform(x)
print(feature[0:5,:])
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#feauture extraction
model = LogisticRegression()
rfe = RFE(model,3)
fit = rfe.fit(x,y)
print(fit)
print("NumFeature: %d " % fit.n_features_)
print("SelectedFeature: %s" % fit.support_)
print("FeatureRanking: %s" % fit.ranking_ )
### principal component Analysis
# feature extraction with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
fit = pca.fit(x)
print("explained variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
### feature importance with extra tree classifier
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
##Evaluate using a train and a test set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
test_size = 0.33
seed = 7
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print("Accuracy  %.3f%%"% (result * 100.0))
# Rescale dataset
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
set_printoptions(precision=4)
print(rescaledX[0:6,:])
#standardize data(0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
set_printoptions(precision=3)
print(rescaledX[0:6,:])
# Normalize data(length of 1)
from sklearn.preprocessing import Normalizer
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
set_printoptions(precision=3)
print(normalizedX[0:5,:])
# binarization
from sklearn.preprocessing import Binarizer
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX =  binarizer.transform(X)
set_printoptions(precision=3)
print(binaryX[0:6,:])

# Feature Extraction with Univariate Statistical Tests(Chi-squared for classification)
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,Y)
set_printoptions(precision=3)
features = fit.transform(X)
print(features[0:6,:])
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model,3)
fit = rfe.fit(X,Y)
print('Num Features: %d' % fit.n_features_)
print('Selected Features: %s' % fit.support_)
print('Features Ranking: %s' % fit.ranking_)
# Feature Extraction with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
fit = pca.fit(X)
print('Explained Variance: %s' % fit.explained_variance_ratio_)
print(fit.components_)
# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
model= ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
# Evaluate using a train and a test set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print('Accuracy:%.3f%%' % (result* 100.0))







