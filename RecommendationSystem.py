import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import missingno as msno
import squarify

%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size':12})


## Reading Data Set
path = 'D:\Projects\Wine Recommendation System\Wine Data'
winedata_1 = pd.read_csv(path + '\winemag-data 1.csv', index_col=0)
winedata_2 = pd.read_csv(path + '\winemag-data 2.csv', index_col=0)
winedata = pd.concat([winedata_1, winedata_2], axis=0)
print("Number of rows and columns", winedata.shape)
winedata.head()

## Analysing the data
winedata.describe(include='all',).T
#missing values
msno.bar(winedata, color=sns.color_palette('viridis'))
#Distribution of wine reviews by Top 20 Countries
print('Number of Country list in the data:', winedata['country'].nunique())
plt.figure(figsize=(14,10))
cnt = winedata['country'].value_counts().to_frame()[0:20]
#plt.xscale('log')
sns.barplot(x=cnt['country'], y=cnt.index, data=cnt, palette='ocean', orient='h')
plt.title("Distribution of Wine Reviews by top 20 Countries");
#Distribution in Wine Prices
f, ax = plt.subplots(1,2,figsize=(14,6))
ax1,ax2 = ax.flatten()
sns.distplot(winedata['price'].fillna(winedata['price'].mean()),color="r",ax=ax1)
ax1.set_title("Distribution In Wine Prices")
sns.boxplot(x=winedata['price'], ax=ax2)
ax2.set_ylabel('')
ax2.set_title('Boxplot of Price') ##Dist. in Wine Prices.PNG##
#Country wise wine prices
cnt = winedata.groupby(['country',]).mean()['price'].sort_values(ascending=False).to_frame()
plt.figure(figsize=(16,8))
sns.pointplot(x=cnt['price'], y = cnt.index, color='r', orient='h', markers='o')
plt.title('Country wise Avg. Wine Prices')
plt.xlabel('Price')
plt.ylabel('Country'); ##Country wise Avg. Wine Prices.PNG##

cnt = winedata.groupby(['country',])['price'].mean().sort_values(ascending=False).to_frame()
plt.figure(figsize=(12,8))
squarify.plot(cnt['price'].fillna(0.1),color=sns.color_palette('rainbow'),label=cnt.index) ##Country wise Avg. Wine Prices Squarify Plot.PNG##
#Country wise MostExpensive & LeastExpensive Wine
fig,ax = plt.subplots(1,2,figsize=(16,8))
ax1,ax2 = ax.flatten()
cnt = winedata.groupby(['country'])['price'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x = cnt['price'], y = cnt.index, palette= 'inferno',ax=ax1)
ax1.set_title('MostExpensive Wine by Country')
ax1.set_ylabel('Variety')
ax1.set_xlabel('')
cnt = winedata.groupby(['country'])['price'].min().sort_values(ascending=True).to_frame()[:15]
sns.barplot(x = cnt['price'], y = cnt.index, palette = 'rainbow_r',ax=ax2)
ax2.set_title('LeastExpensive Wine by Country')
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.subplots_adjust(wspace=0.3);  ##Country wise MostExpensive & LeastExpensive Wine.PNG##

plt.figure(figsize=(16,6))
sns.boxplot(x = winedata['country'], y = winedata['price'])
plt.yscale("log")
plt.title('Country wise boxplot of price on log scale')
plt.xticks(rotation=90);  ##Country wise boxplot of price on log scale.PNG##
#Country wise Avg. rating of Wines
cnt = winedata.groupby(['country',]).mean()['points'].sort_values(ascending=False).to_frame()
plt.figure(figsize=(16,8))
sns.pointplot(x = cnt['points'] ,y = cnt.index ,color='r',orient='h')
plt.title('Country wise Avg. Wine Rating')
plt.xlabel('Points');  ##Country wise Avg. rating of Wines.PNG##
#Country wise Highest & Lowest Rated Wine
fig,ax = plt.subplots(1,2,figsize=(16,8))
ax1,ax2 = ax.flatten()
cnt = winedata.groupby(['country'])['points'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x = cnt['points'], y = cnt.index, palette= 'hot',ax=ax1)
ax1.set_title('Country wise Highest Rated Wine')
ax1.set_ylabel('Variety')
ax1.set_xlabel('')
cnt = winedata.groupby(['country'])['points'].min().sort_values(ascending=True).to_frame()[:15]
sns.barplot(x = cnt['points'], y = cnt.index, palette = 'ocean',ax=ax2)
ax2.set_title('Country wise Lowest Rated Wine')
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.subplots_adjust(wspace=0.3);  ##Country wise Highest Rated & Lowest Rated Wine.PNG##

plt.figure(figsize=(16,6))
sns.boxplot(x = winedata['country'], y = winedata['points'])
#sns.pointplot(x = winedata['country'], y = winedata['points'])
plt.title('Country wise boxplot of Rating')
plt.xticks(rotation=90);
#Trying to determine the relation between Price & Rating
sns.jointplot(x=winedata['points'], y=winedata['price'],color='g');
#Top Wine in Each Variety
print('No. of Varity of Wines',winedata['variety'].nunique())
fig,ax = plt.subplots(1,2,figsize=(16,8))
ax1,ax2 = ax.flatten()
cnt = winedata.groupby(['variety'])['price'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x = cnt['price'], y = cnt.index, palette= 'cool',ax=ax1)
ax1.set_title('The Grapes used for Most Expensive Wine')
ax1.set_ylabel('Variety')
ax1.set_xlabel('')
cnt = winedata.groupby(['variety'])['points'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x = cnt['points'], y = cnt.index, palette = 'Wistia',ax=ax2)
ax2.set_title('The Grapes used for Highest Rated Wine')
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.subplots_adjust(wspace=0.3);

fig,ax = plt.subplots(1,2,figsize=(16,8))
ax1,ax2 = ax.flatten()
cnt = winedata.groupby(['variety'])['price'].min().sort_values(ascending=True).to_frame()[:15]
sns.barplot(x = cnt['price'], y = cnt.index,palette = 'ocean_r',ax=ax1)
ax1.set_title('The Grapes used for Lowest Priced Wine')
ax1.set_xlabel('')
ax1.set_ylabel('Variety')
cnt = winedata.groupby(['variety'])['points'].min().sort_values(ascending=True).to_frame()[:15]
sns.barplot(x = cnt['points'], y = cnt.index,palette= 'rainbow', ax=ax2)
ax2.set_title('The Grapes used for Lowest Rated Wine')
ax2.set_xlabel('')
ax2.set_ylabel('')
plt.subplots_adjust(wspace=0.4)

cnt=winedata.groupby(['country', 'points'])['price'].agg(['count','min','max','mean']).sort_values(by='mean', ascending=False)[:10]
cnt.reset_index(inplace=True)
cnt.style.background_gradient(cmap='PuBu', high=0.5)


#Designation##
print('Number of vineyard designation', winedata['designation'].unique())
cnt = winedata.groupby(['designation'])['price'].mean().to_frame().sort_values(by='price',ascending=False)[:15]
f,ax = plt.subplots(1,2,figsize= (14,6))
ax1,ax2 = ax.flatten()
sns.barplot(cnt['price'], y = cnt.index, palette = 'Paired', ax = ax1)
ax1.set_xlabel('')
ax1.set_ylabel('Designation(Vineyard)')
ax1.set_title('Most Expensive Wine Preparering Vineyard')

cnt = winedata.groupby(['designation'])['points'].mean().to_frame().sort_values(by = 'points', ascending = False)[:15]
sns.barplot(cnt['points'], y = cnt.index, palette = 'Set3', ax = ax2)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('Highest Rated Wine Preparing Vineyard')
plt.subplots_adjust(wspace=0.3)

#Winery
print('Number of Wineries:',winedata['winery'].nunique())
f,ax = plt.subplots(1,2,figsize=(16,6))
ax1,ax2 = ax.flatten()
cnt = winedata.groupby(['winery'])['price'].max().to_frame().sort_values(by='price',ascending=False)[:15]
sns.barplot(cnt['price'],y = cnt.index,palette = 'ocean',ax = ax1)
ax1.set_title('The Most Expensive Wine Preparing Winery')
cnt = winedata.groupby(['winery'])['points'].max().to_frame().sort_values(by = 'points', ascending = False)[:15]
sns.barplot(cnt['points'], y = cnt.index, palette = 'hot')
plt.title('Highest Rated Wine Preparing Winery');

#Province
print('Number of Province:',winedata['province'].nunique())
cnt = winedata.groupby(['province'])['price'].mean().sort_values(ascending=False).to_frame()[:30]
plt.figure(figsize=(16,8))
squarify.plot(cnt['price'].fillna(0.001),label=cnt.index,color= sns.color_palette('Set3'))
plt.title('The Avg. Price of Wine by Province');

fig,ax = plt.subplots(1,2,figsize=(16,8))
ax1,ax2 = ax.flatten()
cnt = winedata.groupby(['province'])['price'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x = cnt['price'], y = cnt.index, palette= 'RdBu',ax=ax1)
ax1.set_title('Most Expensive Wine Availabe in the Province')
ax1.set_ylabel('Variety')
ax1.set_xlabel('')
cnt = winedata.groupby(['province'])['price'].min().sort_values(ascending=True).to_frame()[:15]
sns.barplot(x = cnt['price'], y = cnt.index, palette = 'summer',ax=ax2)
ax2.set_title('Lowest Priced Wine Available in the Provice')
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.subplots_adjust(wspace=0.3);

cnt = winedata.groupby(['province','country','points'])['price'].agg(['count','min','max','mean']).sort_values(by='mean',ascending= False)[:10]
cnt.reset_index(inplace=True)
cnt.style.background_gradient(cmap='Blues',high=0.5)

#Region_1
print('Number of Regions:',winedata['region_1'].nunique())
cnt = winedata.groupby(['region_1'])['price'].mean().sort_values(ascending=False).to_frame()[:30]
plt.figure(figsize=(16,8))
squarify.plot(cnt['price'].fillna(0.001),label=cnt.index,color= sns.color_palette('Oranges'))
plt.title('The Avg. Price of Wine by Region 1');

cnt = winedata.groupby(['country','province','points','region_1',])['price'].agg(
    ['count','min','max','mean']).sort_values(by = 'mean',ascending = False)[:20]
cnt.reset_index(inplace = True)
cnt.style.highlight_max()

#Region_2
print('Number of reqion2: ',winedata['region_2'].nunique())
print('Null values in reqion2: ',winedata['region_2'].isnull().sum())
cnt = winedata.groupby(['country','province','region_1','region_2','points'])['price'].agg(['count','min','max','mean']).sort_values(by = 'mean',ascending = False)[:20]
cnt.reset_index(inplace=True)
cnt.style.set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'yellow')]}])

cnt = winedata.groupby(['country','region_2']).count().reset_index()
cnt['country'].unique()

##Taster
print(winedata[['taster_name', 'taster_twitter_handle']].describe().T)
f,ax = plt.subplots(1,2, figsize = (16,8))
ax1,ax2 = ax.flatten()
sns.countplot(y = winedata['taster_name'], palette = 'cividis', ax =ax1)
ax1.set_title('Taster Name')
ax1.set_xlabel('')
ax1.set_ylabel('')
sns.countplot(y = winedata['taster_twitter_handle'], palette = 'ocean', ax =ax2)
ax2.set_title('Tasers Twiter Handle')
ax2.set_xlabel('')
ax2.set_ylabel('');

plt.figure(figsize = (16,6))
cnt = winedata.groupby(['country','taster_name',]).count().reset_index()
sns.countplot(x = cnt['country'], palette='hot')
plt.xticks(rotation = 90);

#Description Visulisation
plt.figure(figsize= (16,8))
plt.title('Word cloud of Description')
wc = WordCloud(max_words=1000,max_font_size=40,background_color='black', stopwords = STOPWORDS,colormap='Set1')
wc.generate(' '.join(winedata['description']))
plt.imshow(wc,interpolation="bilinear")
plt.axis('off')

plt.figure(figsize= (16,8))
plt.title('Word cloud of Description by France')
wc = WordCloud(max_words=1000,max_font_size=40,background_color='black', stopwords = STOPWORDS)
wc.generate(' '.join(winedata[winedata['country'] =='France']['description']))
plt.imshow(wc.recolor(colormap='Set2'),interpolation="bilinear")
plt.axis('off')

####Recommendation System Using Nearest Neighbors####
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

WineData = winedata.copy()
col = ['province', 'variety', 'points']
WineData = winedata[col]
WineDate = WineData.dropna(axis = 0)
WineData = WineData.drop_duplicates(['province', 'variety'])
WineData = WineData[WineData['points'] > 85]
WineDataPivot = WineData.pivot(index = 'variety', columns = 'province', values = 'points').fillna(0)
WineDataPivotMatrix = csr_matrix(WineDataPivot)

KNN = NearestNeighbors(n_neighbors = 10, algorithm = 'brute', metric = 'cosine')
KNNmodel = KNN.fit(WineDataPivotMatrix)

###Prediction###
QueryIndex = np.random.choice(WineDataPivot.shape[0])
distance, indice = KNNmodel.kneighbors(WineDataPivot.iloc[QueryIndex,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0, len(distance.flatten())):
    if  i == 0:
        print('Recmmendation for {0}:\n'.format(WineDataPivot.index[QueryIndex]))
    else:
        print('{0}: {1} with distance: {2}'.format(i,WineDataPivot.index[indice.flatten()[i]],distance.flatten()[i]))