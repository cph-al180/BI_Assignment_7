import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pylab import polyfit, poly1d
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
import math
import folium
from folium.plugins import HeatMap



#Uncomment if first time using Vader.
#nltk.download('vader_lexicon')
dataSet = "data/hn_items.csv"
HNData = pd.read_csv(dataSet)

def vaderNLTK(data):
    model = SentimentIntensityAnalyzer()
    data = pd.DataFrame(data)
    df = data.dropna(subset=["text"])
    pos = []
    neg = []
    index = 0
    print 'running sentiment analysis..'
    for row in df.iterrows():
        text = df.iloc[index]["text"]
        res = model.polarity_scores(text)
        pos.append(res['pos'])
        neg.append(res['neg'])
        index = index+1   
    print 'sentiment analysis finished'
    df['pos'] = pos
    df['neg'] = neg
    best = df.nlargest(5, 'pos')
    worst = df.nlargest(5, 'neg')     
    print 'Best: ', '\n', best[['text', 'pos']]
    print 'Worst: ', '\n', worst[['text', 'neg']]

    x, y, X = pos, neg, np.array([pos])
    X = X.T
    fit = np.polyfit(x,y,deg=1)
    fit_fn = np.poly1d(fit)
    plt.plot(X, y,'ro', X, fit_fn(X), 'b')
    plt.show()
    plt.savefig('HackerNewsPlot.png')
    y = np.array([neg])
    y = y.T
    folds = KFold(n_splits = 10)
    for train_indices, test_indices in folds.split(X, y):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        pl = PolynomialFeatures(degree=10, include_bias=False)
        lm = LinearRegression()
        pipeline = Pipeline([("pl", pl), ("lm", lm)])
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        score = pipeline.score(X_test, y_test)
        print score
        
def part3():
    df = pd.read_csv("boliga_zealand.csv").drop(['Index', '_1', 'Unnamed: 0'], axis=1)
    dataset = df[['lon','lat','price']].dropna()
    folium_map = folium.Map(location=[55.6831224,12.5693172], zoom_start=10)
    heat_data = [(i.lat,i.lon,float(i.price)) for i in dataset.itertuples()]
    HeatMap(heat_data, radius=8).add_to(folium_map)
    folium_map.save('heatmap.html')
    
def part4():
    distance_df = df.dropna()
    distance_df = distance_df.assign(distance_to_center=distance_df.apply(lambda row: calc_distance(row['lat'],row['lon'],55.676098, 12.568337),axis=1))
    
    regr = linear_model.LinearRegression()

    x_values= distance_df['distance_to_center'].astype(float).values.reshape(-1,1)
    y_values = distance_df['price'].astype(int).values.reshape(-1,1)
    z_values = distance_df['size_in_sq_m'].astype(float).values.reshape(-1,1)
    
    folds = KFold(n_splits=10)

    metrics_df = pd.DataFrame(columns=['Coefficients','Intercept','MAE','MSE','Pearson'])

    metrics_list = []

    for train_indices, test_indices in folds.split(x_values, y_values, z_values):

        xz_values = np.stack([x_values,y_values], axis=1).reshape(-1,2)

        xz_train, xz_test = xz_values[train_indices], xz_values[test_indices]

        y_train, y_test = y_values[train_indices], y_values[test_indices]

        regr = linear_model.LinearRegression()

        regr.fit(xz_train, y_train)

        prediction = regr.predict(xz_test)

        coef = str(regr.coef_)
        intercept = str(regr.intercept_)
        MAE = str(metrics.mean_absolute_error(y_test,prediction))
        MSE = str(math.sqrt(metrics.mean_squared_error(y_test,prediction)))
        Pearson = str(metrics.r2_score(y_test, prediction))


        metrics_list.append({'Coefficients':coef,'Intercept':intercept,'MAE':MAE,'MSE':MSE,'Pearson':Pearson})

    metrics_df = metrics_df.append(metrics_list)

    print(metrics_df)


def calc_distance(lat1,lon1,lat2,lon2):
    R = 6371; # Radius of the earth in km
    
    dLat = math.radians(lat2-lat1)
    dLon = math.radians(lon2-lon1) 
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
     
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c # Distance in km
    return d
    
def run():
    vaderNLTK(HNData)    
    part3()
    part4()
    print 'done'

run()


