# Business Intelligence Assignment 7 - Text classification and geographical data

## Part 1 - Finding positive/negative posts

Top 5 positive:  
`a`  
`a`  
`a`  
`a`  
`a`  

Top 5 negative:  
`a`  
`a`  
`a`  
`a`   
`a`

## Part 2 - Classifying post emotions

10-Fold Cross Validation results:  
`1`  
`2`  
`3`  
`4`  
`5`  
`6`  
`7`  
`8`  
`9`  
`10` 
  
Code for Part 1 & 2:  
```python
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
``` 

## Part 3 - Housing price heatmap 

## Part 4 - Housing price model

