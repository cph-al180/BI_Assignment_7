# Business Intelligence Assignment 7 - Text classification and geographical data

## Part 1 - Finding positive/negative posts

Top 5 positive:  
`Text: "sure" - pos: 1.0`  
`Text: "ha!" - pos: 1.0`  
`Text: "Beautiful." - pos: 1.0`  
`Text: "Great, thanks!" - pos: 1.0`  
`Text: "True" - pos: 1.0`  

Top 5 negative:  
`Text: "dupe" - neg: 1.0`  
`Text: "spam" - neg: 1.0`  
`Text: "No." - neg: 1.0`  
`Text: "dupe. " - neg: 1.0`   
`Text: "desperation" - neg: 1.0`

## Part 2 - Classifying post emotions

10-Fold Cross Validation results:  
`1: 0.00561715393813`  
`2: 0.0324921025367`  
`3: -0.00537660312359`  
`4: 0.0312217483345`  
`5: 0.0288416783115`  
`6: 0.0112348025737`  
`7: 0.0388012324612`  
`8: 0.021099872471`  
`9: 0.0196133656656`  
`10: 0.00498961244902` 
  
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

