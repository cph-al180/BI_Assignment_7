# Business Intelligence Assignment 6 - Multivariate linear regression and logistic regression  

## Part 1   - HackerNews MAE & RMSE

Theoretically speaking, if the new metric has any correlation with the amount of karma a given user has, then the new model should have better results. If the results are the same or worse, then the new metric adds nothing to the model.  
As we can see from the results, the MAE is quite a bit lower. The RMSE is still fairly high, but it makes sense that the result with testing data is higher than training, which counts for both MAE and RMSE (using new data). Overall the new results are a big improvement.  
However, the difference in score between the 2 models is pretty interesting. Strangely enough, the score of the new model is worse than the old one. If we again look at the training results from the old and new model, we can see that the new model is vastly better. However, if we only compare the testing results, then the new model isn't that much better, and possbily even worse. This highlights the issue with using the 80/20 split, as the testing data from the new model clearly deviates far from the model, which results in a worse score.

Without # of posts (old):  
`MAE (Training): 4535.2278195244253`  
`MAE (Testing): 4363.9936837520208`  
`RMSE (Training): 10230.170612147558`  
`RMSE (Testing): 7858.119559984959`  
`Score: 0.121689402714`  

With # of posts (new):  
`MAE (Training): 1685.15838569`  
`MAE (Testing): 3621.08926426`  
`RMSE (Training): 4420.63815212`  
`RMSE (Testing): 10015.2422784`   
`Score: -0.0741407688896`

Training:  
```python
def trainModel():
    global model
    x, y = TRAIN_X, train_karma
    model = linear_model.LinearRegression()
    model.fit(x, y)  
```
  
MAE:  
```python
def calcMAE():
    train_karma_pred = model.predict(TRAIN_X)
    test_karma_pred = model.predict(TEST_X) 
    train_MAE = mean_absolute_error(train_karma, train_karma_pred)
    test_MAE = mean_absolute_error(test_karma, test_karma_pred)
```  
  
RMSE:  
```python
def calcRMSE():
    train_karma_pred = model.predict(TRAIN_X)
    test_karma_pred = model.predict(TEST_X)
    train_MSE = mean_squared_error(train_karma, train_karma_pred)
    test_MSE = mean_squared_error(test_karma, test_karma_pred)
    train_MSE = math.sqrt(train_MSE)
    test_MSE = math.sqrt(test_MSE)
```   

## Part 2 - K-Fold Cross Validation  

As we can see from the results, all of the metrics from the 10-fold Cross Validation are better than the regular 80/20 split. The biggest outliers are naturally the results using testing data, since we use ALL of the data for testing, and not just 20%.  
As previously mentioned, this is because of the inconsistency of the 80/20 split, where the last 20% might deviate heavily from the training data.  
The score clearly shows the issue with the 80/20 split, since the score of the 10-fold Cross Validation is much higher than the score of the model using the 80/20 split, which explains why the score of the 80/20 model was lower than the old model, which did not include # of posts.

Average 10-Fold Cross Validation results:  
`MAE (Training): 1685.38972976`  
`MAE (Testing): 1685.38972976`  
`RMSE (Training): 4419.64858547`  
`RMSE (Testing): 4368.67775664`   
`Score: 0.741106182258`  
  
K-Fold Cross Validation:  
```python
def kfold():
    X = []
    y = []
    x_temp_1 = []
    x_temp_2 = []
    total_score = 0
    total_train_MAE = 0
    total_test_MAE = 0
    total_train_RMSE = 0
    total_test_RMSE = 0
    
    for i in training_data:
        if not i.has_key('karma') or not i.has_key('created') or not i.has_key('submitted'):
            i["karma"] = 0;
            i["created"] = 1509813038
            i["submitted"] = 0
        y.append(i["karma"])
        x_temp_1.append(i["created"])
        x_temp_2.append(i["submitted"])      
    X = np.array([x_temp_1, x_temp_2])
    X = X.T
    y = np.array([y])
    y = y.T
    
    folds = KFold(n_splits = 10)
    for train_indices, test_indices in folds.split(X, y):
        i = i+1
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        pl = PolynomialFeatures(degree=1, include_bias=False)
        lm = LinearRegression()
    
        pipeline = Pipeline([("pl", pl), ("lm", lm)])
        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_RMSE = mean_squared_error(y_train, y_train_pred)
        test_RMSE = mean_squared_error(y_test, y_test_pred)
        train_RMSE = math.sqrt(train_RMSE)
        test_RMSE = math.sqrt(test_RMSE)

        train_MAE = mean_absolute_error(y_train, y_train_pred)
        test_MAE = mean_absolute_error(y_test, y_test_pred)

        total_train_MAE = total_train_MAE + train_MAE
        total_test_MAE = total_test_MAE + test_MAE
        total_train_RMSE = total_train_RMSE + train_RMSE
        total_test_RMSE = total_test_RMSE + test_RMSE
        total_score = total_score + pipeline.score(X_test, y_test)

    print '10-fold avg Score:', total_score / 10
    print '10-fold avg Train MAE:', total_train_MAE / 10
    print '10-fold avg Test MAE:', total_test_MAE / 10
    print '10-fold avg Train RMSE:', total_train_RMSE / 10
    print '10-fold avg Test RMSE:', total_test_RMSE / 10
``` 

## Part 3 - Logistic Model
The dataset consists of ID, diagnosis and 10 parameters describing the cancer, these 10 parameters all have an average value, a standard error and a worst

We can say with 92.9 percent confidence whether the tumor is benign or malignant
