## take a look some of the data available

prop.table(table(traindata$Sex))
#    female      male 
# 0.3618677 0.6381323 

prop.table(table(traindata$Survived))
#         0         1 
# 0.5894942 0.4105058 

prop.table(table(traindata$Sex, traindata$Survived))      
#                 0          1
# female 0.08949416 0.27237354
# male   0.50000000 0.13813230

## install glmnet to make use of Lasso regularization
install.packages("glmnet", repos = "http://cran.us.r-project.org")
require(glmnet)

# Sex field needs to be numeric to run binomial family for LASSO
traindata.factored <- traindata
traindata.factored$Sex <- as.numeric(traindata.factored$Sex)
testdata.factored <- testdata
testdata.factored$Sex <- as.numeric(testdata.factored$Sex)

# get the LASSO model
lasso.model <- glmnet(as.matrix(traindata.factored[,2:5]), as.matrix(traindata.factored[,1]), family="binomial", alpha=1)

# get cross-validated LASSO model to estimate the optimal lambda
cv.lasso.model <- cv.glmnet(as.matrix(traindata.factored[,2:5]), as.matrix(traindata.factored[,1]), family="binomial", alpha=1)

# estimated optimal lambda
cv.lasso.model$lambda.min # 0.00245966

plot(lasso.model)
plot(cv.lasso.model)

# get the ROC curve of the cross-validated LASSO model
auc.cv.lasso.model <- cv.glmnet(x = as.matrix(traindata.factored[,2:5]), y = as.matrix(traindata.factored[,1]), family="binomial", alpha=1, type.measure="auc")
test.prob <- predict(auc.cv.lasso.model, type="response", newx = as.matrix(testdata.factored[,2:5]), s = 'lambda.min')

# predict the test data on whether someone survived or not
library(ROCR)
require(ROCR)
pred.auc <- prediction(test.prob[,1], testdata.factored[,1])
performance.auc <- performance(pred.auc, "tpr", "fpr")
performance.auc(pred.auc, "auc")
performance(pred.auc, "auc")

# plot the ROC curve
plot(performance.auc, colorizer=TRUE)
