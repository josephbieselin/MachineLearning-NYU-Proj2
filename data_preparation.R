# read in data
data <- read.csv("http://christianherta.de/lehre/dataScience/machineLearning/data/titanic-train.csv",header=T)
dim(data) # [1] 891 12

# get rid of unecessary columns and unusable cases
data <- data[,c(2,3,5,6,10)]
data <- data[complete.cases(data),]
dim(data) # [1] 714 5

# set seed so sample returns the same thing every run
set.seed(101)

# get the training and testing data
testing <- sample(714, 200, replace=FALSE)
traindata <- data[-testing, ]
testdata <- data[testing, ]
dim(traindata) # 514 5
dim(testdata) # 200 5