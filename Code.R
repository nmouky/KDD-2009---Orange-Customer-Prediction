#### ==== Packages ==== #####
library('dplyr')
library('readr')
library('finalfit') 
library('gbutils') 
library("Hmisc")
library("magrittr")
library('VIM')
library('mice')
library('corrplot')
library('caret')
library('psych')
library('Stack')
library('magrittr')
library('imputeMissings')
#library('tidyverse')
library('stats')
library('fastDummies')
library('aCRM')
library('DMwR')
library('glmnet')
library('car')
library('caret')
library('e1071')
library('schoolmath')
library('CORElearn')
library('AppliedPredictiveModeling')
library('pROC')
library('naniar')
library(readr)
library(ggplot2)
library(doParallel)


#### ========== Reading in the data ============= ####

# Import the Data

train_X <- read_delim("Desktop/Data Science/Machine learning/assignment2_data/train_X.csv",
                      "\t", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)

test_X <- read_delim("Desktop/Data Science/Machine learning/assignment2_data/test_X.csv",
                     "\t", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)

train_Y <- read_delim("Desktop/Data Science/Machine learning/assignment2_data/train_Y.csv",
                      "\t", escape_double = FALSE, col_names = TRUE, trim_ws = TRUE)


# Replace -1 Level with 0
train_Y$churn = ifelse(train_Y$churn == -1,0,1)
train_Y$appetency = ifelse(train_Y$appetency == -1,0,1)
train_Y$upselling = ifelse(train_Y$upselling == -1,0,1)


# Randomise 
rand_data <- cbind(train_X, train_Y)
rand_data <- rand_data[sample(nrow(rand_data)),]

train_X <- rand_data[,1:230]
train_Y <- rand_data[,231:233]


# Bind the Train_X and Train_Y 
all_data <- rbind(train_X, test_X)
all_data[all_data==""]<-NA


#### ========== Descriptive Statistics ============= ####

#Checking the imbalance
table(train_Y$appetency)
table(train_Y$churn)
table(train_Y$upselling)

counts <- table(train_Y$appetency)
barplot(counts, main="Appetency",
        xlab="Number of Customers", col=c("steelblue4","grey"),
        legend = rownames(counts))

counts <- table(train_Y$churn)
barplot(counts, main="Churn",
        xlab="Number of Customers", col=c("steelblue3","grey"),
        legend = rownames(counts))

counts <- table(train_Y$upselling)
barplot(counts, main="Upselling",
        xlab="Number of Customers", col=c("steelblue2","grey"),
        legend = rownames(counts))



# 4 classes: 'spec_tbl_df', 'tbl_df', 'tbl' and 'data.frame'
str(train_X)
str(test_X)
str(train_Y)

# Number of NA's in each dataframe
n_miss(train_X) #5296131
n_miss(test_X)  #2728021
n_miss(train_Y) #0

# Percentage of datasets that encompass NA values
pct_miss(train_X) #69.77563
pct_miss(test_X)  #69.77446

ff_glimpse(train_X)

# Table of variables with number of NA's in them
miss_var_table(train_X)
miss_var_table(test_X)

# Sorts variables with most to least instances of NA 
#vis_miss(train_X, sort = TRUE, warn_large_data = FALSE)
#vis_miss(test_X, sort = TRUE, warn_large_data = FALSE)


#### ========== Variables Reduction and Missing Values ============== ####


## Looking for any patterns in the missingness

over_80 <- all_data[colSums(is.na(all_data))/nrow(all_data) <1]
over_80 <- over_80[colSums(is.na(over_80))/nrow(over_80) >.80]

ff_glimpse(over_80)
missing_plot(over_80)


# Removing all variables where the degree of missingness is greater than 80%
all_data <- all_data[colSums(is.na(all_data))/nrow(all_data) < .80]

# Creating a new variable which takes into account the number of missing values per observation (person) in the dataframe
all_data$na_row <- apply(all_data, 1, function(x) sum(is.na(x)))

# Informative Missingness:
miss_test <- all_data[colSums(is.na(all_data))/nrow(all_data) > .2] #All variables that are more than 20% (but less than 80%) missing
dim(miss_test) #ten variables 

#Recoding them so an NA value results in 0 and an non-missing value in 1
miss_test[!is.na(miss_test)] <- 1
miss_test[is.na(miss_test)] <- 0

#join the Y values, but only to the train set
miss_sig <- cbind(miss_test[1:23100,], train_Y[1:23100,])


#using chi-square to check again for significance 

ch_dat <- miss_test[1:23100,] #training only

ch_dat$X4 <- as.factor(ch_dat$X4)  #converting all variables to factors
ch_dat$X22 <- as.factor(ch_dat$X22)
ch_dat$X27 <- as.factor(ch_dat$X27)
ch_dat$X36 <- as.factor(ch_dat$X36)
ch_dat$X50 <- as.factor(ch_dat$X50)
ch_dat$X70 <- as.factor(ch_dat$X70)
ch_dat$X84 <- as.factor(ch_dat$X84)
ch_dat$X114 <- as.factor(ch_dat$X114)
ch_dat$X126 <- as.factor(ch_dat$X126)
ch_dat$X182 <- as.factor(ch_dat$X182)


churn_chi <- lapply(ch_dat, function(x) chisq.test(x = x , y = miss_sig$churn))  #chi-square test for all variables with each outcome
app_chi <- lapply(ch_dat, function(x) chisq.test(x = x , y = miss_sig$appetency))
up_chi <- lapply(ch_dat, function(x) chisq.test(x = x , y = miss_sig$upselling))

churn_chi


#I now have lists of variables which may have informative missingness as they are significant 
missing_sig <- colnames(miss_test)[-c(4)]



#Dataframes of the new variable misssing data to be added at the end
all_miss <- miss_test[missing_sig]


#col_names <- colnames(churn_miss)
#churn_miss[,col_names] <- lapply(churn_miss[,col_names] , as.numeric)

#col_names <- colnames(app_miss)
#app_miss[,col_names] <- lapply(app_miss[,col_names] , as.numeric)

#col_names <- colnames(up_miss)
#up_miss[,col_names] <- lapply(up_miss[,col_names] , as.numeric)

#deleting all variables with missing levels over 20% as we have created new representations of them when neccessary 
all_data <- all_data[colSums(is.na(all_data))/nrow(all_data) < .20]

####==========Removing all variables where the variance is low==========#####
novar <- nearZeroVar(all_data)
all_data <- all_data[,-novar]

####========================Checking for multicollinearity=====================####
corr_data <- cor(na.omit(Filter(is.numeric, all_data[1:23100,])))

#Significance values for the correlations 
corr_p <- rcorr(as.matrix(na.omit(Filter(is.numeric, all_data[1:23100,]))))
corr_p[is.na(corr_p)] <- 1 #changing NA to 1

#Correlation plot that also shows which correlations are significant
corrplot(corr_data, method="color", type='lower', p.mat=corr_p$P, sig.level=.05)

#Removing highly correlated variables so i can use my dataset for imputation
#Solution found here: https://stackoverflow.com/questions/18275639/remove-highly-correlated-variables

corr_var <- findCorrelation(corr_data, cutoff=0.6, names=TRUE) 
corr_var <- sort(corr_var)

#Removing all variables with a correlation of over .6 
all_data <-all_data[, !(colnames(all_data) %in% c(corr_var))]


#Checking that worked which it did :)
corr_test <- cor(na.omit(Filter(is.numeric, all_data)))
corrplot(corr_test, method="color", type='lower')

dim(all_data) #44 variables left

####======================Imputation=======================####

#Change to factor
chrac <- all_data[, sapply(all_data, class) == 'character']
chr_names <- colnames(chrac)
all_data[,chr_names] <- lapply(all_data[,chr_names] , factor)

fact_data_all <- Filter(is.factor, all_data) # factor data

chr_names <- colnames(fact_data_all)

## Categorical variable handling

#We removed any factor variable with more than 10 levels. They are not practical for dummification 

fact_data <- fact_data_all[, sapply(fact_data_all, function(col) length(unique(col))) <= 5000]

num_data <- all_data[, !(colnames(all_data) %in% c(colnames(fact_data_all)))] # numerical data

#I start by using the mice package to impute the numerical variables 
mice_num <- mice(num_data, m=1, seed = 5, method = 'mean', maxit=1) #5 )
complete_num <- mice::complete(mice_num,1)

save.image('after_mice')
#load('after_mice')
write_csv(complete_num, path= 'complete_num.csv')

#Imputing missing factor data 

#1. Using the mode
#mode_fact <- imputeMissings(fact_data)
#missing_plot(mode_fact)

#2. Here we replace all NA values with the level 'unknown'. The data must be character to do this so we change it back.
fact_data[,1:length(fact_data)] <- lapply(fact_data[,1:length(fact_data)], as.character)
fact_data[is.na(fact_data)]<-'Unknown'
unknown_fact <- fact_data
missing_plot(unknown_fact)

chr_names <- colnames(fact_data)

####===========Verify Skewness of the Numerical variables =====######

(skewValues = apply(complete_num, 2, skewness))

(neg_cols<-names(complete_num)[sapply(complete_num, function(x) min(x)) < 0])


# I select only positive values, for them I add + 1 cause BoxCox works for values > 0
# Adding 1 to the distribution does not have influence on the skewness, cause it keeps 
# the distribution equal

complete_num <- complete_num + 1 

# Apply BoxCox for these values
complete_num = as.data.frame(complete_num) %>%
  mutate_at(vars(-c(2,11))  ,funs( BoxCoxTrans(.) %>% predict(.)))


# Apply YeoJohnson for Variables with Negative values
YeoJ = preProcess(complete_num[, c(2,10)], method = 'YeoJohnson')
complete_num = predict(YeoJ, complete_num)

# Verify if the Process worked
(skewValues = apply(complete_num, 2, skewness))

#####======= Standardising the Data =======########

standardz = complete_num %>% 
  preProcess( method = c('scale','center'))
complete_num = predict(standardz, complete_num)

####====== Remove Outliers =====#####

complete_num = spatialSign(complete_num)

complete_num <- as.data.frame(complete_num)



#New master dataset

unknown_fact <- as.data.frame(unclass(unknown_fact))

all_data <- cbind(complete_num, unknown_fact)

all_data <- cbind(all_data, all_miss)

all_data$X4 <- as.factor(all_data$X4)
all_data$X22 <- as.factor(all_data$X22)
all_data$X27 <- as.factor(all_data$X27)
all_data$X50 <- as.factor(all_data$X50)
all_data$X126 <- as.factor(all_data$X126)
all_data$X182 <- as.factor(all_data$X182)

#### Categorical feature level reduction ####
high_level <- Filter(is.factor, all_data)

high_level <- high_level[, sapply(high_level, function(col) length(unique(col))) > 10]

str(high_level)

sort(table(high_level$X189), decreasing = TRUE)[1:13]

library("tidyverse")

all_data$X90 <-fct_lump(high_level$X90,n =20, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X93 <-fct_lump(high_level$X93,n =10, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X116 <-fct_lump(high_level$X116,n =30, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X119 <-fct_lump(high_level$X119,n =50, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X138 <-fct_lump(high_level$X138,n =50, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X156 <-fct_lump(high_level$X156,n =50, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X187 <-fct_lump(high_level$X187,n =10, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X190 <-fct_lump(high_level$X190,n =18, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X29 <-fct_lump(all_data$X29,n = 13, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X154 <-fct_lump(all_data$X154,n = 8, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X189 <-fct_lump(all_data$X189,n = 13, w = NULL, other_level = "Other",ties.method = c("min"))
all_data$X204 <-fct_lump(all_data$X204,n = 5, w = NULL, other_level = "Other",ties.method = c("min"))

#### Relief feature selection ####
#Churn

set.seed(1)

relief_churn <- cbind(all_data[0:33001,], 'churn'= train_Y$churn)
relief_churn_train <- relief_churn[1:23100,]
relief_churn_train$churn <- as.factor(relief_churn_train$churn)
relief_churn_train <- SMOTE(churn ~ ., relief_churn_train, perc.over = 200,perc.under=200, knn = 5)
#table(relief_churn_train$churn)

set.seed(1)

perm = permuteRelief(x = relief_churn_train[,-length(relief_churn_train)],
                     y = relief_churn_train$churn,
                     nperm = 500,
                     estimator = 'ReliefFequalK',
                     ReliefIterations = 100)

st_dev = perm$standardized

st_dev
relief_var = st_dev[abs(st_dev) > 1.65] # variables to keep 

relief_var
relief_var = names(relief_var[!is.na(relief_var)]) # Remove Missing values and keep only the key names


churn_all = all_data[, relief_var] # Create dataset with only variables selected by Relief


####Binding the data frames together ####
set.seed(7)

churn_test <- churn_all[33002:nrow(churn_all),]   # Test dataset
write_csv(churn_test, path= 'churn_test_1.csv')
churn_train_all <- cbind(churn_all[1:33001,], 'churn' = train_Y$churn)
churn_train <- churn_train_all[1:23100,]
churn_train$churn <- as.factor(churn_train$churn)
churn_train <- SMOTE(churn ~ ., churn_train, perc.over = 100,perc.under=200, knn = 5) # Train dataset
churn_train <- churn_train[sample(nrow(churn_train)),]
write_csv(churn_train, path= 'churn_train_1.csv')
churn_val <- churn_train_all[23101: nrow(churn_train_all),] # Validation dataset
write_csv(churn_val, path= 'churn_val_1.csv')

#save.image('after pre_processing')
#load('after pre_processing')

#churn_train <- read.csv("churn_train.csv", header = TRUE, na='?')
#churn_val <- read.csv("churn_val.csv", header = TRUE, na='?')


####Recheck factor levels ####

copy_train <- churn_train
copy_val <- churn_val

recheck <- rbind(churn_train[-35], churn_val[-35])
recheck <- rbind(recheck, churn_test)


high_level <- Filter(is.factor, recheck)

high_level <- high_level[, sapply(high_level, function(col) length(unique(col))) > 2]

str(high_level)

sort(table(high_level$X190), decreasing = TRUE)[1:19]


recheck$X29 <-fct_lump(high_level$X29,n =10, w = NULL, other_level = "Other",ties.method = c("min"))
recheck$X79 <-fct_lump(high_level$X79,n =5, w = NULL, other_level = "Other",ties.method = c("min"))
recheck$X90 <-fct_lump(high_level$X90,n =7, w = NULL, other_level = "Other",ties.method = c("min"))
#recheck$X112 <-fct_lump(high_level$X112,n =5, w = NULL, other_level = "Other",ties.method = c("min"))
recheck$X116 <-fct_lump(high_level$X116,n =10, w = NULL, other_level = "Other",ties.method = c("min"))
recheck$X119 <-fct_lump(high_level$X119,n =12, w = NULL, other_level = "Other",ties.method = c("min"))
recheck$X125 <-fct_lump(high_level$X125,n =2, w = NULL, other_level = "kIsH",ties.method = c("min"))
#recheck$X138 <-fct_lump(high_level$X138,n =10, w = NULL, other_level = "Other",ties.method = c("min"))
recheck$X154 <-fct_lump(high_level$X154,n =7, w = NULL, other_level = "Other",ties.method = c("min"))
#recheck$X155 <-fct_lump(high_level$X155,n =18, w = NULL, other_level = "Other",ties.method = c("min"))
recheck$X190 <-fct_lump(recheck$X190,n = 13, w = NULL, other_level = "Other",ties.method = c("min"))
recheck$X187 <-fct_lump(recheck$X187,n = 7, w = NULL, other_level = "Other",ties.method = c("min"))
#recheck$X212 <-fct_lump(recheck$X212,n = 5, w = NULL, other_level = "Other",ties.method = c("min"))




churn_test <- recheck[16662:nrow(recheck),]
churn_train <- cbind(recheck[1:6760,], 'churn' = copy_train$churn)
churn_val <- cbind(recheck[6761:16661,], 'churn' = copy_val$churn)



####Random Forest####
library('randomForest')
# churn

#churn_val$churn <- as.factor(churn_val$churn)

rfctrl <- trainControl(method='repeatedcv', 
                       number=5, 
                       repeats=3,
                       search = 'random')
mtry <- sqrt(27)
tunegrid <- expand.grid(.mtry=mtry)

rf_churn <- train(churn_train[,-13], as.factor(churn_train[,13]), 
                  method='rf', 
                  metric='Kappa', 
                  tuneLength=10, 
                  trControl=rfctrl)

rfcm <- caret::confusionMatrix(
  data = as.factor(predict(rf_churn, churn_val[,-13])),
  reference = as.factor(churn_val$churn)
)

(kappa_all = rfcm$overall[2])
(sensitivity_all = rfcm$byClass[1])
(specificity_all = rfcm$byClass[2])
(trapezoid_all = rfcm$byClass[11])
(precision_all = rfcm$byClass[5])
(recall_all = rfcm$byClass[6])



####XGBoost####

library(caret)
library(plyr)
library(xgboost)

set.seed(0) 

##https://stackoverflow.com/questions/49984506/caretxgtree-there-were-missing-values-in-resampled-performance-measures


#Custom function that allows model to train for specificity and sensitivity rather tha accuracy


fourStats <- function (data, lev = levels(data$obs), model = NULL) {
  out <- c(twoClassSummary(data, lev = levels(data$obs), model = NULL))
  coords <- matrix(c(1, 1, out["Spec"], out["Sens"]), 
                   ncol = 2, 
                   byrow = TRUE)
  colnames(coords) <- c("Spec", "Sens")
  rownames(coords) <- c("Best", "Current")
  c(out, Dist = dist(coords)[1])
}


xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = TRUE,
  classProbs = TRUE,
  search = "random",
  savePredictions = "final"
  ,summaryFunction = fourStats
)


xgbGrid <- expand.grid(nrounds = 250,
                       max_depth = 5,
                       colsample_bytree = 1,
                       eta = 0, 
                       gamma= c(0,0.5,1),
                       min_child_weight = 1,
                       subsample = 1)


#set.seed(0) 



churn_train$churn <- ifelse(churn_train$churn == 1, 'yes', ifelse(churn_train$churn == 0, 'no', NA))

churn_train$churn <- as.factor(churn_train$churn)

xgb_model = train(churn ~., data=churn_train,  
                  trControl = xgb_trcontrol,
                  #tuneGrid = xgbGrid,
                  method = "xgbTree",
                  metric="Dist",
                  tuneLength = 50,
                  scale_pos_weight = sum(churn_train$churn == 'yes')/sum(churn_train$churn == 'no'),
                  verbose = 1,
                  maximize = FALSE)
xgb_model$bestTune


churn_val$churn <- ifelse(churn_val$churn == 1, 'yes',
                          ifelse(churn_val$churn == 0, 'no', NA))


cm_gbm <- caret::confusionMatrix(
  data = as.factor(predict(xgb_model, churn_val)),
  reference = as.factor(churn_val$churn)
)

cm_gbm

(kappa_all = cm_gbm$overall[2])
(sensitivity_all = cm_gbm$byClass[1])
(specificity_all = cm_gbm$byClass[2])
(trapezoid_all = cm_gbm$byClass[11])
(precision_all = cm_gbm$byClass[5])
(recall_all = cm_gbm$byClass[6])


#### AUC  ####

require(pROC)
require(ROCR)

xgb_model$pred$R

plot(roc(xgb_model$pred$obs,xgb_model$pred$no))



## Train values
pred <- predict(xgb_model, type='raw')

pred <- ifelse(pred == 'yes', 1,
               ifelse(pred == 'no', 0, NA))
actual_train <- churn_train$churn

actual_train <- ifelse(actual_train == 'yes', 1,
                       ifelse(actual_train == 'no', 0, NA))


                       

## Val values
prob <- predict(xgb_model,churn_val, type='prob')


pred_val <- predict(xgb_model,churn_val, type='raw')
pred_val <- ifelse(pred_val == 'yes', 1,
               ifelse(pred_val == 'no', 0, NA))

actual_val <- churn_val$churn


actual_val <- ifelse(actual_val == 'yes', 1,
                       ifelse(actual_val == 'no', 0, NA))


roc <- roc(actual_val, prob$yes)



roc
plot(roc)

####Postprocessing####

ci.coords(roc, x = "best",
          input=c("threshold", "specificity", "sensitivity"),
          ret=c("threshold", "specificity", "sensitivity"),
          best.method=c("youden", "closest.topleft"), best.weights=c(1, 0.5),
          best.policy = c("stop", "omit", "random"),
          conf.level=0.95, boot.n=2000,
          boot.stratified=TRUE,
          progress=getOption("pROCProgress")$name)

table(churn_val$churn, prob$yes > .39)

####Final predictions####
test_predict <- predict(xgb_model, churn_test)

test_predict <- ifelse(test_predict == 'yes', 1, ifelse(test_predict == 'no', -1, NA))

test_predict <- as.data.frame(test_predict)

write_csv(test_predict, path= 'churn_predictions.csv')


length(test_predict)

length(churn_test)
