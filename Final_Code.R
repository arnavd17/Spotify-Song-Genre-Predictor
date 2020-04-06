rm(list = ls())
setwd("C:/Users/pivo/Documents/GitHub/R_project")
library(readr)
library(kknn)
library(gbm)
library(mosaic)

set.seed(79643)
df_spotify <- read_csv("songDb.csv")
genres <- read_table("genre_dataset.txt", col_names = c('Genre'))


#Data cleaning
df_clean <- na.omit(df_spotify)
drop <- c('Uri','Name','Ref_Track', 'time_signature', 'Type', 'URL_features')
df_clean <- df_clean[, !(names(df_clean)%in%drop)]
df_clean <- df_clean[(df_clean$Duration_ms > 120000),]
dim(df_clean)

#We selected a couple of genres and used them to classify our data based on the content of the actual Genre column
df_genre <- df_clean
genres_aux = c('rock', 'pop', 'country','classical','hiphop','jazz','blues')
m_genre = matrix(data = NA, nrow = nrow(df_genre), ncol = 7)
colnames(m_genre) = genres_aux

for (x in genres_aux) {  #if the column Genre contains the genres we seleceted it will place a 1 in the relevant column
  df_genre[grepl(x,df_genre$Genre),x] <- 1
  df_genre[!grepl(x,df_genre$Genre),x] <- 0
}

df_genre2 <- df_genre[rowSums(df_genre[15:21])==1,]
for (x in genres_aux){ #We keep only the ones that are part of 1 column and use that genre
  df_genre2[df_genre2[x]==1, 'genre'] = x
}
df_genre2 <- df_genre2[,!(names(df_genre2) %in% genres_aux)]
df_genre2$genre = factor(df_genre2$genre)
df_genre2$Tempo = as.numeric(df_genre2$Tempo)
dim(df_genre2)
summary(df_genre2)

unique.levels <- sort(unique(df_genre2$genre))
count <- table(df_genre2$genre)
count.df <- data.frame(unique.levels, count)
#Now plot:
plot <- ggplot(count.df, aes(unique.levels, Freq, fill=unique.levels))
plot + geom_bar(stat="identity") + 
  labs(title="Genre Count",
       subtitle="count for every genre in the dataset",
       y="Count", x="Genre") + 
  theme(legend.position="none")


## NEW DATA SET
r=df_genre2[df_genre2$genre=='rock',][1:1171,]
p=df_genre2[df_genre2$genre=='pop',][1:1171,]
h=df_genre2[df_genre2$genre=='hiphop',][1:1171,]
b=df_genre2[df_genre2$genre=='blues',][1:1171,]
co=df_genre2[df_genre2$genre=='country',][1:1171,]
cl=df_genre2[df_genre2$genre=='classical',][1:1171,]
j=df_genre2[df_genre2$genre=='jazz',][1:1171,]
songs=data.frame(rbind(r,p,h,b,co,cl,j))

# RUN THIS TO CHANGE TO BALANCED DATASET
df_genre2 <- songs


### Train - Test division
train = .7
train_sample = sample(1:nrow(df_genre2), nrow(df_genre2)*train)
train_df = df_genre2[train_sample,]
test_df = df_genre2[-train_sample,]

### Scale for KNN and other methods
scale_train_df <- train_df
scale_train_df[,-c(12,14,15)]<- scale(train_df[,-c(12,14,15)]) 
scale_test_df <- test_df
scale_test_df[,-c(12,14,15)] <- scale(test_df[,-c(12,14,15)])


### KNN ###
#General model using all the variables and cross validation to find K
model_knn <- train.kknn(genre~(.-ID-Mode-Genre), data=scale_train_df, kmax = 100, kcv = 10)
model_knn$best.parameters
final_model = kknn(genre~(.-ID-Mode-Genre),train_df,test_df,k= model_knn$best.parameters$k, kernel = "rectangular")
sum(diag(table(final_model$fitted.values, test_df$genre)))/length(final_model$fitted.values)

#Using only the most important variables we found and cross validation for K
model_knn2 <- train.kknn(genre~(Danceability + Energy + Loudness + Speechness+Acousticness+Instrumentalness+Valence+Liveness+Duration_ms), data=scale_train_df, kmax = 100, kcv = 10)
model_knn2$best.parameters
final_model2 = kknn(genre~(Danceability + Energy + Loudness + Speechness+Acousticness+Instrumentalness+Valence+Liveness+Duration_ms),train_df,test_df,k= model_knn$best.parameters$k, kernel = "rectangular")
sum(diag(table(final_model2$fitted.values, test_df$genre)))/length(final_model2$fitted.values)

### TREE ###
# Classification tree
library(tree)
set.seed(79643)
# choose the x,y columns and build a tree model
data = read.csv("train.csv")
df=data[,c(1:11,13,15)]
attach(df)
set.seed(79643)
train = sample(1:nrow(df), nrow(df)*0.02)
tree.music=tree(genre~.,data=df, subset = train)
summary(tree.music)
plot(tree.music)
text(tree.music,pretty=0)
MSE=NULL
#estimate the test error using test dataset
test=read.csv("test.csv")
test = test[,c(1:11,13,15)]
tree.pred = predict(tree.music,test, type = 'class')
result = data.frame(test$genre,tree.pred)
result[result$tree.pred == result$test.genre,"Equal"] <- 1
accuracy_tree = nrow(subset(result, result$Equal == 1)) / nrow(result)
accuracy_tree 

# Prune the tree model 
prun.tree=prune.tree(tree.music,best=8)
# plot the prune tree
plot(prun.tree,type="uniform")
text(prun.tree,pretty=0)
# estimate the test error of prune tree using test dataset
pruntree.pred = predict(prun.tree,test, type = 'class')
result = data.frame(test$genre,pruntree.pred)
result[result$pruntree.pred == result$test.genre,"Equal"] <- 1
accuracy_pruntree = nrow(subset(result, result$Equal == 1)) / nrow(result)
accuracy_pruntree 







### BAGGING ###
library(randomForest)
set.seed (79643)
Accuracy_bagging = NULL
ntree <-c(50,100,200,500,2000)
# try trees = 50,100,200,500,2000 with all the variables
for (i in ntree){
  bag.music =randomForest(genre ~ Danceability + Energy + Key + Loudness + Mode + Speechness + Acousticness + Instrumentalness + Liveness + Valence + Tempo + Duration_ms, data=train_df ,
                          mtry=12, ntree = i) 
  # predict reuslt and calculate accuracy rate
  yhat.bag = predict(bag.music,newdata = test_df)
  aux =  mean( yhat.bag == test_df$genre)
  # get a list of accuracy rates 
  Accuracy_bagging = c(Accuracy_bagging,aux)
}

# plot number of trees versus accuracy rates
plot(ntree, Accuracy_bagging,type="b",xlab="ntree",col="blue",ylab="Accuracy",lwd=2,cex.lab=1.2, main = "ntree vs. Accuracy")
# get highest accuracy rate
Accuracy_bagging[which.max(Accuracy_bagging)]
set.seed (79643)
# training model with the optimal number of trees and splits
model_bagging<-randomForest(genre~.-ID-Genre,data=train_df,ntree=2000,mtry=12,importance=TRUE)

# To check important variables
importance(model_bagging)      
varImpPlot(model_bagging) 

### BOOSTING ###
# cross validation was left out of final code due to the time it took (we ran a nested for loops for n.trees and shrinkage)
ntrees=1000
boostfit = gbm(genre~.-ID-Genre-Key-Mode,data=train_df,distribution='multinomial', #Multinomial makes the tree take longer to run but it's the output we need
               interaction.depth=5,n.trees=ntrees,shrinkage=.01)
pred = predict(boostfit,newdata=test_df,n.trees=ntrees, type = 'response')
df_p = data.frame(pred)
colnames(df_p) = c('blues', 'classical','country','hiphop','jazz','pop','rock')
df_p['genre'] = colnames(df_p)[apply(df_p,1,which.max)] #gets the predicted genre out of the top probability
sum(diag(table(test_df$genre, df_p$genre)))/length(test_df$genre)

### RANDOM FOREST ###

#fit random forest and plot variable importance
# array of number of tree values to use
ntr<-c(50,200,500,2000,5000)
max_acc=0
# Training model with different number of trees and splits to get the optimal values for each
for (n in ntr){
  a=c()
  i=5
  for (i in 3:8) {
    model_rf <- randomForest(genre~.-ID-Genre, data = train_df, ntree = n, mtry = i, importance = TRUE)
    predValid <- predict(model_rf, test_df, type = "class")
    a[i-2] = mean(predValid == test_df$genre)
    if (a[i-2]>max_acc){
      max_acc=a[i-2]
      opt_tree=n
      opt_m=i
    }
  }
  print(paste0('Number of trees: ',n))
  print(a)
}

# training model with the optimal number of trees and splits
model_rf<-randomForest(genre~.-ID-Genre,data=train_df,ntree=opt_tree,mtry=opt_m,importance=TRUE)
# To check important variables
importance(model_rf)      
# plotting the importance of predictors
varImpPlot(model_rf) 
# testing for completely new data
s<-data.frame("Danceability"= 0.326,
              "energy"= 0.0993,
              "key"= 7,
              "loudness"= -22.496,
              "mode"= 1,
              "Speechness"= 0.072,
              "acousticness"= 0.988,
              "Instrumentalness"= 0.916,
              "Liveness"= 0.108,
              "Valence"= 0.417,
              "Tempo"=137.274,
              "Duration_ms"=150973)

# predicting the genre
predValid<-predict(model_rf_4,newdata=s)
predValid

# model<- c('KNN','Classification Tree','Bagging', 'Random Forest', 'Boosting')
# skew <- c(0.5388, 0.4495, 0.6015, 0.61, 0.5652)
# balanced <-c(0.5293, 0.4682, 0.6199, 0.6310, 0.5988)
# 
# df_model <- data_frame(model,skew,balanced)
# df_model
# ggplot(df_model, aes(x=model)) +
#   geom_bar(aes(...))

           