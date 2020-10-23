rm(list = ls())    #delete objects
cat("\014")        #clear console
library(keras)  
library(tensorflow)
#Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). 
#Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). 
p                   =     2500
imdb                =     dataset_imdb(num_words = p, skip_top = 00) #, skip_top = 10
train_data          =     imdb$train$x
train_labels        =     imdb$train$y
test_data           =     imdb$test$x
test_labels         =     imdb$test$y

numberWords.train   =   max(sapply(train_data, max))
numberWords.test    =   max(sapply(test_data, max))

#c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

#The variables train_data and test_data are lists of reviews; each review is a list of
#word indices (encoding a sequence of words). train_labels and test_labels are
#lists of 0s and 1s, where 0 stands for negative and 1 stands for positive:
str(train_data[[1]])

word_index                   =     dataset_imdb_word_index() #word_index is a named list mapping words to an integer index
reverse_word_index           =     names(word_index) # Reverses it, mapping integer indices to words
names(reverse_word_index)    =     word_index

review_index                 =     17
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}) 
# Decodes the review. 
# Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices 
# for “padding,” “start of sequence,” and “unknown.”

#cat(decoded_review)

vectorize_sequences <- function(sequences, dimension = p) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

X.train          =        vectorize_sequences(train_data)
X.test           =        vectorize_sequences(test_data)

#str(X.train[1,])
y.train          =        as.numeric(train_labels)
n.train          =        length(y.train)
y.test           =        as.numeric(test_labels)
n.test           =        length(y.test)


#fit                     =        glm(y.train ~ X.train, family = "binomial")
library(glmnet) 
fit                     =        glmnet(X.train, y.train, family = "binomial", lambda=0.0)
sort(prob.train,decreasing = TRUE)
beta0.hat               =        fit$a0
beta.hat                =        as.vector(fit$beta)

thrs                    =        0.5

prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))

#find closet to 0.5
index_fit=which(abs(prob.train-0.5)==min(abs(prob.train-0.5)))
X.train[index_fit, ]
review_index                 =    index_fit
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
paste( unlist(decoded_review), collapse=' ')

#find closet to 1
index_fit_positive=which(abs(prob.train-1.0)==min(abs(prob.train-1.0)))
X.train[index_fit_positive, ]
review_index                 =    index_fit_positive
decoded_review_positive <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
paste( unlist(decoded_review_positive), collapse=' ')

#find closet to 0
index_fit_negative=which(abs(prob.train-0)==min(abs(prob.train-0)))
X.train[index_fit_negative, ]
review_index                 =    index_fit_negative
decoded_review_negative <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
paste( unlist(decoded_review_negative), collapse=' ')

#overlapping histgram
library(ggplot2)
train.num=X.train %*% beta.hat +  beta0.hat
dataf.train = data.frame(y.train, train.num)
ggplot(dataf.train, aes(length, fill = veg)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..),position = 'identity')


dat <- data.frame(xx=train.num,yy = y.train)

ggplot(dat,aes(x=xx)) + 
  geom_histogram(data=subset(dat,yy == '1'),fill = "red", bins=80,alpha = 0.2) +
  geom_histogram(data=subset(dat,yy == '0'),fill = "blue" , bins=80,alpha = 0.2)+ labs(x="logit",y="classification",title = "Histogram of logit based on positive/negative reviews")

#y2=X.train[prob.train==0.5]
y.hat.train             =        ifelse(prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
P.train                 =        sum(y.train==1) # total positives in the data
N.train                 =        sum(y.train==0) # total negatives in the data
FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
typeI.err.train         =        FPR.train
typeII.err.train        =        1 - TPR.train

print(paste( "train: err        = ", sprintf("%.2f" , mean(y.train != y.hat.train))))
print(paste( "train: typeI.err  = ", sprintf("%.2f" , typeI.err.train)))
print(paste( "train: typeII.err = ", sprintf("%.2f" , typeII.err.train)))

prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
y.hat.test              =        ifelse(prob.test > thrs,1,0) #table(y.hat.test, y.test)  
FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
P.test                  =        sum(y.test==1) # total positives in the data
N.test                  =        sum(y.test==0) # total negatives in the data
TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
typeI.err.test          =        FPR.test
typeII.err.test         =        1 - TPR.test
print(paste("--------------- --------------------"))
print(paste( "test: err        = ", sprintf("%.2f" , mean(y.test != y.hat.test))))
print(paste( "test: typeI.err  = ", sprintf("%.2f" , typeI.err.test)))
print(paste( "test: typeII.err = ", sprintf("%.2f" , typeII.err.test)))


#find top 10 words related to positive/negative reviews
library(dplyr)
obh                    =       order(beta.hat) 
mw                     =       10
word.index.negatives   =       obh[1:mw]
word.index.positives   =       obh[(p-(mw-1)):p]


negative.Words         =       sapply(word.index.negatives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(negative.Words)


positive.Words         =       sapply(word.index.positives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(positive.Words)

print(paste("word most associated with negative reviews = ", reverse_word_index[[as.character((which.min(beta.hat)-3))]]))
print(paste("word most associated with positive reviews = ", reverse_word_index[[as.character((which.max(beta.hat)-3))]]))

#calculate TPR,FPR based on different thresholds
thrs_test <- seq(from=0,to=1,by=0.01)
n=length(thrs_test)
y.hat.test3 <- rep(0,n)
y.hat.train3 <- rep(0,n)
tpr1 <- fpr1 <- rep(0,n)
tpr2 <- fpr2 <- rep(0,n)
for (i in 1:n){
  thrs_temp               =        thrs_test[i]
  y.hat.train2            =        ifelse(prob.train > thrs_temp, 1, 0) 
  FP.train                =        sum(y.train[y.hat.train2==1] == 0) 
  TP.train                =        sum(y.hat.train2[y.train==1] == 1)
  P.train                 =        sum(y.train==1) 
  N.train                 =        sum(y.train==0) 
  FPR.train               =        FP.train/N.train 
  TPR.train               =        TP.train/P.train 
  typeI.err.train         =        FPR.train
  typeII.err.train        =        1 - TPR.train
  prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  y.hat.test2             =        ifelse(prob.test > thrs_temp,1,0) 
  FP.test                 =        sum(y.test[y.hat.test2==1] == 0) 
  TP.test                 =        sum(y.hat.test2[y.test==1] == 1) 
  P.test                  =        sum(y.test==1) 
  N.test                  =        sum(y.test==0) 
  TN.test                 =        sum(y.hat.test2[y.test==0] == 0)
  FPR.test                =        FP.test/N.test 
  TPR.test                =        TP.test/P.test 
  typeI.err.test          =        FPR.test
  typeII.err.test         =        1 - TPR.test
  tpr1[i] <- TPR.train
  fpr1[i] <- FPR.train
  tpr2[i] <- TPR.test
  fpr2[i] <- FPR.test
  y.hat.train3[i] <- y.hat.train2
  y.hat.test3[i] <- y.hat.test2
}


#calculate auc for training set/test set
fpr_diff_1 <-  fpr_diff_2 <- rep(0,100)
tpr_diff_1 <- tpr_diff_2 <- rep(0,100)
for (i in 1:100)
{ fpr_diff_1[i] = fpr1[101-i]-fpr1[101-i+1]
tpr_diff_1[i] = (tpr1[101-i]+tpr1[101-i+1])/2
fpr_diff_2[i] = fpr2[101-i]-fpr2[101-i+1]
tpr_diff_2[i] = (tpr2[101-i]+tpr2[101-i+1])/2
}
# install.packages("DescTools")
# library(DescTools)
# AUC(x=fpr2,y=tpr2)
auc_train <- sum(fpr_diff_1*tpr_diff_1)
auc_test <- sum(fpr_diff_2*tpr_diff_2)

library("pROC")  
roc1 <- roc(controls=fpr1,cases=tpr1)   
roc2 <- roc(controls=fpr2,cases=tpr2)  
plot(roc1, col="blue")  
plot.roc(roc2, add=TRUE, col="red")

library(ROCit)
class <- tpr1

score <- fpr1
class2 <- tpr2
score2 <- fpr2
rocit_bin <- rocit(score = score, 
                   class = class, 
                   method = "bin")
rocit_bin2 <- rocit(score = score2, 
                    class = class2, 
                    method = "bin")

summary(rocit_bin)
summary(rocit_bin2)
plot(rocit_bin, col = c(2,"gray50"), 
     legend = FALSE, YIndex = FALSE,values = TRUE)
lines(rocit_bin2$TPR~rocit_bin2$FPR, 
      col = 4, lwd = 2)
legend("bottomright", col = c(2,4),
       c("training ROC",
         "test ROC"),lwd = 2)
legend("topright",col = c(2,4),legend=c(rocit_bin$AUC,rocit_bin2$AUC))



plot(fpr1,tpr1,type='l',xlab="false positive ratio",ylab="true positive ratio",col=2,main="ROC Curve")
lines(fpr2,tpr2, type='l',
      col = 4, lwd = 2)
legend("bottomright", col = c(2,4),
       c("training ROC",
         "test ROC"),lwd = 2)
legend("topright", col = c(2,4),
       legend=c(auc_train,auc_test),lwd = 2)
