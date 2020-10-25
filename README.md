# IMDB-Reviews-Analysis
Binary classification for IMDB reviews using logistic regression
- Load training & test datasets
```
library(keras)  
library(tensorflow) 
p                   =     2500
imdb                =     dataset_imdb(num_words = p)
train_data          =     imdb$train$x
train_labels        =     imdb$train$y
test_data           =     imdb$test$x
test_labels         =     imdb$test$y

numberWords.train   =   max(sapply(train_data, max))
numberWords.test    =   max(sapply(test_data, max))
```

- The variables train_data and test_data are lists of reviews; each review is a list of word indices (encoding a sequence of words). train_labels and test_labels are lists of 0s and 1s, where 0 stands for negative and 1 stands for positive. We can try to decode review using the following codes.
```
word_index                   =     dataset_imdb_word_index() #word_index is a named list mapping words to an integer index
reverse_word_index           =     names(word_index) # Reverses it, mapping integer indices to words
names(reverse_word_index)    =     word_index

review_index                 =     17
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
```

- train the model using glmnet
```
# word vectorization and data preparation
vectorize_sequences <- function(sequences, dimension = p) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

X.train          =        vectorize_sequences(train_data)
X.test           =        vectorize_sequences(test_data)
y.train          =        as.numeric(train_labels)
n.train          =        length(y.train)
y.test           =        as.numeric(test_labels)
n.test           =        length(y.test)

#fit the model and calculate probability
library(glmnet) 
fit                     =        glmnet(X.train, y.train, family = "binomial", lambda=0.0)
sort(prob.train,decreasing = TRUE)
beta0.hat               =        fit$a0
beta.hat                =        as.vector(fit$beta)
thrs                    =        0.5
prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
```
- calculate training error and predict y for test set
```
#calculate TPR,FPR for training set
y.hat.train             =        ifelse(prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
P.train                 =        sum(y.train==1) # total positives in the data
N.train                 =        sum(y.train==0) # total negatives in the data
FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
typeI.err.train         =        FPR.train
typeII.err.train        =        1 - TPR.train
#print results
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
```
As we can see from the result, test error is relatively high. So the next steps would be testing different thresholds to improve test accuracy.

![testerror](https://user-images.githubusercontent.com/72762392/97096292-16b6f200-1638-11eb-8a8e-eb4c03a92791.JPG)

- exploring reviews
Using threshold as 0.5, we can find the most positive review, the most negative review and the review that is hardest to classify.
For the most positive review, We just need to find the index which has probability that is closest to 1(positive) and decode the review.
```
index_fit_positive=which(abs(prob.train-1.0)==min(abs(prob.train-1.0)))
X.train[index_fit_positive, ]
review_index                 =    index_fit_positive
decoded_review_positive <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
paste( unlist(decoded_review_positive), collapse=' ')
```
![positive1](https://user-images.githubusercontent.com/72762392/97096346-d5731200-1638-11eb-8f60-5dfdcb3c1bdb.png)

We can use the same method to find review with probability that is closet to 0(negative) and 0.5.
```
#find closet to 0
index_fit_negative=which(abs(prob.train-0)==min(abs(prob.train-0)))
X.train[index_fit_negative, ]
review_index                 =    index_fit_negative
decoded_review_negative <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
paste( unlist(decoded_review_negative), collapse=' ')

#find closet to 0.5
index_fit=which(abs(prob.train-0.5)==min(abs(prob.train-0.5)))
X.train[index_fit, ]
review_index                 =    index_fit
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
paste( unlist(decoded_review), collapse=' ')
```
Let's check these reviews.
![negative1](https://user-images.githubusercontent.com/72762392/97096373-3864a900-1639-11eb-8c4b-540191e415b7.png)
![neutral1](https://user-images.githubusercontent.com/72762392/97096375-41557a80-1639-11eb-9392-b679d10632df.png)

There's also another way to check the distribution of reviews.Let's do an overlapping histrogram.
```
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
  ```
![hist](https://user-images.githubusercontent.com/72762392/97096411-ea03da00-1639-11eb-8298-67215becd382.JPG)

- Testing different thresholds
```
#calculate TPR,FPR based on different thresholds
thrs_test <- seq(from=0,to=1,by=0.01)
n=length(thrs_test)
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

}
#plot ROC curve for training/test set
plot(fpr1,tpr1,type='l',xlab="false positive ratio",ylab="true positive ratio",col=2,main="ROC Curve")
lines(fpr2,tpr2, type='l',
      col = 4, lwd = 2)
legend("bottomright", col = c(2,4),
       c("training ROC",
         "test ROC"),lwd = 2)
legend("topright", col = c(2,4),
       legend=c(auc_train,auc_test),lwd = 2)
```
![roc curve](https://user-images.githubusercontent.com/72762392/97096436-449d3600-163a-11eb-9a4c-e84417fb405f.JPG)

The AUC of training set is : 0.9754
The AUC of test set is : 0.9311
