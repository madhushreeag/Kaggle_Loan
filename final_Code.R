### load train and Test Data
train = read.csv("train.csv",header=TRUE)
test = read.csv("test.csv",header=TRUE)

### Preprocessing steps
#removing unique values and sd=0 values
data5 = train
uniq = sapply(data5,function(x) length(unique(x)))
del1 = uniq[uniq>104400]
data6=data5[,-which(rownames(data.matrix(del1)) %in% colnames(data5))]

const_cols = which(apply(train10.1,2,sd,na.rm=TRUE)==0)
const_cols.df = data.frame(const_cols)

data6=subset(data6,-const_cols.df)

# making target variable fraud
y=data6[,735]
fraud = y
fraud[fraud!=0]=1

# imputing train data
data6_notDup = data6
imp_train = impute(data6_notDup,what="mean")

# removing high corr variables (0.8)
corr_train = cor(imp_train)
highCorr_train =findCorrelation(corr_train,cutoff = 0.8)
train_pre1 = imp_train[,-highCorr_train]
train_pre1 = data.frame(train_pre1)
train_pre2 = train_pre1[,-204]

train_pre2 = cbind(train_pre2,fraud)

# standarising the train data
for(i in 1:(ncol(train_pre2)-1))
{
  mean_tr = mean(train_pre2[,i])
  std_tr = sd(train_pre2[,i])
  train_pre2[,i] = (train_pre2[,i]-mean_tr)/std_tr
}

# making train data set for training the classification model
set1 = train_pre2[sample(nrow(train_pre2),size=10547,replace=FALSE),]
set1 = data.frame(set1)

# feature selection for classification model
model1 = randomForest(fraud ~., data = set1,importance=TRUE)
f_imp<-importance(model1,class=1,type=1)

ft_select<-f_imp[,1]
ft_select<-ft_select[order(-ft_select)]
ft_select<-cbind(rownames(ft_select),ft_select)
ft_select1<-rownames(ft_select)
fts<-ft_select1[1:20]

fts20.crf =c("f636",  "f394",  "f430",  "f338",	"f393",	"f367",	"f287",	"f587",
             "f724",	"f358",	"f375",	"f588",	"f471",	"f451",	"f649",	"f536",
             "f333",	"f448",	"f420",	"f129")
fts20.rf = c("f670",  "f377",  "f68",	"f273",	"f213",	"f677",	"f675",	"f333",	
             "f391",	"f726",	"f323",	"f588",	"f653",	"f139",	"f413",	"f32",
             "f73",	"f63",	"f289",	"f671")  
# golden features for train
gold.tr = imp.std.tr[,c("f271")]
diff.tr =imp.std.tr[,c("f528")]-imp.std.tr[,c("f527")]
diff2.tr =imp.std.tr[,c("f528")]-imp.std.tr[,c("f274")]
golden.tr = cbind(f271=gold.tr,diff=diff.tr,diff2 = diff2.tr)
golden.tr = data.frame(golden.tr)


# golden features for test
gold.ts = imp.std.ts[,c("f271")]
diff.ts =imp.std.ts[,c("f528")]-imp.std.ts[,c("f527")]
diff2.ts =imp.std.ts[,c("f528")]-imp.std.ts[,c("f274")]
golden.ts = cbind(f271=gold.ts,diff=diff.ts,diff2 = diff2.ts)
golden.ts = data.frame(golden.ts)

################### end pre-processing #######################3


# all imputed and standarised train
imp.std.tr = impute(train,what="mean")
imp.std.tr = data.matrix(imp.std.tr)
imp.std.tr = data.frame(imp.std.tr)

# all imputed and standarised test
imp.std.ts = impute(test,what="mean")
imp.std.ts = data.matrix(imp.std.ts)
imp.std.ts = data.frame(imp.std.ts)

# standarisation
for(i in 2:ncol(imp.std.ts))
{
  mn = mean(imp.std.tr[,i])
  std = sd(imp.std.tr[,i])
  imp.std.tr[,i] = (imp.std.tr[,i]-mn)/std
  imp.std.ts[,i] = (imp.std.ts[,i]-mn)/std
}

############################# Train Modelling-RF####################################

################################### classification model
# features for classification
class.train = cbind(imp.std.tr[,fts20.crf],golden.tr,fraud = as.factor(fraud))

# divide into train and val set
set.seed(10)
train_ind = sample(seq_len(nrow(class.train)),size=10547*7,replace=TRUE)
train_main = class.train[train_ind,]
train_val = class.train[-train_ind,]

# run rf model on train set
train.rf.c = randomForest(fraud ~., data = train_main,ntree=10)

# test on val set
val.pred.rf.c = predict(train.rf.c,train_val)

# predict model on train
train.pred.c = predict(train.rf.c,class.train)

# classification results
table(train_val$fraud)
summary(val.pred.rf.c)

table(class.train$fraud)
summary(train.pred.c)

######################################## Regression model
# if separate features for regrerssion
reg.train = cbind(imp.std.tr[,fts20.rf],golden.tr,loss = train$loss,fraud=train.pred.c)

# select only fraud==1
frauds.tr = subset(reg.train,reg.train[,25]==1)
frauds.tr = frauds.tr[,-25]

# divide data in train set and val set
train_ind.r = sample(seq_len(nrow(frauds.tr)),size=nrow(frauds.tr)*.7,replace=FALSE)
train_main.r = frauds.tr[train_ind.r,]

train_val.r = frauds.tr[-train_ind.r,]

# run rf on train set
train.rf.r = randomForest(loss ~., data = train_main.r,ntree=10)

# predict on val set
val.pred.rf.r = predict(train.rf.r,train_val.r,n.trees=1)

# predict on whole train model
train.pred.r = predict(train.rf.r,reg.train,n.trees=1)
sim = data.frame(val.pred.rf.r)
mae.rf.tr = mae(sim,train_val.r[,24])
pred<- prediction(val.pred.rf.c, train_val[,24])
auc <- attr(performance(pred, "auc"), "y.values")[[1]]

############################# Test Prediction ###################################

################################ Classification Prediction
class.test = cbind(imp.std.ts[,fts20.crf],golden.ts)
# class.test.pred = prediction
test.pred.c = predict(train.rf.c,class.test)
summary(test.pred.c)

######################################## Regression Model
reg.test = cbind(imp.std.ts[,fts20.rf],golden.ts)
reg.test = cbind(reg.test,fraud = test.pred.c)

#getting loss data for frauds
frauds.test = cbind(id=test$id,reg.test)
frauds = subset(frauds.test,frauds.test[,25]==1)
loss.ts = frauds
loss.ts = loss.ts[,-25]


nofrauds = subset(frauds.test,frauds.test[,25]==0)
noloss = nofrauds
noloss$loss = rep(0,nrow(nofrauds))
noloss = noloss[,-25]

frauds = frauds[,-c(1,25)]
#### prediction
test.pred.reg = predict(train.rf.r,frauds,n.trees=1)

##### making submission file For Kaggle

loss.ts = cbind(loss.ts,loss=test.pred.reg)

submission = rbind(loss.ts,noloss)
submission = submission[order(submission$id),]
submission_rf = submission[,c(1,25)]
write.csv(submission_rf,"Submission.csv")
