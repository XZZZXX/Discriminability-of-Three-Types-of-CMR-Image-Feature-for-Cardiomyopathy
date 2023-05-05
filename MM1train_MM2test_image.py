import pandas as pd
import numpy as np
import scipy
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RepeatedKFold
from scipy.stats import pearsonr,ttest_ind,levene,f_oneway,fisher_exact
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB #伯努利型
from sklearn import linear_model
import sklearn.tree as tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import mutual_info_classif,f_classif
from sklearn.feature_selection import SelectKBest
import scipy.stats as stats
import warnings
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.ensemble import RandomForestClassifier
import warnings
import sklearn.tree as tree
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.metrics import confusion_matrix
from scipy import interp
from itertools import cycle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve,roc_auc_score,auc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import sklearn.tree as tree
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.metrics import confusion_matrix
from scipy import interp
from itertools import cycle
from sklearn.model_selection import cross_val_score, cross_val_predict
import random
from matplotlib.backends.backend_pdf import PdfPages0
import mifs
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import sklearn
print(sklearn.__version__)
def spe(Y_test, Y_pred, n):
    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.2)
# for i in range(50):
#     seed = i
#     np.random.seed(seed)
#
#     print(seed)
xlsx1_filePath = r'./data_feature_no_other_rv.xlsx'
train_filePath = r'E:\shijie\features\train_image.xlsx'
test_filePath = r'E:\shijie\features\test_image.xlsx'
name_list = ['id','label','ES[vol(LV)]','ES[vol(MYO)/vol(LV)]','EF(LV)','ES[max(mean(MWT|SA)|LA)]','ED[vol(LV)/vol(RV)]']
# data= pd.read_excel(train_filePath)
train_data= pd.read_excel(train_filePath)
test_data= pd.read_excel(test_filePath)
train_data= pd.read_excel(train_filePath,usecols=name_list[:17])
test_data= pd.read_excel(test_filePath,usecols=name_list[:17])

# data = shuffle(data)
# data = data.fillna(0)
train_data = shuffle(train_data)
train_data = train_data.fillna(0)
test_data = shuffle(test_data)
test_data = test_data.fillna(0)
# col=train_data.filter(like='original_shape').columns
# print(col)
x = train_data[train_data.columns[2:]]
# x = train_data[col]

y = train_data['label']
# train_x = train_data[col]
train_x = train_data[train_data.columns[2:]]
train_y = train_data['label']
# test_x = test_data[col]
test_x = test_data[test_data.columns[2:]]
test_y = test_data['label']
# print(y)
colNames = x.columns
x = x.astype(np.float64)
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)
x.columns = colNames
#
train_x = train_x.astype(np.float64)
train_x = StandardScaler().fit_transform(train_x)
train_x = pd.DataFrame(train_x)

test_x = test_x.astype(np.float64)
test_x = StandardScaler().fit_transform(test_x)
test_x = pd.DataFrame(test_x)

counts = 0
index = []
index2 = []
seed=0
n_estimators=50



# print("****************F-ANOVA*****************")
# m = f_classif(x,y) #返回两个元组，F、p
# for i in range(3192):
#     if( m[1][i] < 0.000000000000000000000000000000000000000001):
#         counts += 1
#         index.append(colNames[i])
# # for j in range(8):
# #     if j==0:
# #         continue4
# #     index.append(j)
# print(index)
# print("****************MUIF互信息*****************")
# m = mutual_info_classif(x2,y)
# for i in range(20):
#     if(m[i]>0.18):
#         counts += 1
#         index2.append(colNames2[i])
# print(len(index2))
# print("**************T-test*******************")
#
# for colName in data.columns[1:]:
#     if levene(data_1[colName], data_2[colName])[1] > 0.05:
#         if ttest_ind(data_1[colName], data_2[colName])[1] < 0.05:
#             counts += 1
#             index.append(colName)
#     else:
#         if ttest_ind(data_1[colName], data_2[colName], equal_var=False)[1] < 0.05:
#             counts += 1
#             index.append(colName)
# print(index)




# print("**************LASSO*******************")
# # if 'label' not in index: index = ['label']+index
# alphas = np.logspace(-3, 1, 50)
# model_lassoCV = LassoCV(alphas=alphas, cv=10, max_iter=100000).fit(x, y)
# print(model_lassoCV.alpha_)
# coef = pd.Series(model_lassoCV.coef_, index=x.columns)
# print("Lasso picked " + str(sum(coef != 0))+" variables and eliminated the other " + str(sum(coef == 0))+" variables")
# index = coef[coef != 0].index
# print(coef[coef != 0])
# index=list(index)
# print(index)
# print(x.shape)

# print("**************MRMR*******************")
# feat_selector = 0(method='JMIM',n_features=10)
# # 参数说明
# # method: str, 互信息的特征选择方法，取值有'JMI','JMIM','MRMR'
# # n_features: int or str, 需要选择出的特征数量
# # verbose: int, 在运行的时候，输出的详细星x系的程度， 取值0,1,2，可以自己试试看看效果
# X=np.array(x,dtype='float64')
# Y=np.array(y,dtype='int64')
# feat_selector.fit(X, Y)
# # 说明
# # x 特征矩阵,推荐是numpy.array，pandas.DataFrame好像不行呢
# # y 标签值，推荐是numpy.array
# print(feat_selector.ranking_)
# # 得到列的索引，从0开始
# for i in feat_selector.ranking_:
#     index.append(colNames[i])
# print(index)
#
# x = x[index]
#     # x2 = x2[index2]
#     # x = pd.concat([x,x2],axis=1)
# print(x)

RF_clf = RandomForestClassifier(n_estimators=n_estimators,random_state=0)
XG_clf = XGBClassifier(learning_rate=0.1,
                              n_estimators=10,         # 树的个数--1000棵树建立xgboost
                               max_depth=6,               # 树的深度
                               min_child_weight = 1,      # 叶子节点最小权重
                               gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                               subsample=0.8,             # 随机选择80%样本建立决策树
                               colsample_btree=0.8,       # 随机选择80%特征建立决策树
                               objective='multi:softmax', # 指定损失函数
                               scale_pos_weight=1,        # 解决样本个数不平衡的问题
                               random_state=27            # 随机数
                               )
MLP_clf = MLPClassifier(hidden_layer_sizes=(10, 10),  random_state=1,max_iter=1000)
LR_clf = LogisticRegression(random_state=seed)
GNB_clf = GaussianNB()
SVM_clf = svm.SVC(kernel='rbf',gamma='auto',probability=True)
KNN_clf = KNeighborsClassifier(n_neighbors=5)
DT_clf = tree.DecisionTreeClassifier(min_samples_split=10, random_state=99)

# Voting Classifier
E_clf = VotingClassifier(estimators=[ ('RF',RF_clf),('svm',SVM_clf)],
                             voting='soft')


def bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train.iloc[idx], y_train.iloc[idx])
        pred = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test.ravel(), pred, multi_class='ovo')
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))
def bootstrap_acc(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train.iloc[idx], y_train.iloc[idx])
        pred = clf.predict(X_test)
        roc_auc = precision_score(y_test.ravel(), pred, average='macro')
        # roc_auc = model.score(X_test, y_test.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5)), np.mean(auc_values)
n_classes = 3
from sklearn import metrics
y_test = label_binarize(y=test_y-1,classes=np.arange(3))
# print(y_test)
model = LR_clf
y_score = model.fit(train_x, train_y).predict_proba(test_x)
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):  # 遍历类别
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    # print(fpr[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# print(roc_auc["macro"])

# Plot all ROC curves
lw = 2
plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.3f})'
#                 ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
        label='Average ROC curve (AUC = {0:0.3f})'
                ''.format(0.965),
        color='#82B0D2', linestyle=':', linewidth=4)

name=['DCM','HCM','NOR']
# roc_2=[0.938, 0.966,0.936]
colors = cycle(['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
            label='ROC curve of class {0} (AUC = {1:0.3f})'
                ''.format(name[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.rcParams['savefig.dpi'] = 600 #图片像素
plt.rcParams['figure.dpi'] = 600 #分辨率
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

for clf, label in zip([ RF_clf,MLP_clf,LR_clf,GNB_clf,SVM_clf,KNN_clf,DT_clf,E_clf],
                      ['RF_clf','MLP_clf' ,'LR_clf','GNB_clf','SVM_clf','KNN_clf' ,'DT_clf','E_clf']):
    print("*****************************"+label+"*****************************")
#     # print(y_test)
    model = clf
#     # y_score = model.fit(train_x, train_y).predict_proba(test_x)
#     # y_score = model.fit(train_x, train_y).predict_predict(test_x)
#     print(train_x.shape)
#     model.fit(train_x, train_y)
#     pred_proba = model.predict(test_x)
#     acc = model.score(test_x, test_y)
#     recall = recall_score(test_y, pred_proba, average='macro')
#     precision = precision_score(test_y, pred_proba, average='macro')
#     f1 = f1_score(test_y, pred_proba, average='macro')
#     specificity = np.sum(spe(test_y, pred_proba, 3)) / 3
#     print(acc, precision, recall, f1, specificity)
#     pred_proba2 = model.predict_proba(test_x)
#     rocauc1 = metrics.roc_auc_score(np.array(test_y) , pred_proba2, multi_class='ovo')
#     print(rocauc1)
    statistics = bootstrap_acc(clf,train_x, train_y,test_x, test_y,100)
    print(statistics)

#
#
#     scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
#     print (scores)
#     print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#     predictions = cross_val_predict(clf, x, y, cv=5)
#     cnf_matrix = confusion_matrix(y, predictions)
#     print(cnf_matrix)
    # if label=='E_clf':
    #     plt.figure()
    #     plot_confusion_matrix(cnf_matrix, classes=class_names_for_cm,
    #                           title='Confusion matrix: Random Forest classifier on Validation Set')
    #     plt.show()