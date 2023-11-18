from main4 import Classification
import warnings
warnings.filterwarnings("ignore")
# clss=["rf","dt","svm","mlp","ab","lr"]
# imp=["mean","median","mode","knn"]
# for i in clss:
#     for j in imp:
classifier = Classification(clf_opt="rf", impute_opt="knn", feature_selc="corr")
classifier.classification()
    