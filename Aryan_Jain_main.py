from Aryan_Jain_Classification import Classification
import warnings
warnings.filterwarnings("ignore")
# clss=["rf","dt","svm","ab","lr","lsv"]
# imp=["mean","median","mode","knn"]
# for i in clss:
#     for j in imp:
#         print(i,"for",j) 
classifier = Classification(clf_opt="ab", impute_opt="knn", feature_selc="mi")
classifier.classification()
           
