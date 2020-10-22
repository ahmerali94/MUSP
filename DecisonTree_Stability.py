import pandas as pd
import graphviz 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_text 



data = pd.read_csv('C:/Users/Ahmar/OneDrive/Work/MUSP/Book1.csv') #Excel file containing the data
display(data)
feature_cols  = ['n [rpm]', 'vf [mm/min]']  #featues columns in data
target_cols = ['Stability condition']#target columns
X = data[feature_cols]              # Features
y = data[target_cols]          #target label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) #split traing and testing data randomly taking 2 test samples 


display(X_train, y_train, X_test, y_test)


clf = DecisionTreeClassifier()  #decision tree classifier using GINI inpurity index
clf = clf.fit(X_train,y_train)  #fiting according to X training and y Training
y_pred = clf.predict(X_test)    #classifying according to X testing data 

display("Accuracy:",metrics.accuracy_score(y_test, y_pred))   # comparing results from y predicted and y test

clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3)  #decision tree classifier using entropy
clf_entropy = clf_entropy.fit(X_train,y_train)  #fiting according to X training and y Training

display("Accuracy_Entropy:",metrics.accuracy_score(y_test, y_pred))   # comparing results from y predicted and y test


 
dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['Chatter','Stable']) 
graph = graphviz.Source(dot_data) 
graph.render("Stability")  #saving tree in Stability PDF
display(graph)

dot_data = tree.export_graphviz(clf_entropy, out_file=None, filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['Chatter','Stable']) 
graph = graphviz.Source(dot_data) 
graph.render("Stability_entopy")  #saving tree in Stability PDF
display(graph)



rules = export_text(clf, feature_names = feature_cols)  #exporting rules estbilished in text form
print(rules)

rules_entropy = export_text(clf_entropy, feature_names = feature_cols)  #exporting rules estbilished in text form
print(rules_entropy)
