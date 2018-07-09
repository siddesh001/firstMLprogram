from sklearn import datasets
iris=datasets.load_iris()
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
#----------------------------------------------------------------------
#Code to create an object of decision tree and building a decision tree.
#----------------------------------------------------------------------
df1=pd.DataFrame(iris.data,columns=iris.feature_names)
y=iris.target
dtree=DecisionTreeClassifier()
dtree.fit(df1,y)

#----------------------------------------------------------------------
#Code for visualization of the decision tree in human readable format
#----------------------------------------------------------------------
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
