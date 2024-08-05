import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_style("darkgrid")

data = pd.read_csv("starbucks.csv",index_col=0)

print(data.head(5))
print(data.describe)
print(data.columns)
print(data.info)
print(data["item"].nunique()) #77 item
print(data["item"].unique()) # item name
print(data["type"].unique())
print(data.groupby("type")["item"].count())

data.groupby("type")["item"].count().plot()
plt.title("ürün türleri")
plt.text(0.,50,"ürün katagori yelpazasi",bbox =dict(facecolor = "yellow",alpha = 0.5))
plt.show()

sns.countplot(x = "type",data = data,palette="Set2")
plt.title("ürün sayıları")
plt.show()

sns.catplot(kind="bar",x="type",y="calories",data = data)
plt.title("kalori değerleri")
plt.show()


sns.catplot(kind="bar",x="type",y="protein",data = data)
plt.title("protein değerleri")
plt.show()

sns.catplot(kind="bar",x="type",y="carb",data = data)
plt.title("carb değerleri")
plt.show()

sns.catplot(kind="bar",x="type",y="fiber",data = data)
plt.title("fiber değerleri")
plt.show()


"""sns.heatmap(data.corr(),annot=True)"""


"""plt.title("kalori", "yağ")
sns.scatterplot(x = "calories",y = "fat",data = data,s = 30,edgecolor = "red")
plt.show()"""

sns.displot(x = "calories",data = data,color = "red",kde = True)
plt.title("kalori grafiği")
plt.show()


sns.displot(x = "protein",data = data,color = "blue",kde = True)
plt.title("protein grafiği")
plt.show()

sns.displot(x = "fat",data = data,color = "pink",kde = True)
plt.title("fat grafiği")
plt.show()


sns.displot(x = "carb",data = data,color = "green",kde = True)
plt.title("carb grafiği")
plt.show()


data.head()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


x = data[["calories","fat","carb","fiber","protein"]]
y = data["type"]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print("doğruluk değeri",accuracy)


prediction = model.predict([[250,2,40,17,19]])

print("tahmin",prediction)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(15,10))
plot_tree(model, feature_names=x.columns, filled=True)  # 'class_names' yerine sadece 'filled' ekledik.
plt.show()


