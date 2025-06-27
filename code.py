import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('retail_store.csv')   # item , ppu , quantity , total spent and discount applied has missing value 
df.info()

df1 = df.copy()

# Display unique value counts and unique values for each column
for col in df.columns:
    unique_values = df[col].dropna().unique()  
    unique_count = df[col].nunique() 
    
    print("Column:",col)
    print("Unique Value Count:",unique_count)
    print("Unique Values:",unique_values)

print("\nMode of Item : ")
item_mode = st.mode(df1.loc[~df1["Item"].isna() , "Item"])
print(item_mode)
df1.loc[df1["Item"].isna() , "Item"] = item_mode

print("\nMean of Item : " )
price_per_unit_mean = np.mean(df1.loc[~df1["Price Per Unit"].isna() , "Price Per Unit"])
print(price_per_unit_mean)
df1.loc[df1["Price Per Unit"].isna() , "Price Per Unit"] = price_per_unit_mean

print("\nMode of Quantity : ")
quantity_mode = st.mode(df1.loc[~df1["Quantity"].isna() , "Quantity"])
print(quantity_mode)
df1.loc[df1["Quantity"].isna() , "Quantity"] = quantity_mode

print("\nMean of Total spent : ")
total_spent_mean = np.mean(df1.loc[~df1["Total Spent"].isna() , "Total Spent"])
print(total_spent_mean)
df1.loc[df1["Total Spent"].isna() , "Total Spent"] = total_spent_mean

print("\nMode of Customer ID : ")
Customer_mode = st.mode(df1.loc[~df1["Customer ID"].isna() , "Customer ID"])
print(Customer_mode)
df1.loc[df1["Customer ID"].isna() ,"Customer ID"] = Customer_mode


plt.hist(df1["Price Per Unit"], bins=10, color='red', edgecolor='black')
plt.title("Price Per Unit Distribution")
plt.xlabel("Price Per Unit")
plt.ylabel("Frequency")
plt.show()

plt.hist(df1["Quantity"], bins=10, color='green', edgecolor='black')
plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

plt.hist(df1["Total Spent"], bins=10, color='blue', edgecolor='black')
plt.title("Total Spent Distribution")
plt.xlabel("Total Spent")
plt.ylabel("Frequency")
plt.show()


df2 = df1.copy()
df2 = df2.dropna(axis=0)
df2.info()
# describe the statistical analysis on the selected dataset
# Select only numerical columns
df3= df2.select_dtypes(include='number')

# Perform statistical analysis
stats = df3.describe()
print(stats)

df2["Discount Applied"] = df2["Discount Applied"].notna().astype(int)

X = df2[["Price Per Unit", "Quantity", "Total Spent"]]
y = df2["Discount Applied"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

k=int(input("Enter the value of k:"))
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
clas=classification_report(y_test,y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n",clas)