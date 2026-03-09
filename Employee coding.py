import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

employee_data = pd.read_csv("C:\\Users\\MUTHARASI\\Downloads\\Extended_Employee_Performance_and_Productivity_Data.csv.zip")
employee_data

employee_data.shape

employee_data.isnull().sum()

filter_employee_data = employee_data[["Employee_ID","Department","Gender","Age","Job_Title","Years_At_Company","Monthly_Salary","Performance_Score"]]

filter_employee_data

import seaborn as sns
sns.boxplot(x=filter_employee_data["Monthly_Salary"])

from matplotlib import pyplot as plt
filter_employee_data['Job_Title'].value_counts().head(10).plot(kind='bar')
plt.xlabel("Job_Title")
plt.ylabel("Performance_Score")
plt.title("Department VS Highest Performance")
plt.show()

X = filter_employee_data[["Years_At_Company"]]
y = filter_employee_data["Monthly_Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
print(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))