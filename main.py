import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

df = pd.read_csv('C:\\Users\\matth\\Desktop\\creditcard.csv', encoding='latin-1')

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

pipeline = Pipeline([
    ('scale', StandardScaler()),
     ("model", LogisticRegression(class_weight='balanced'))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test,predictions))