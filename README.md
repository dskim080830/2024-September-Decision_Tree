# 의사 결정 나무 과제
```
1. 필요한 라이브러리 가져오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('drug200.csv')
2. 데이터 확인
print(df.head())
3. 레이블 설정
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]  
y = df['Drug'] 

X = pd.get_dummies(X, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

4. 데이터를 학습 데이터, 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

5. 의사결정나무 모델 학습
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

6. 의사 결정 나무 시각화
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()

7. 모델 정확도 계산
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```
