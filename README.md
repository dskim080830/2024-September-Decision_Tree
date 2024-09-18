# 의사 결정 나무 과제
```
1. 필요한 라이브러리 가져오기
import pandas as pd
// 의사 결정 나무 알고리즘 시각화 및 정확도 분석을 위해 sklearn에 있는 라이브러리를 불러옵니다.
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
// 시각화를 위해 matplotlib 라이브러리를 불러옵니다. 
import matplotlib.pyplot as plt
// colab에서 컴퓨터에 있는 파일을 불러옵니다.
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('drug200.csv')
2. csv 파일에 있는 데이터를 불러와 출력해봅니다.
print(df.head())
3. 의사결정나무를 구현하기 위해 레이블을 설정합니다.
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]  
y = df['Drug'] 

X = pd.get_dummies(X, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

4. 정확도 분석을 위해서 데이터를 학습 데이터, 테스트 데이터로 분할하여 학습할 때와 테스트할 때의 데이터를 구분합니다. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

5. 여러 가지 분류 기준 중 엔트로피라는 방식을 사용하여 한 번에 데이터를 42개씩 불러오도록 하였습니다.
여기서 엔트로피란, 머신러닝에서 사용하는 데이터의 불확실성을 확인하는 것입니다. 엔트로피는 데이터에서 각 클래스가 발생할 확률을 계산합니다.
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

6. (20, 10)으로 사이즈를 설정한 후, 학습한 데이터를 바탕으로 의사결정나무 시각화를 합니다.
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()

7. 학습한 데이터를 사용하여 테스트를 해보고 그것을 통해 정확도를 출력합니다.
accuracy = clf.score(X_test, y_test) // 정확도 분석
print(f"Test Accuracy: {accuracy:.2f}") // 정확도 출력
```
