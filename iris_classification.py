# 利用决策树对鸢尾花进行分类
import pandas as pd
import numpy as np
# 导入鸢尾花数据集
from sklearn.datasets import load_iris
# 导入用于分类的决策树
from sklearn.tree import DecisionTreeClassifier
# 主要是用于帮助我们生成一些绘图的数据
from sklearn.tree import export_graphviz
# 用于切分测试集和训练集的工具
from sklearn.model_selection import train_test_split
# 用于评估准确率
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

# 导入数据
iris_obj = load_iris()
X_data = pd.DataFrame(iris_obj.data)
y = iris_obj.target
X_data.columns = iris_obj.feature_names
X_data['Species'] = y

# 花瓣的长度以及宽度
# 根据索引号来定位我们要取的数据 第一个：表示取所有数据 取索引号为2和3的2列数据
X = X_data.iloc[:, 2:4]
# 取所有行的最后一列
y = X_data.iloc[:, -1]

# 切分训练集以及测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

# 获取决策树模型
# criterion 分裂指标用基尼系数
tree_obj = DecisionTreeClassifier(max_depth=8, criterion='gini')
# 训练模型
tree_obj.fit(x_train, y_train)
# 获取预测值
y_preidict = tree_obj.predict(x_test)
print("准确率：", accuracy_score(y_test, y_preidict))
print("模型参数：", tree_obj.feature_importances_)

# 绘制决策树
export_graphviz(
    tree_obj,
    out_file="./iris_classification_tree.dot",
    feature_names=X_data.columns[2:4],
    class_names=iris_obj.target_names,
    rounded=True,
    filled=True
)
#
tree_depth = np.arange(1, 15)
# 存放错误率
error_list = []
# 存放最优树深度
tree_best_depth_list = []
for single_depth in tree_depth:
    tree_clf_obj = DecisionTreeClassifier(max_depth=single_depth, criterion='gini')
    tree_clf_obj.fit(x_train, y_train)
    y_tests_predict = tree_clf_obj.predict(x_test)
    predict_result_status = y_tests_predict == y_test
    # 求平均
    errs = 1 - np.mean(predict_result_status)
    error_list.append(errs)
    if errs <= 0.02:
        tree_best_depth_list.append(single_depth)
        print(single_depth, '错误率: %.2f%%' % (100 * errs), 'best')
        # break
    print(single_depth, '错误率: %.2f%%' % (100 * errs))

print(tree_best_depth_list)
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(facecolor='w')
plt.plot(tree_depth, error_list, 'ro-', lw=2)
plt.xlabel('tree_depth', fontsize=15)
plt.ylabel('error', fontsize=15)
plt.title('tree_depth and error', fontsize=20)
plt.grid(True)
plt.show()




