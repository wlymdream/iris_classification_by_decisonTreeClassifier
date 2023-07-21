from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

M = 100
x_data = np.random.rand(M) * 6 - 2
x_data.sort()
# print(x)
y = np.sin(x_data) + np.random.randn(M) * 0.05
# print(y)

x = x_data.reshape(-1, 1)

decision_tree_obj = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4)
decision_tree_obj.fit(x, y)

x_test = np.linspace(-4, 4, 100).reshape(-1, 1)
y_predict = decision_tree_obj.predict(x_test)

plt.plot(x, y, 'y*')
# plt.plot(x, y_predict, 'r-')



max_depths = [2,4,6,8]
colors = 'rgbmy'
for single_depth, color in zip(max_depths, colors):
    dt_reg_obj = DecisionTreeRegressor(max_depth=single_depth)
    dt_reg_obj.fit(x, y)
    predict = dt_reg_obj.predict(x_test)
    plt.plot(x_test, predict, '-', color=color, linewidth=2, label='depth=%d' % single_depth)

plt.legend(loc='upper right')
plt.show()
plt.savefig('./dt_reg_result')