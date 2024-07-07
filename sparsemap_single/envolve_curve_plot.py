import matplotlib.pyplot as plt
import numpy as np
import matplotlib

print(matplotlib.get_data_path())

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 示例数据
x = np.arange(1, 31)  # 横轴：进化代数，范围是1到30
y = np.array([6.134842e+09, 5.524842e+09, 5.024842e+09, 4.524842e+09, 4.024842e+09,
              3.524842e+09, 3.024842e+09, 2.524842e+09, 2.024842e+09, 1.524842e+09,
              1.024842e+09, 9.24842e+08, 8.74842e+08, 8.24842e+08, 7.74842e+08,
              7.24842e+08, 6.74842e+08, 6.24842e+08, 5.74842e+08, 5.24842e+08,
              4.74842e+08, 4.24842e+08, 3.74842e+08, 3.24842e+08, 2.74842e+08,
              2.24842e+08, 1.74842e+08, 1.24842e+08, 7.4842e+07, 5.0244242e+08])  # 你的实际数据

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b', label='种群平均EDP')

# 添加标题和标签
plt.title('种群平均EDP随进化代数变化图')
plt.xlabel('进化代数')
plt.ylabel('种群平均EDP (cycles * pJ)')

# 显示网格
plt.grid(True)

# 添加图例
plt.legend()

# 保存图像
plt.savefig('conv3_population_average_EDP.png')  # 保存为PNG文件

# 如果不需要显示图像，可以注释掉这行
# plt.show()
