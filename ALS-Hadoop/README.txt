五个版本的数据集地址：
https://files.grouplens.org/datasets/movielens/ml-100k.zip
https://files.grouplens.org/datasets/movielens/ml-1m.zip
https://files.grouplens.org/datasets/movielens/ml-10m.zip
https://files.grouplens.org/datasets/movielens/ml-20m.zip
https://files.grouplens.org/datasets/movielens/ml-25m.zip



Mahout库（Hadoop 2.6.0; jdk1.8.0_60; Scala 2.10.6; 使用Maven 3.8.4编译）：
mahout-core-0.9.jar
mahout-math-0.9.jar

MapReduce代码：
ALS文件夹

我们的程序可以分为如下三部分：预处理，交替最小二乘，评价。
预处理 预处理阶段主要完成的工作是读取数据，构建用户-物品评分矩阵 R，并对矩
阵 U 和 M 进行初始化，具体过程为：
(a) 读取文本格式的训练集数据文件，使用 MapReduce 输出（物品编号，所有用户
对该物品的评分向量）这样的键值对，即每一个条目存储了矩阵 R 中的一列。
同时根据输出文件的总行数可以得到物品的总数量。
(b) 读取 (a) 中的 MapReduce 结果，使用 MapReduce 对其进行转置，输出（用户
编号，该用户对所有物品的评分向量）这样的键值对，即每一个条目存储了矩
阵 R 中的一行。同时根据输出文件的总行数可以得到用户的总数量。
(c) 读取 (a) 中的 MapReduce 结果，使用 MapReduce 计算出每个物品的平均评
分。利用该评分对 M 矩阵进行初始化，每个物品特征的第一维设为其平均得
分，其它维随机初始化，并将 M 输出到文件中。
交替最小二乘 这一步通过交替执行以下两个 MapReduce Job，迭代求解矩阵 U 和
M：
(d) 固定 M，求解 U。每个 Mapper 求解一个用户对应的向量。首先在 setup 阶段
将上一步计算得到的 M 读入，并在所有线程间共享。然后每个 Mapper 从 (b)
的输出结果中读取一个用户的评分向量 Au，并求解对应的最小二乘问题

求解部分直接使用了 Mahout 库中的 AlternatingLeastSquaresSolver，将问题
转化为求解
(M˜ TM˜ + λνI)x = M˜ TRu
6
其中 ν 是用户 u 打分的物品数量，M˜ 为矩阵 M 中用户 u 打分的物品对应的行
构成的矩阵。通过对矩阵 (M˜ TM˜ + λνI) 进行 QR 分解，可以求得 Ui 的解，作
为新的用户矩阵 U 中的第 i 行输出到文件中。
(e) 固定 U，求解 M。每个 Mapper 求解一个物品对应的向量。具体过程与 (d) 同
理。
评价 求得用户矩阵 U 和物品矩阵 M 以后，通过以下步骤在测试集上计算预测评分与
真实评分的均方根误差（RMSE）：
(f) 使用 MapReduce 计算每条测试数据的误差。首先在 setup 阶段读入计算得到
的 U 和 M 矩阵，然后每个 Mapper 从文本格式的测试集数据文件中读取一行
评分数据，根据用户编号 i 与物品编号 j 计算出预测评分 U
T
i Mj，并将误差输
出。对输出文件中的误差计算平方平均并开根号即可得到 RMSE。
