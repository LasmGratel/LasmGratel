"""
无人机巡检路径规划 - GA vs ACO 算法对比实验
=============================================
问题：TSP（旅行商问题）
- 20个水质监测站，无人机从起点出发遍历所有节点后返回起点
- 目标：最小化总飞行距离

算法：
  1. 遗传算法（GA）：PMX交叉 + 两点交换变异 + 精英保留 + 锦标赛选择
  2. 蚁群算法（ACO）：状态转移概率 + 信息素挥发 + 精英蚂蚁全局更新
"""

import time
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合无显示环境
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ──────────────────────────────────────────────────────────────────────────────
# 全局设置
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 固定随机种子，保证可复现
SEED = 42
NUM_NODES = 20


# ──────────────────────────────────────────────────────────────────────────────
# 节点坐标生成与距离矩阵计算
# ──────────────────────────────────────────────────────────────────────────────
def generate_coordinates(n: int = NUM_NODES, seed: int = SEED) -> np.ndarray:
    """生成 n 个水质监测站的二维坐标，范围 [0, 100)"""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2) * 100  # shape: (n, 2)


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """计算各节点间的欧氏距离矩阵
    d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    """
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def path_length(path: list, dist: np.ndarray) -> float:
    """计算路径总长度（最后回到起点）"""
    total = sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))
    total += dist[path[-1]][path[0]]  # 回到起点
    return total


# ──────────────────────────────────────────────────────────────────────────────
# 遗传算法（GA）
# ──────────────────────────────────────────────────────────────────────────────
class GeneticAlgorithm:
    """
    遗传算法求解TSP
    编码：整数排列（permutation encoding）
    选择：锦标赛选择（tournament selection）
    交叉：PMX（Partially Mapped Crossover，部分映射交叉）
    变异：两点交换变异（swap mutation）
    精英：每代保留最优个体
    """

    def __init__(
        self,
        dist: np.ndarray,
        pop_size: int = 100,
        n_generations: int = 300,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 5,
        elite_size: int = 2,
        seed: int = SEED,
    ):
        self.dist = dist
        self.n = dist.shape[0]          # 节点数
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        random.seed(seed)
        np.random.seed(seed)

    # ── 适应度 ─────────────────────────────────────────────────────────────────
    def _fitness(self, individual: list) -> float:
        """适应度 = 1 / 路径长度（越大越好）"""
        return 1.0 / path_length(individual, self.dist)

    # ── 初始种群 ───────────────────────────────────────────────────────────────
    def _init_population(self) -> list:
        """随机生成初始种群（每个个体是 0..n-1 的一个排列）"""
        base = list(range(self.n))
        population = []
        for _ in range(self.pop_size):
            ind = base[:]
            random.shuffle(ind)
            population.append(ind)
        return population

    # ── 锦标赛选择 ─────────────────────────────────────────────────────────────
    def tournament_selection(self, population: list, fitnesses: list) -> list:
        """
        锦标赛选择：随机抽取 tournament_size 个个体，取适应度最高者
        保证选择压力适中，避免过早收敛
        """
        candidates_idx = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(candidates_idx, key=lambda i: fitnesses[i])
        return population[best_idx][:]

    # ── PMX 交叉 ───────────────────────────────────────────────────────────────
    def pmx_crossover(self, parent1: list, parent2: list) -> tuple:
        """
        PMX（Partially Mapped Crossover，部分映射交叉）
        步骤：
          1. 随机选择两个交叉点 [p1, p2]
          2. 子代1 的 [p1,p2] 段复制自 parent1；
             子代2 的 [p1,p2] 段复制自 parent2
          3. 利用映射关系填充剩余位置，保证排列不重复
        """
        n = len(parent1)
        p1, p2 = sorted(random.sample(range(n), 2))

        # 初始化子代（-1 表示未填充）
        child1 = [-1] * n
        child2 = [-1] * n

        # 直接复制交叉段
        child1[p1:p2 + 1] = parent1[p1:p2 + 1]
        child2[p1:p2 + 1] = parent2[p1:p2 + 1]

        # 建立映射关系
        mapping1 = {parent1[i]: parent2[i] for i in range(p1, p2 + 1)}
        mapping2 = {parent2[i]: parent1[i] for i in range(p1, p2 + 1)}

        def _fill(child, parent, mapping):
            for i in list(range(0, p1)) + list(range(p2 + 1, n)):
                val = parent[i]
                # 沿映射链追溯，直到不冲突
                while val in mapping:
                    val = mapping[val]
                child[i] = val

        _fill(child1, parent2, mapping1)
        _fill(child2, parent1, mapping2)
        return child1, child2

    # ── 两点交换变异 ───────────────────────────────────────────────────────────
    def swap_mutation(self, individual: list) -> list:
        """
        两点交换变异：随机选择两个位置，交换对应节点
        时间复杂度 O(1)，简单高效
        """
        ind = individual[:]
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
        return ind

    # ── 主运行函数 ─────────────────────────────────────────────────────────────
    def run(self):
        """
        运行遗传算法
        返回：(best_path, best_distance, history)
          history: 每代最优路径长度列表
        """
        population = self._init_population()
        fitnesses = [self._fitness(ind) for ind in population]

        best_idx = int(np.argmax(fitnesses))
        best_path = population[best_idx][:]
        best_distance = path_length(best_path, self.dist)
        last_update_gen = 0

        history = [best_distance]

        for gen in range(1, self.n_generations + 1):
            # ── 精英保留：对适应度排序，保留最优的 elite_size 个体
            sorted_idx = sorted(range(self.pop_size), key=lambda i: fitnesses[i], reverse=True)
            elites = [population[i][:] for i in sorted_idx[:self.elite_size]]

            new_population = elites[:]

            # ── 生成子代
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)

                # 交叉
                if random.random() < self.crossover_rate:
                    c1, c2 = self.pmx_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                # 变异
                if random.random() < self.mutation_rate:
                    c1 = self.swap_mutation(c1)
                if random.random() < self.mutation_rate:
                    c2 = self.swap_mutation(c2)

                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = new_population
            fitnesses = [self._fitness(ind) for ind in population]

            # ── 更新全局最优
            cur_best_idx = int(np.argmax(fitnesses))
            cur_best_dist = path_length(population[cur_best_idx], self.dist)
            if cur_best_dist < best_distance:
                best_distance = cur_best_dist
                best_path = population[cur_best_idx][:]
                last_update_gen = gen

            history.append(best_distance)

        return best_path, best_distance, history, last_update_gen


# ──────────────────────────────────────────────────────────────────────────────
# 蚁群算法（ACO）
# ──────────────────────────────────────────────────────────────────────────────
class AntColonyOptimization:
    """
    蚁群算法求解TSP
    状态转移概率：p_ij = (tau_ij^alpha * eta_ij^beta) / sum_k(tau_ik^alpha * eta_ik^beta)
    信息素挥发：  tau_ij = (1 - rho) * tau_ij
    精英蚂蚁全局更新：tau_ij += Q / L_best（仅最优路径上的边）
    """

    def __init__(
        self,
        dist: np.ndarray,
        n_ants: int = 50,
        n_iterations: int = 300,
        alpha: float = 1.0,   # 信息素重要程度
        beta: float = 5.0,    # 启发式信息重要程度
        rho: float = 0.5,     # 信息素挥发系数
        Q: float = 100.0,     # 信息素增量常数
        seed: int = SEED,
    ):
        self.dist = dist
        self.n = dist.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        np.random.seed(seed)

        # 启发式信息矩阵 eta_ij = 1 / d_ij（避免除零）
        with np.errstate(divide='ignore', invalid='ignore'):
            self.eta = np.where(dist > 0, 1.0 / dist, 0.0)  # shape: (n, n)

        # 初始化信息素矩阵（全部设为1）
        self.tau = np.ones((self.n, self.n))

    # ── 构建单只蚂蚁的路径 ────────────────────────────────────────────────────
    def _build_solution(self, start: int = 0) -> list:
        """
        蚂蚁从 start 节点出发，按状态转移概率逐步构建路径
        p_ij = (tau_ij^alpha * eta_ij^beta) / sum_k(tau_ik^alpha * eta_ik^beta)
        """
        visited = [False] * self.n
        path = [start]
        visited[start] = True

        for _ in range(self.n - 1):
            current = path[-1]
            unvisited = [j for j in range(self.n) if not visited[j]]
            # 计算可去节点的概率分子（仅对未访问节点）
            weights = np.array([
                (self.tau[current][j] ** self.alpha) * (self.eta[current][j] ** self.beta)
                for j in unvisited
            ])  # tau_ij^alpha * eta_ij^beta

            total = weights.sum()
            if total == 0:
                # 若全部为0（数值问题），均匀随机选择未访问节点
                next_node = np.random.choice(unvisited)
            else:
                probs = weights / total  # 归一化为概率分布
                # 仅从未访问节点中采样，避免浮点精度问题选到已访问节点
                next_node = int(np.random.choice(unvisited, p=probs))

            path.append(next_node)
            visited[next_node] = True

        return path

    # ── 信息素更新 ────────────────────────────────────────────────────────────
    def _update_pheromone(self, best_path: list, best_dist: float):
        """
        全局信息素更新（精英蚂蚁策略）：
          1. 挥发：tau_ij = (1 - rho) * tau_ij
          2. 精英蚂蚁沉积：tau_ij += Q / L_best（仅最优路径上的边）
        """
        # 挥发
        self.tau *= (1.0 - self.rho)

        # 精英蚂蚁沉积（最优路径的每条边增加信息素）
        delta = self.Q / best_dist
        for i in range(len(best_path)):
            a = best_path[i]
            b = best_path[(i + 1) % len(best_path)]  # 循环，最后回起点
            self.tau[a][b] += delta
            self.tau[b][a] += delta  # 对称路径

    # ── 主运行函数 ────────────────────────────────────────────────────────────
    def run(self):
        """
        运行蚁群算法
        返回：(best_path, best_distance, history, last_update_gen)
        """
        best_path = None
        best_distance = float('inf')
        last_update_gen = 0
        history = []

        for iteration in range(self.n_iterations):
            # 所有蚂蚁构建路径（起点随机分配，提高多样性）
            solutions = []
            for k in range(self.n_ants):
                start = k % self.n  # 均匀分配起点
                path = self._build_solution(start)
                dist = path_length(path, self.dist)
                solutions.append((path, dist))

            # 找本次迭代最优
            iter_best_path, iter_best_dist = min(solutions, key=lambda x: x[1])

            # 更新全局最优
            if iter_best_dist < best_distance:
                best_distance = iter_best_dist
                best_path = iter_best_path[:]
                last_update_gen = iteration + 1  # 1-based

            # 用全局最优更新信息素（精英蚂蚁策略）
            self._update_pheromone(best_path, best_distance)

            history.append(best_distance)

        return best_path, best_distance, history, last_update_gen


# ──────────────────────────────────────────────────────────────────────────────
# 可视化函数
# ──────────────────────────────────────────────────────────────────────────────
def plot_path(
    coords: np.ndarray,
    path: list,
    title: str,
    filename: str,
    distance: float,
):
    """
    绘制最优路径图
    - 节点以散点图标注，起点特殊标记
    - 路径用箭头连线
    - 标题含最优距离
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # 完整路径（回到起点）
    full_path = path + [path[0]]
    xs = [coords[i][0] for i in full_path]
    ys = [coords[i][1] for i in full_path]

    # 绘制路径线段
    ax.plot(xs, ys, 'b-', linewidth=1.2, alpha=0.7, zorder=1)

    # 绘制箭头（每段路径中点放箭头，表示方向）
    for i in range(len(full_path) - 1):
        x1, y1 = coords[full_path[i]]
        x2, y2 = coords[full_path[i + 1]]
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2),
            zorder=2,
        )

    # 绘制普通节点
    non_start = [i for i in path if i != path[0]]
    ax.scatter(
        [coords[i][0] for i in non_start],
        [coords[i][1] for i in non_start],
        c='dodgerblue', s=80, zorder=3, label='监测站'
    )

    # 突出显示起点
    ax.scatter(
        coords[path[0]][0], coords[path[0]][1],
        c='red', s=150, marker='*', zorder=4, label=f'起点 (节点 {path[0]})'
    )

    # 节点编号标注
    for i in range(len(coords)):
        ax.annotate(
            str(i),
            (coords[i][0], coords[i][1]),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=8,
            color='darkblue',
        )

    ax.set_title(f'{title}\n最优路径长度：{distance:.2f}', fontsize=13, fontweight='bold')
    ax.set_xlabel('X 坐标', fontsize=11)
    ax.set_ylabel('Y 坐标', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> 已保存路径图：{filename}')


def plot_convergence(
    ga_history: list,
    aco_history: list,
    filename: str,
):
    """
    绘制两种算法的收敛曲线对比图
    X轴：迭代次数，Y轴：最优路径长度
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ga_iters = list(range(len(ga_history)))
    aco_iters = list(range(len(aco_history)))

    ax.plot(ga_iters, ga_history, color='tomato', linewidth=1.8,
            label=f'遗传算法 (GA)  最终：{ga_history[-1]:.2f}')
    ax.plot(aco_iters, aco_history, color='steelblue', linewidth=1.8,
            label=f'蚁群算法 (ACO) 最终：{aco_history[-1]:.2f}')

    # 标注最终值
    ax.axhline(y=ga_history[-1], color='tomato', linestyle=':', alpha=0.5)
    ax.axhline(y=aco_history[-1], color='steelblue', linestyle=':', alpha=0.5)

    ax.set_title('GA vs ACO 收敛曲线对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('最优路径长度', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> 已保存收敛曲线：{filename}')


def plot_mse_comparison(
    ga_self_mse: float,
    aco_self_mse: float,
    cross_mse: float,
    filename: str,
):
    """
    绘制三个 MSE 指标的柱状图（对数刻度），并标注具体数值
    """
    labels = ['GA自身MSE', 'ACO自身MSE', 'GA-ACO交叉MSE']
    values = [ga_self_mse, aco_self_mse, cross_mse]
    colors = ['tomato', 'steelblue', 'mediumseagreen']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)

    # 每根柱子标注具体数值
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.1,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
        )

    ax.set_yscale('log')  # 对数刻度，适应数值差异较大的情况
    ax.set_title('收敛曲线 MSE 对比（scikit-learn）', fontsize=14, fontweight='bold')
    ax.set_ylabel('MSE（对数刻度）', fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> 已保存 MSE 对比图：{filename}')


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1. 生成问题实例
    coords = generate_coordinates(NUM_NODES, SEED)
    dist = compute_distance_matrix(coords)

    print('=' * 60)
    print('       无人机巡检路径规划 - GA vs ACO 算法对比实验')
    print('=' * 60)
    print(f'节点数量: {NUM_NODES}')
    print('-' * 60)

    # ── 2. 运行遗传算法
    ga = GeneticAlgorithm(
        dist=dist,
        pop_size=100,
        n_generations=300,
        crossover_rate=0.8,
        mutation_rate=0.1,
        seed=SEED,
    )
    t0 = time.time()
    ga_path, ga_dist, ga_history, ga_last_gen = ga.run()
    ga_time = time.time() - t0

    # ── 3. 运行蚁群算法
    aco = AntColonyOptimization(
        dist=dist,
        n_ants=50,
        n_iterations=300,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        Q=100.0,
        seed=SEED,
    )
    t0 = time.time()
    aco_path, aco_dist, aco_history, aco_last_gen = aco.run()
    aco_time = time.time() - t0

    # ── 4. 打印对比表格
    print(f'{"算法":<14}{"最优路径长度":>14}{"收敛代数":>12}{"运行时间(s)":>14}')
    print('-' * 60)
    print(f'{"遗传算法(GA)":<14}{ga_dist:>14.2f}{ga_last_gen:>12}{ga_time:>14.2f}')
    print(f'{"蚁群算法(ACO)":<14}{aco_dist:>14.2f}{aco_last_gen:>12}{aco_time:>14.2f}')
    print('-' * 60)

    # 寻优精度对比
    if ga_dist < aco_dist:
        winner = 'GA'
        gap_pct = (aco_dist - ga_dist) / aco_dist * 100
    elif aco_dist < ga_dist:
        winner = 'ACO'
        gap_pct = (ga_dist - aco_dist) / ga_dist * 100
    else:
        winner = '平局'
        gap_pct = 0.0
    print(f'寻优精度对比: {winner} 算法找到更优解，差距 {gap_pct:.2f}%')
    print('=' * 60)

    # ── 5a. MSE 收敛质量分析
    ga_self_mse = mean_squared_error([ga_history[-1]] * len(ga_history), ga_history)
    aco_self_mse = mean_squared_error([aco_history[-1]] * len(aco_history), aco_history)
    min_len = min(len(ga_history), len(aco_history))
    cross_mse = mean_squared_error(ga_history[:min_len], aco_history[:min_len])

    print('-' * 60)
    print('【MSE 收敛质量分析（基于 scikit-learn）】')
    print(f'GA  自身收敛MSE（与最终值偏差）：{ga_self_mse:>14.2f}')
    print(f'ACO 自身收敛MSE（与最终值偏差）：{aco_self_mse:>14.2f}')
    print('  -> 自身MSE越小，说明算法收敛越平稳、波动越小')
    print(f'GA vs ACO 交叉MSE（两曲线差异）：{cross_mse:>14.2f}')
    print('  -> 交叉MSE反映两算法整体收敛路径的差异程度')
    print('-' * 60)

    # ── 5. 分析结论
    # 收敛速度：比较前50代下降幅度
    def early_drop_ratio(history, early=50):
        """前 early 代相对于初始值的下降比例"""
        if len(history) <= early:
            return 0.0
        return (history[0] - history[early]) / history[0] * 100

    ga_drop = early_drop_ratio(ga_history)
    aco_drop = early_drop_ratio(aco_history)
    faster = 'GA' if ga_drop > aco_drop else 'ACO'
    local_opt = 'GA' if ga_last_gen < aco_last_gen else 'ACO'

    print('\n【分析结论】')
    print(
        f'1. 收敛速度：{faster} 算法在迭代初期收敛更快，'
        f'前50代 GA 下降 {ga_drop:.1f}%，ACO 下降 {aco_drop:.1f}%。'
        f'ACO 依赖信息素积累，初期随机性较强，但后期正反馈机制使其快速聚焦；'
        f'GA 通过种群多样性维持探索。'
    )
    print(
        f'2. 局部最优：{local_opt} 更容易陷入局部最优（最优解最后更新代数更早）。'
        f'GA 的最优解在第 {ga_last_gen} 代最后更新，'
        f'ACO 在第 {aco_last_gen} 代最后更新。'
        f'ACO 的信息素正反馈在后期可能导致路径固化，'
        f'GA 的变异算子提供持续扰动，有助于跳出局部最优。'
    )
    print(
        '3. 参数敏感性：ACO 对 alpha、beta、rho 较敏感，'
        '尤其是 beta（启发式权重）过大时蚂蚁几乎变成贪心策略；'
        'GA 的 PMX 交叉能较好保持排列有效性，'
        '变异率过高则破坏良好个体，建议保持在 0.05~0.15 之间。'
    )
    stable_winner = 'GA' if ga_self_mse < aco_self_mse else 'ACO'
    cross_level = '较大' if cross_mse > max(ga_self_mse, aco_self_mse) else '较小'
    print(
        f'4. MSE收敛质量：GA自身MSE为 {ga_self_mse:.2f}，ACO自身MSE为 {aco_self_mse:.2f}。\n'
        f'   {stable_winner} 算法自身MSE更小，说明其收敛过程更稳定，路径长度波动更小；\n'
        f'   两算法交叉MSE为 {cross_mse:.2f}，反映两者整体收敛轨迹存在{cross_level}差异。'
    )
    print('=' * 60)

    # ── 6. 生成图表
    print('\n正在生成图表...')
    plot_path(coords, ga_path,  '遗传算法(GA) 最优巡检路径',  'ga_best_path.png',  ga_dist)
    plot_path(coords, aco_path, '蚁群算法(ACO) 最优巡检路径', 'aco_best_path.png', aco_dist)
    plot_convergence(ga_history, aco_history, 'convergence_comparison.png')
    plot_mse_comparison(ga_self_mse, aco_self_mse, cross_mse, 'mse_comparison.png')
    print('\n所有图表已保存完毕。')


if __name__ == '__main__':
    main()
