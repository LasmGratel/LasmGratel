import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体，优先使用 SimHei，回退到 DejaVu Sans
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 论域 U = {1, 2, 3, 4, 5}
domain = np.array([1, 2, 3, 4, 5])

# 输入模糊集
A  = np.array([1.0, 0.6, 0.3, 0.0, 0.0])   # 温度低
B  = np.array([0.0, 0.0, 0.3, 0.6, 1.0])   # 风门大
A_ = np.array([0.8, 1.0, 0.6, 0.3, 0.0])   # 温度较低（事实 A'）


def build_relation_matrix(A, B):
    """
    构建模糊关系矩阵 R（Mamdani 取小蕴含）
    R[i][j] = min(A[i], B[j])  — 笛卡尔积取最小
    """
    n = len(A)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i][j] = min(A[i], B[j])   # μ_R(x,y) = A(x) ∧ B(y)
    return R


def fuzzy_composition(A_prime, R):
    """
    模糊最大-最小合成推理
    B'[j] = max_i { min(A'[i], R[i][j]) }   — A' o R
    """
    n = len(A_prime)
    B_prime = np.zeros(n)
    for j in range(n):
        # 对每列 j：先取 min(A'[i], R[i][j])，再取所有 i 的 max
        B_prime[j] = max(min(A_prime[i], R[i][j]) for i in range(n))
    return B_prime


def defuzz_max_membership(B_prime, domain):
    """
    最大隶属度法：取隶属度最大的论域元素
    返回 (y, membership)
    """
    idx = np.argmax(B_prime)   # argmax μ_{B'}(y)
    return domain[idx], B_prime[idx]


def defuzz_weighted_average(B_prime, domain):
    """
    加权平均判决法（重心法）：
    y* = Σ(y · μ(y)) / Σμ(y)
    """
    total = np.sum(B_prime)
    if total == 0:
        return domain[0], 0.0
    y_star = np.sum(domain * B_prime) / total   # 加权平均
    return int(round(y_star)), y_star


def defuzz_median(B_prime, domain):
    """
    中位数法：找最小的 y，使累积隶属度之和 >= 总和的一半
    返回 (y, cumsum_at_y)：y 为判决值，cumsum_at_y 为到达该点的累积隶属度
    cumsum[j] >= total/2
    """
    total = np.sum(B_prime)
    half = total / 2.0
    cumsum = 0.0
    for j, y in enumerate(domain):
        cumsum += B_prime[j]
        if cumsum >= half:   # 累积和超过半数即停止
            return y, float(cumsum)
    return domain[-1], float(np.sum(B_prime))


def plot_fuzzy_sets(A, B, A_prime, domain, filename='fuzzy_input_sets.png'):
    """绘制输入模糊集 A、B、A' 的折线图"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(domain, A,       'b-o', label='A  (温度低)')
    ax.plot(domain, B,       'r-s', label='B  (风门大)')
    ax.plot(domain, A_prime, 'g-^', label="A' (温度较低)")
    ax.set_xlabel('论域 U')
    ax.set_ylabel('隶属度')
    ax.set_title('输入模糊集')
    ax.set_xticks(domain)
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)


def plot_relation_matrix(R, filename='fuzzy_relation_matrix.png'):
    """绘制模糊关系矩阵 R 的热力图（含数值标注）"""
    n = R.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(R, cmap='YlOrRd', vmin=0, vmax=1)
    # 标注每个格子的数值
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{R[i, j]:.1f}', ha='center', va='center', fontsize=10)
    ax.set_xticks(range(n))
    ax.set_xticklabels(range(1, n + 1))
    ax.set_yticks(range(n))
    ax.set_yticklabels(range(1, n + 1))
    ax.set_xlabel('风门开度论域 (B 方向)')
    ax.set_ylabel('温度论域 (A 方向)')
    ax.set_title('模糊关系矩阵 R = A × B（Mamdani 取小）')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)


def plot_result(B_prime, domain, results, filename='fuzzy_output.png'):
    """
    绘制推理输出 B' 的柱状图，并用竖线标注三种判决结果
    results = {'max_membership': y1, 'weighted_avg': y2, 'median': y3}
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(domain, B_prime, color='steelblue', alpha=0.7, label="B' (推理结果)")

    # 三种判决线（不同颜色区分）
    ax.axvline(x=results['max_membership'], color='red',    linestyle='--', linewidth=2,
               label=f"最大隶属度法 y={results['max_membership']}")
    ax.axvline(x=results['weighted_avg'],   color='green',  linestyle='-.',  linewidth=2,
               label=f"加权平均法 y={results['weighted_avg']}")
    ax.axvline(x=results['median'],         color='orange', linestyle=':',   linewidth=2,
               label=f"中位数法 y={results['median']}")

    ax.set_xlabel('论域 U（风门开度）')
    ax.set_ylabel('隶属度')
    ax.set_title("模糊推理输出 B' 及判决结果")
    ax.set_xticks(domain)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)


def main():
    # ------------------------------------------------------------------ #
    # 1. 构建模糊关系矩阵 R = A × B（Mamdani 取小蕴含）
    # ------------------------------------------------------------------ #
    R = build_relation_matrix(A, B)

    # ------------------------------------------------------------------ #
    # 2. 模糊合成推理 B' = A' o R（最大-最小合成）
    # ------------------------------------------------------------------ #
    B_prime = fuzzy_composition(A_, R)

    # ------------------------------------------------------------------ #
    # 3. 三种判决（去模糊化）
    # ------------------------------------------------------------------ #
    y_max, mem_max           = defuzz_max_membership(B_prime, domain)
    y_wavg, exact_wavg       = defuzz_weighted_average(B_prime, domain)
    y_med, cumsum_med            = defuzz_median(B_prime, domain)

    # ------------------------------------------------------------------ #
    # 4. 控制台输出（精确格式）
    # ------------------------------------------------------------------ #
    print('=' * 60)
    print('         模糊控制推理实验 - 风门开度确定')
    print('=' * 60)
    print('论域 U = {1, 2, 3, 4, 5}')
    print()
    print('【输入模糊集】')
    print('  温度低  (A)  = [' + ', '.join(f'{v}' for v in A) + ']')
    print('  风门大  (B)  = [' + ', '.join(f'{v}' for v in B) + ']')
    print("  温度较低(A') = [" + ', '.join(f'{v}' for v in A_) + ']')
    print()
    print('【模糊关系矩阵 R = A x B（Mamdani 取小）】')
    rows = []
    for row in R:
        rows.append('[' + ', '.join(f'{v:.1f}' for v in row) + ']')
    print('  [' + rows[0] + ',')
    for row_str in rows[1:-1]:
        print('   ' + row_str + ',')
    print('   ' + rows[-1] + ']')
    print()
    print('【模糊合成推理 B\' = A\' o R（最大-最小合成）】')
    print("  B' = [" + ', '.join(f'{round(v, 1)}' for v in B_prime) + ']')
    print()
    print('【判决结果】')
    print(f'  1. 最大隶属度法   -> 风门开度 = {y_max}  (隶属度 = {mem_max})')
    print(f'  2. 加权平均判决法 -> 风门开度 = {y_wavg}  (精确值 = {exact_wavg:.2f})')
    print(f'  3. 中位数法       -> 风门开度 = {y_med}  (累积中位 = {y_med})')
    print('=' * 60)

    # ------------------------------------------------------------------ #
    # 5. 绘图输出
    # ------------------------------------------------------------------ #
    plot_fuzzy_sets(A, B, A_, domain)
    plot_relation_matrix(R)
    results = {
        'max_membership': y_max,
        'weighted_avg':   y_wavg,
        'median':         y_med,
    }
    plot_result(B_prime, domain, results)


if __name__ == '__main__':
    main()
