"""
optics_xiu_modified_v2.py
海洋微结构数据OPTICS聚类分析
特征：EPSILON、dSdz、N2
数据来源：OPTICS/data
输出目录：OPTICS/result3
"""

import numpy as np
import matplotlib.pyplot as plt
# 修复字体设置 - 关闭LaTeX，使用中文字体
plt.rcParams['text.usetex'] = False  # 关闭LaTeX渲染
plt.rcParams['font.family'] = ['sans-serif']  # 使用无衬线字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文字体优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
import os
import json
import pickle
from scipy.io import loadmat
import pandas as pd

# 设置随机种子确保可重复性
np.random.seed(42)

# ==================== 1. 辅助函数 ====================
def explore_mat_file(filepath):
    """探查.mat文件的内容结构"""
    print(f"\n[探查] {os.path.basename(filepath)}")
    try:
        data = loadmat(filepath)
        print(f"  变量列表: {list(data.keys())}")
        for key in list(data.keys())[:6]:
            if not key.startswith('__'):
                var = data[key]
                if hasattr(var, 'shape'):
                    print(f"    {key}: shape={var.shape}, dtype={var.dtype}")
                else:
                    print(f"    {key}: type={type(var)}")
        return data
    except Exception as e:
        print(f"  读取失败: {e}")
        return None

# ==================== 2. 数据加载与特征提取 ====================
def load_and_extract_features(data_dir='OPTICS/data'):
    """从指定目录加载数据并提取EPSILON, dSdz, N2特征"""
    all_features = {}
    file_features = {}

    if not os.path.exists(data_dir):
        print(f"目录 '{data_dir}' 不存在。将使用模拟数据。")
        return generate_simulated_features()

    # 优先查找 *_fit.mat 文件
    fit_files = [f for f in os.listdir(data_dir) if f.endswith('_fit.mat')]
    if not fit_files:
        print(f"在 '{data_dir}' 中未找到 *_fit.mat 文件。")
        other_mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat') and '_fit' not in f]
        fit_files = other_mat_files[:1]
        if not fit_files:
            print("也未找到其他 .mat 文件，将使用模拟数据。")
            return generate_simulated_features()

    for fit_file in fit_files:
        filepath = os.path.join(data_dir, fit_file)
        data = explore_mat_file(filepath)
        if data is None:
            continue

        features = {}
        # 尝试提取特征：EPSILON, dSdz, N2
        eps_keys = ['EPSILON', 'epsilon', 'eps', 'dissipation', 'data']
        dSdz_keys = ['dSdz', 'SAL_GRAD', 'salt_gradient', 'data']
        n2_keys = ['N2', 'buoyancy_frequency_squared', 'data']
        
        # 为每个特征设置对应的列索引
        col_idx_map = {'EPSILON': 0, 'dSdz': 1, 'N2': 2}
        key_lists = [eps_keys, dSdz_keys, n2_keys]
        feat_names = ['EPSILON', 'dSdz', 'N2']
        
        for key_list, feat_name in zip(key_lists, feat_names):
            value = None
            for k in key_list:
                if k in data:
                    val = data[k]
                    if hasattr(val, 'flatten'):
                        val_flat = val.flatten()
                        if len(val_flat) > 0:
                            if val.ndim == 2 and val.shape[1] > 1:
                                col_idx = col_idx_map[feat_name]
                                if val.shape[1] > col_idx:
                                    value = val[:, col_idx]
                                else:
                                    value = val_flat
                            else:
                                value = val_flat
                            print(f"      从变量 '{k}' 提取特征 '{feat_name}'，形状 {value.shape}")
                            break
            features[feat_name] = value

        # 检查是否成功提取所有特征
        if all(v is not None for v in features.values()):
            min_len = min(len(v) for v in features.values())
            for k in features:
                features[k] = features[k][:min_len]
            file_features[fit_file] = features
            print(f"  √ 成功从 '{fit_file}' 提取特征")
        else:
            print(f"  × 从 '{fit_file}' 提取特征不完整。")
            
            # 尝试从'data'矩阵中提取特征
            if '_fit' in fit_file and 'data' in data:
                data_matrix = data['data']
                if data_matrix.ndim == 2 and data_matrix.shape[1] >= 3:
                    features = {
                        'EPSILON': data_matrix[:, 0],
                        'dSdz': data_matrix[:, 1],
                        'N2': data_matrix[:, 2]
                    }
                    file_features[fit_file] = features
                    print(f"  √ 从 '{fit_file}' 的 'data' 矩阵中提取特征")
            
            # 尝试从'columns'和'NN_input_all'中提取特征
            elif 'columns' in data and 'NN_input_all' in data:
                columns_data = data['columns']
                nn_input_data = data['NN_input_all']
                
                # 提取列名
                column_names = []
                for i in range(len(columns_data)):
                    if hasattr(columns_data[i][0], '__len__'):
                        column_names.append(str(columns_data[i][0]))
                    else:
                        column_names.append(str(columns_data[i]))
                
                print(f"  列名: {column_names}")
                
                # 查找目标列索引
                target_columns = {
                    'EPSILON': ['epsilon', 'eps', 'dissipation'],
                    'dSdz': ['dsdz', 'sal_grad', 'salt_gradient'],
                    'N2': ['n2', 'buoyancy', 'frequency']
                }
                
                extracted_features = {}
                for feature, aliases in target_columns.items():
                    found = False
                    for i, col_name in enumerate(column_names):
                        col_lower = col_name.lower()
                        for alias in aliases:
                            if alias in col_lower and i < nn_input_data.shape[1]:
                                extracted_features[feature] = nn_input_data[:, i]
                                print(f"    从列'{col_name}'提取{feature}")
                                found = True
                                break
                        if found:
                            break
                
                if all(feat in extracted_features for feat in ['EPSILON', 'dSdz', 'N2']):
                    min_len = min(len(extracted_features['EPSILON']), 
                                 len(extracted_features['dSdz']), 
                                 len(extracted_features['N2']))
                    for k in extracted_features:
                        extracted_features[k] = extracted_features[k][:min_len]
                    file_features[fit_file] = extracted_features
                    print(f"  √ 从 '{fit_file}' 的 'NN_input_all' 中提取特征")

    if not file_features:
        print("无法从任何文件中提取有效特征，将使用模拟数据。")
        return generate_simulated_features()

    first_file = list(file_features.keys())[0]
    all_features = file_features[first_file]
    print(f"\n使用文件 '{first_file}' 的特征进行分析。")
    return all_features

def generate_simulated_features(n_samples=1500):
    """生成模拟的海洋微结构特征数据，包含深度和时间信息"""
    print("生成模拟数据...")
    np.random.seed(42)
    
    # 生成深度和时间信息
    depth = np.linspace(0, 300, n_samples)  # 0-300米
    time = np.linspace(0, 24, n_samples)    # 24小时
    
    # 生成模拟的海洋结构
    # 表层混合层 (0-50m)
    surf_mask = depth < 50
    # 温跃层 (50-200m)
    thermo_mask = (depth >= 50) & (depth < 200)
    # 深层 (200-300m)
    deep_mask = depth >= 200
    
    # 初始化特征
    EPSILON = np.zeros(n_samples)
    dSdz = np.zeros(n_samples)
    N2 = np.zeros(n_samples)
    
    # 表层: 高EPSILON, 中等dSdz, 低N2
    EPSILON[surf_mask] = np.random.lognormal(np.log(1e-5), 0.5, np.sum(surf_mask))
    dSdz[surf_mask] = np.random.lognormal(np.log(0.01), 0.5, np.sum(surf_mask))
    N2[surf_mask] = np.random.lognormal(np.log(1e-6), 0.3, np.sum(surf_mask))
    
    # 温跃层: 中等EPSILON, 高dSdz, 高N2
    EPSILON[thermo_mask] = np.random.lognormal(np.log(1e-6), 0.4, np.sum(thermo_mask))
    dSdz[thermo_mask] = np.random.lognormal(np.log(0.1), 0.4, np.sum(thermo_mask))
    N2[thermo_mask] = np.random.lognormal(np.log(1e-4), 0.2, np.sum(thermo_mask))
    
    # 深层: 低EPSILON, 低dSdz, 低N2
    EPSILON[deep_mask] = np.random.lognormal(np.log(1e-8), 0.3, np.sum(deep_mask))
    dSdz[deep_mask] = np.random.lognormal(np.log(0.001), 0.3, np.sum(deep_mask))
    N2[deep_mask] = np.random.lognormal(np.log(1e-5), 0.2, np.sum(deep_mask))
    
    # 添加时间变化
    EPSILON *= (1 + 0.3 * np.sin(2 * np.pi * time / 12))
    dSdz *= (1 + 0.2 * np.sin(2 * np.pi * time / 12))
    
    features = {
        'EPSILON': EPSILON,
        'dSdz': dSdz,
        'N2': N2,
        'depth': depth,
        'time': time
    }
    
    print(f"模拟数据生成完成: EPSILON({len(EPSILON)}), dSdz({len(dSdz)}), N2({len(N2)})")
    return features

# ==================== 3. 数据预处理 ====================
def preprocess_features(features_dict):
    """按照论文方法预处理特征：对数变换 + 标准化"""
    EPSILON = features_dict['EPSILON'].astype(np.float64)
    dSdz = features_dict['dSdz'].astype(np.float64)
    N2 = features_dict['N2'].astype(np.float64)
    
    # 检查dSdz的范围
    dSdz_min, dSdz_max = dSdz.min(), dSdz.max()
    print(f"dSdz原始范围: {dSdz_min:.3e} 到 {dSdz_max:.3e}")
    
    # 如果dSdz包含负值，不能直接取对数
    if dSdz_min <= 0:
        # 对于包含负值的dSdz，可以平移，使最小值变为一个小的正数
        dSdz_shifted = dSdz - dSdz_min + 1e-10
        dSdz_log = np.log10(dSdz_shifted)
        print(f"dSdz包含负值，已平移处理")
    else:
        dSdz_log = np.log10(dSdz + 1e-10)
    
    # 对N2取对数
    N2_log = np.log10(N2 + 1e-20)
    
    # 假设EPSILON已经是log10尺度，不需要再次取对数
    EPSILON_log = EPSILON
    
    # 组合为特征矩阵
    X_raw = np.column_stack([EPSILON_log, dSdz_log, N2_log])
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 移除无效值
    valid_mask = np.all(np.isfinite(X_scaled), axis=1)
    X_clean = X_scaled[valid_mask]
    
    print(f"预处理结果: 原始样本 {len(X_raw)} -> 有效样本 {len(X_clean)}")
    return X_clean, scaler, valid_mask

# ==================== 4. 滑动窗口平均 ====================
def create_moving_average_windows(X, window_size=100, step=50):
    """创建滑动窗口平均值，模拟论文中的垂直窗口平均"""
    n_samples = X.shape[0]
    windows = []
    start_indices = []
    for i in range(0, n_samples - window_size + 1, step):
        window = X[i:i+window_size]
        window_mean = np.mean(window, axis=0)
        windows.append(window_mean)
        start_indices.append(i)
    X_windows = np.array(windows)
    print(f"窗口平均: 窗口大小 {window_size}, 步长 {step}")
    print(f"生成 {len(X_windows)} 个窗口特征向量")
    return X_windows, np.array(start_indices)

# ==================== 5. OPTICS 聚类 ====================
def perform_optics_clustering(X, min_samples=5, xi=0.1):
    """执行OPTICS聚类，参数参考论文设置"""
    print("\n执行 OPTICS 聚类...")
    optics = OPTICS(min_samples=min_samples, max_eps=np.inf,
                    cluster_method='xi', xi=xi, metric='euclidean',
                    n_jobs=-1)
    optics.fit(X)
    labels = optics.labels_
    reachability = optics.reachability_
    ordering = optics.ordering_
    core_distances = optics.core_distances_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  聚类数: {n_clusters}")
    print(f"  噪声点: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    if n_clusters > 0:
        cluster_sizes = np.bincount(labels[labels>=0])
        print(f"  各聚类大小: {cluster_sizes}")
    
    return {
        'model': optics,
        'labels': labels,
        'reachability': reachability,
        'ordering': ordering,
        'core_distances': core_distances,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }

# ==================== 6. 可视化结果 ====================
def create_visualizations(X, labels, reachability, ordering, core_distances, n_clusters, output_dir, data_source):
    """生成可视化图表"""
    print("\n生成可视化图表...")
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 子图1: 三维散点图（原始数据）
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], 
                          c='blue', s=20, alpha=0.7)
    ax1.set_title(f'原始数据（{data_source}，无标签）', fontsize=12)
    ax1.set_xlabel('EPSILON (log)', fontsize=10)
    ax1.set_ylabel('dSdz (log)', fontsize=10)
    ax1.set_zlabel('N*N (log)', fontsize=10)
    
    # 子图2: 三维散点图（OPTICS聚类结果）
    ax2 = fig.add_subplot(232, projection='3d')
    labels_flat = labels.flatten() if labels.ndim > 1 else labels
    
    # 为每个聚类生成颜色
    norm = plt.Normalize(labels_flat.min(), labels_flat.max())
    cmap = plt.cm.tab20
    colors_optics = cmap(norm(labels_flat))
    
    # 为噪声点设置特殊颜色（灰色）
    noise_mask = labels_flat == -1
    if noise_mask.any():
        colors_optics[noise_mask] = [0.5, 0.5, 0.5, 1.0]  # 灰色，RGBA格式
    
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], 
                          c=colors_optics, s=20, alpha=0.7)
    ax2.set_xlabel('EPSILON (log)', fontsize=10)
    ax2.set_ylabel('dSdz (log)', fontsize=10)
    ax2.set_zlabel('N*N (log)', fontsize=10)
    ax2.set_title(f'OPTICS聚类结果 (ξ方法, {n_clusters}个聚类)', fontsize=12)
    
    # 创建图例
    from matplotlib.patches import Patch
    legend_elements = []
    if n_clusters > 0:
        for i in range(n_clusters):
            legend_elements.append(Patch(facecolor=cmap(norm(i)), 
                                         label=f'聚类 {i+1}'))
    if noise_mask.any():
        legend_elements.append(Patch(facecolor=[0.5, 0.5, 0.5, 1.0], 
                                     label='噪声点'))
    if legend_elements:
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # 子图3: 可达性图（OPTICS核心输出）
    ax3 = fig.add_subplot(233)
    # 按OPTICS排序顺序绘制可达性距离
    reach_ordered = reachability[ordering]
    labels_ordered = labels[ordering]
    space = np.arange(len(X))
    
    # 用不同颜色标记聚类
    unique_labels = set(labels[ordering])
    for label in unique_labels:
        if label == -1:
            continue
        class_member_mask = (labels[ordering] == label)
        xy = space[class_member_mask]
        ax3.plot(xy, reach_ordered[class_member_mask], 'o', markersize=4)
    
    # 标记噪声点
    noise_mask_plot = (labels[ordering] == -1)
    if noise_mask_plot.any():
        ax3.plot(space[noise_mask_plot], reach_ordered[noise_mask_plot], 'o', 
                 markersize=4, color='gray', alpha=0.5, label='噪声点')
    
    ax3.set_xlabel('样本点 (OPTICS排序)', fontsize=10)
    ax3.set_ylabel('可达性距离', fontsize=10)
    ax3.set_title('OPTICS可达性图', fontsize=12)
    ax3.grid(True, alpha=0.3)
    if noise_mask_plot.any():
        ax3.legend()
    
    # 子图4: 核心距离分布
    ax4 = fig.add_subplot(234)
    # 过滤掉无穷大值
    finite_core_distances = core_distances[np.isfinite(core_distances)]
    if len(finite_core_distances) > 0:
        ax4.hist(finite_core_distances, bins=50, 
                 alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('核心距离', fontsize=10)
        ax4.set_ylabel('频数', fontsize=10)
        ax4.set_title('核心距离分布', fontsize=12)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '无有效核心距离数据', 
                 ha='center', va='center', transform=ax4.transAxes)
    
    # 子图5: 二维投影（EPSILON vs dSdz）
    ax5 = fig.add_subplot(235)
    scatter5 = ax5.scatter(X[:, 0], X[:, 1], c=labels_flat, 
                          cmap='tab20', s=20, alpha=0.7)
    ax5.set_xlabel('EPSILON (log)', fontsize=10)
    ax5.set_ylabel('dSdz (log)', fontsize=10)
    ax5.set_title('EPSILON-dSdz 平面投影', fontsize=12)
    plt.colorbar(scatter5, ax=ax5, label='聚类标签')
    
    # 子图6: 二维投影（dSdz vs N²）
    ax6 = fig.add_subplot(236)
    scatter6 = ax6.scatter(X[:, 1], X[:, 2], c=labels_flat, 
                          cmap='tab20', s=20, alpha=0.7)
    ax6.set_xlabel('dSdz (log)', fontsize=10)
    ax6.set_ylabel('N*N (log)', fontsize=10)
    ax6.set_title('dSdz-N*N 平面投影', fontsize=12)
    plt.colorbar(scatter6, ax=ax6, label='聚类标签')
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'comprehensive_clustering_results.png'), dpi=300, bbox_inches='tight')
    print(f"  综合图表已保存: {os.path.join(output_dir, 'comprehensive_clustering_results.png')}")
    
    plt.show()
    return fig

# ==================== 7. 聚类特征分析 ====================
def analyze_cluster_statistics(X, labels, n_clusters):
    """计算并打印每个聚类的统计特征"""
    print("\n各聚类特征统计:")
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        if np.sum(cluster_mask) > 0:
            cluster_data = X[cluster_mask]
            
            print(f"\n聚类 {cluster_id} (样本数: {np.sum(cluster_mask)})")
            print(f"  EPSILON: {cluster_data[:, 0].mean():.3f} ± {cluster_data[:, 0].std():.3f}")
            print(f"  dSdz: {cluster_data[:, 1].mean():.3f} ± {cluster_data[:, 1].std():.3f}")
            print(f"  N²: {cluster_data[:, 2].mean():.3f} ± {cluster_data[:, 2].std():.3f}")

# ==================== 8. 参数敏感性分析 ====================
def parameter_sensitivity_analysis(X_scaled):
    """测试不同的min_samples参数对聚类结果的影响"""
    print("\n" + "="*50)
    print("参数敏感性分析")
    print("="*50)
    
    # 测试不同的min_samples参数
    min_samples_values = [5,8,10, 15, 20, 25, 30]
    results = []
    
    for min_samples in min_samples_values:
        model = OPTICS(min_samples=min_samples, xi=0.05, 
                       min_cluster_size=0.1, n_jobs=-1)
        model.fit(X_scaled)
        labels_test = model.labels_
        n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
        
        # 计算质量指标（仅非噪声点）
        mask_test = labels_test != -1
        if np.sum(mask_test) > 1 and n_clusters_test > 1:
            silhouette_test = metrics.silhouette_score(
                X_scaled[mask_test], labels_test[mask_test]
            )
        else:
            silhouette_test = -1
        
        results.append({
            'min_samples': min_samples,
            'n_clusters': n_clusters_test,
            'n_noise': list(labels_test).count(-1),
            'silhouette': silhouette_test
        })
    
    print("\nmin_samples参数影响:")
    print("min_samples | 聚类数 | 噪声点数 | 轮廓系数")
    print("-" * 40)
    for r in results:
        print(f"{r['min_samples']:^11} | {r['n_clusters']:^6} | {r['n_noise']:^8} | {r['silhouette']:.3f}")

# ==================== 9. 实用函数：提取聚类边界 ====================
def extract_cluster_boundaries(X, labels):
    """
    提取每个聚类的边界信息
    返回的字典键是字符串类型，值转换为Python原生类型
    """
    boundaries = {}
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
            
        cluster_mask = (labels == cluster_id)
        cluster_data = X[cluster_mask]
        
        # 将聚类ID转换为字符串
        cluster_id_str = str(cluster_id)
        
        # 将numpy类型转换为Python原生类型
        boundaries[cluster_id_str] = {
            'EPSILON_min': float(cluster_data[:, 0].min()),
            'EPSILON_max': float(cluster_data[:, 0].max()),
            'EPSILON_mean': float(cluster_data[:, 0].mean()),
            'dSdz_min': float(cluster_data[:, 1].min()),
            'dSdz_max': float(cluster_data[:, 1].max()),
            'dSdz_mean': float(cluster_data[:, 1].mean()),
            'N2_min': float(cluster_data[:, 2].min()),
            'N2_max': float(cluster_data[:, 2].max()),
            'N2_mean': float(cluster_data[:, 2].mean()),
            'n_points': int(len(cluster_data)),
            'volume': float(np.prod([
                cluster_data[:, 0].max() - cluster_data[:, 0].min(),
                cluster_data[:, 1].max() - cluster_data[:, 1].min(),
                cluster_data[:, 2].max() - cluster_data[:, 2].min()
            ]))
        }
    
    return boundaries

# ==================== 10. 保存聚类结果 ====================
def save_clustering_results(X, X_scaled, labels, optics_model, labels_dbscan, output_dir, data_source, clustering_quality):
    """保存聚类结果到文件"""
    print("\n保存聚类结果到文件...")
    
    # 保存原始数据、标准化数据和聚类标签
    results_data = np.column_stack([
        X,  # 原始数据
        X_scaled,  # 标准化数据
        labels,  # OPTICS聚类标签
        labels_dbscan,  # DBSCAN-like聚类标签
        optics_model.reachability_,  # 可达性距离
        optics_model.core_distances_,  # 核心距离
        optics_model.ordering_  # OPTICS排序
    ])
    
    header = ("EPSILON,dSdz,N2,"
              "EPSILON_scaled,dSdz_scaled,N2_scaled,"
              "optics_label,dbscan_label,"
              "reachability,core_distance,ordering")
    
    result_file = os.path.join(output_dir, 'optics_3d_clustering_results.csv')
    np.savetxt(result_file, results_data, 
               delimiter=',', header=header, fmt='%.6f', comments='')
    
    print(f"结果已保存到 '{result_file}'")
    
    # 保存模型
    model_file = os.path.join(output_dir, 'optics_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(optics_model, f)
    print(f"OPTICS模型已保存到 '{model_file}'")
    
    # 计算统计信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(list(labels).count(-1))  # 转换为Python int
    
    # 确保聚类质量指标是Python原生类型
    for key in clustering_quality:
        if clustering_quality[key] is not None:
            clustering_quality[key] = float(clustering_quality[key])
    
    # 保存元数据
    meta_data = {
        'data_source': data_source,
        'n_samples': int(len(X)),  # 转换为Python int
        'n_clusters': int(n_clusters),  # 转换为Python int
        'n_noise': n_noise,  # 已经是Python int
        'parameters': {
            'min_samples': 20,
            'xi': 0.05,
            'min_cluster_size': 0.1
        },
        'clustering_quality': clustering_quality,
        'features': {
            'EPSILON': '湍流动能耗散率 (log10尺度)',
            'dSdz': '盐度梯度 (log10尺度)',
            'N2': '浮力频率平方 (log10尺度)'
        }
    }
    
    meta_file = os.path.join(output_dir, 'analysis_metadata.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
    print(f"元数据已保存到 '{meta_file}'")
    
    # 保存聚类边界为JSON
    if n_clusters > 0:
        cluster_boundaries = extract_cluster_boundaries(X, labels)
        boundaries_file = os.path.join(output_dir, 'cluster_boundaries.json')
        with open(boundaries_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_boundaries, f, indent=2, ensure_ascii=False)
        print(f"聚类边界信息已保存到 '{boundaries_file}'")

# ==================== 11. 主函数 ====================
def main():
    """主执行函数"""
    print("="*50)
    print("海洋微结构数据OPTICS聚类分析")
    print("特征: EPSILON, dSdz, N2")
    print("输出目录: OPTICS/result3")
    print("="*50)
    
    # 创建输出目录
    output_dir = 'OPTICS/result3'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载与提取特征
    print("\n[阶段1] 加载数据与特征提取")
    features = load_and_extract_features('OPTICS/data')
    
    if features is None:
        print("无法加载数据，程序退出。")
        return
    
    data_source = "真实数据"
    if 'depth' in features and 'time' in features:
        print(f"数据包含深度和时间信息")
    
    # 2. 预处理
    print("\n[阶段2] 数据预处理 (对数变换 + 标准化)")
    X_clean, scaler, valid_mask = preprocess_features(features)
    
    # 3. 滑动窗口平均
    print("\n[阶段3] 滑动窗口平均处理")
    X_windows, window_indices = create_moving_average_windows(X_clean, window_size=100, step=50)
    
    # 4. OPTICS聚类
    print("\n[阶段4] OPTICS聚类分析")
    clustering_result = perform_optics_clustering(X_windows, min_samples=10, xi=0.3)
    
    # 5. DBSCAN-like方法聚类
    print("\n使用DBSCAN-like方法提取聚类...")
    optics_dbscan = OPTICS(
        min_samples=15,
        max_eps=2.0,
        metric='euclidean',
        cluster_method='dbscan',
        n_jobs=-1
    )
    optics_dbscan.fit(X_windows)
    labels_dbscan = optics_dbscan.labels_
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    print(f"DBSCAN方法发现聚类数量: {n_clusters_dbscan}")
    
    # 6. 聚类质量评估
    print("\n聚类质量评估:")
    labels = clustering_result['labels']
    mask = labels != -1
    clustering_quality = {}
    
    if np.sum(mask) > 0 and clustering_result['n_clusters'] > 1:
        silhouette = metrics.silhouette_score(X_windows[mask], labels[mask])
        ch_score = metrics.calinski_harabasz_score(X_windows[mask], labels[mask])
        db_score = metrics.davies_bouldin_score(X_windows[mask], labels[mask])
        
        print(f"  轮廓系数: {silhouette:.3f}")
        print(f"  Calinski-Harabasz指数: {ch_score:.1f}")
        print(f"  Davies-Bouldin指数: {db_score:.3f}")
        
        clustering_quality = {
            'silhouette_score': float(silhouette),
            'calinski_harabasz_score': float(ch_score),
            'davies_bouldin_score': float(db_score)
        }
    else:
        print("聚类数不足，无法计算聚类质量指标")
        clustering_quality = {
            'silhouette_score': None,
            'calinski_harabasz_score': None,
            'davies_bouldin_score': None
        }
    
    # 7. 可视化
    print("\n[阶段5] 生成可视化图表")
    create_visualizations(
        X_windows, 
        labels, 
        clustering_result['reachability'], 
        clustering_result['ordering'],
        clustering_result['core_distances'],  # 添加core_distances参数
        clustering_result['n_clusters'],
        output_dir,
        data_source
    )
    
    # 8. 聚类特征分析
    print("\n[阶段6] 聚类特征分析")
    analyze_cluster_statistics(X_windows, labels, clustering_result['n_clusters'])
    
    # 9. 参数敏感性分析
    print("\n[阶段7] 参数敏感性分析")
    parameter_sensitivity_analysis(X_windows)
    
    # 10. 保存结果
    print("\n[阶段8] 保存聚类结果")
    save_clustering_results(
        X_windows, 
        X_windows,  # 注意：这里X_windows已经是标准化后的数据
        labels, 
        clustering_result['model'], 
        labels_dbscan, 
        output_dir, 
        data_source, 
        clustering_quality
    )
    
    print("\n" + "="*50)
    print("代码执行完成！")
    print(f"所有结果已保存到: {output_dir}")
    print("="*50)

# ==================== 12. 脚本入口 ====================
if __name__ == "__main__":
    main()