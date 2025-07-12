import argparse
import yaml
import numpy as np
import torch.nn.functional as F
from processing import load_dataset
from utils import set_seed, evaluate, test
from model import MCGSGCN

import time
import torch
import os
from sklearn.manifold import TSNE
import seaborn as sns
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def edge_index_to_adj_matrix(edge_index, num_nodes):
    """
    将 edge_index 转换为邻接矩阵（稀疏矩阵）
    :param edge_index: 边索引 (2, num_edges)
    :param num_nodes: 节点数
    :return: 邻接矩阵（稀疏矩阵）
    """
    # 使用 torch_sparse 创建稀疏邻接矩阵
    row, col = edge_index
    adj_matrix = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))

    return adj_matrix


# 正则化损失（你可以根据需求选择不同的正则化方式）
def get_regularization_loss(D_diag, reg_lambda=0.01):
    # L2 正则化
    reg_loss = torch.sum(torch.square(D_diag))  # 或者你也可以选择L1正则化: torch.sum(torch.abs(D_diag))
    return reg_lambda * reg_loss

# D 非正元素惩罚函数
def get_positive_penalty(D_diag, penalty_lambda=10.0):
    # 惩罚 D 中小于零的元素
    penalty = torch.sum(torch.clamp(-D_diag, min=0))  # 对小于零的元素施加惩罚
    return penalty_lambda * penalty
def training(args):
    set_seed(args.seed)
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    feature_list, adjs, lis, lis_t, labels, num_classes, train_idx, val_idx, test_idx ,dl= load_dataset(args)
    in_dims =[feat.shape[1] for feat in feature_list]
    all_node = [feat.shape[0] for feat in feature_list]
    # feat_dim = sum(all_node)
    feature_list = [feat.to(device) for feat in feature_list]
    labels = labels.to(device)
    # num_relations = len(adjs)
    '''用于图KAN'''
    # # 存储所有边的 row 和 col
    # rows, cols = [], []
    # # 遍历 adjs 字典，提取 row 和 col
    # for key, sparse_tensor in adjs.items():
    #     row, col, _ = sparse_tensor.coo()  # 获取 COO 格式的 (row, col, val)
    #     rows.append(row)
    #     cols.append(col)
    # # 拼接所有的 row 和 col
    # edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
    # adjm = edge_index_to_adj_matrix(edge_index,feat_dim)

    model = MCGSGCN(in_dims, num_classes, lis, lis_t, args,device,all_node)#,true_feature_list,missing_nodes_list)#,adjm)
    model.to(device)
    #print(model)

    # optimizer
    optimizer = torch.optim.AdamW([
        # feat_project 层参数
        {'params': model.feat_project.parameters(), 'weight_decay': args.wd, 'lr': args.lr},

        # lin1 层参数
        {'params': model.lin1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},

        # lin2 层参数
        {'params': model.lin2.parameters(), 'weight_decay': args.wd, 'lr': args.lr},

        # 正定非交换多项式的参数 W_poly
        {'params': model.W_poly, 'weight_decay': args.prop_wd, 'lr': args.prop_lr},
        # D 的对角参数
        {'params': model.D_module.diag, 'weight_decay': args.wd, 'lr': args.lr},

    ])

    if args.dataset =="IMDB":
        LOSS = torch.nn.BCEWithLogitsLoss()
    else:
        LOSS = F.cross_entropy

    print("start training...")
    best_result_micro = 0
    best_result_macro = 0
    best_epoch = 0
    result_micro=0
    result_macro=0
    best_val_loss=100000000
    best_x = torch.zeros(feature_list[0].shape).to(device)
    num_target = feature_list[0].shape[0]
    model.train()
    t1=time.time()
    cnt = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits,_ = model(adjs, feature_list)
        logits = logits[:num_target]
        loss = LOSS(logits[train_idx], labels[train_idx])

        loss.backward()
        optimizer.step()

        train_micro, train_macro, valid_micro, valid_macro, test_micro, test_macro, val_loss,x_legendre = test(model, adjs, feature_list, labels, train_idx, val_idx, test_idx, args.dataset,LOSS)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result_micro = test_micro
            result_macro = test_macro
            best_epoch = epoch
            best_x = x_legendre
            cnt = 0
        else:
            cnt+= 1
        
        if cnt==args.early_stopping:
            print("Early_stopping!")
            break

        if epoch%50==0:
            print("Epoch {:05d} | train_micro {:.4f} | Train Loss: {:.4f}| valid_micro {:.4f} | valid_loss {:.4f} |test_micro: {:.4f}" .
                format(epoch, train_micro, loss.item(),  valid_micro, val_loss, test_micro))

    print("Best epoch:{} | Test Micro: {:.4f} | Test Macro: {:.4f}".format(best_epoch, result_micro, result_macro))
    t2=time.time()

    print(f'one experiments stage finish , use {t2-t1} time')
    '''分类结果可视化'''
    # # t-sne画图
    # original_features_np = feature_list[0][test_idx].cpu().numpy()
    # completed_features_np = best_x[:num_target][test_idx].detach().cpu().numpy()
    # os.makedirs("embedding_Visualization", exist_ok=True)
    # # def load_original_labels(file_path):
    # #     """
    # #     从原始标签文件中读取节点标签信息，并返回一个字典
    # #     """
    # #     node_labels = {}
    # #
    # #     with open(file_path, 'r', encoding='utf-8') as f:
    # #         for line in f:
    # #             # 解析每一行数据
    # #             parts = line.strip().split('\t')
    # #             node_id = int(parts[0])  # 节点ID
    # #             # node_name = parts[1]  # 节点名称（不用于标签提取）
    # #             # node_type = int(parts[2])  # 节点类型
    # #             node_label = int(parts[3])  # 节点标签
    # #
    # #             # 将节点标签存入字典
    # #             node_labels[node_id] = node_label
    # #
    # #     return node_labels
    # # 用于多标签
    # ''''''
    # def load_original_labels(file_path):
    #     """
    #     从原始标签文件中读取节点标签信息，并返回一个字典。
    #     支持多标签节点（如标签 "2,3"）的处理。
    #     """
    #     node_labels = {}
    #
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             parts = line.strip().split('\t')
    #             node_id = int(parts[0])
    #             raw_labels = parts[3]  # 可能是 "2", "2,3", "1,4,5" 等
    #
    #             # 支持逗号分隔的多标签
    #             labels = [int(label) for label in raw_labels.split(',')]
    #             node_labels[node_id] = labels
    #
    #     return node_labels
    #
    # def sort_labels_by_node_id(file_path):
    #     """
    #     加载原始标签文件并根据节点 ID 从小到大排序
    #     """
    #     original_labels = load_original_labels(file_path)
    #
    #     # 按照节点 ID（字典的键）从小到大排序
    #     sorted_labels = {k: original_labels[k] for k in sorted(original_labels.keys())}
    #
    #     # 转换为与 test_feature 对应的标签数组
    #     labels_array = list(sorted_labels.values())
    #     return labels_array
    # #
    # original_labels = sort_labels_by_node_id(os.path.join(dl.path, 'label.dat.test'))
    # #
    # # def plot_tsne(features, labels, title, filename):
    # #     tsne = TSNE(n_components=2, random_state=42)
    # #     reduced_features = tsne.fit_transform(features)
    # #
    # #     plt.figure(figsize=(8, 6))
    # #     sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette="viridis", s=60,
    # #                     legend='full')
    # #     plt.title(title)
    # #     plt.xlabel("t-SNE 1")
    # #     plt.ylabel("t-SNE 2")
    # #     plt.legend(title='Class')
    # #     plt.savefig(filename, format='pdf', bbox_inches="tight")
    # #     plt.close()  # 避免占用内存
    # #     # plt.show()
    # #
    # # plot_tsne(original_features_np, original_labels, "t-SNE: Original Node Features (Before Completion)",
    # #           f"Original_Node_Features_information_Vision_{args.seed}_{args.dataset}.pdf")
    # #
    # # # 2. 使用补全后的特征和标签的可视化
    # # plot_tsne(completed_features_np, original_labels, "t-SNE: Completed Node Features (After Completion)",
    # #           f"Completed_Node_Features_information_Vision_{args.seed}_{args.dataset}.pdf")
    # def expand_features_for_multilabel(features, labels):
    #     expanded_features = []
    #     expanded_labels = []
    #
    #     for feat, label_list in zip(features, labels):
    #         for label in label_list if isinstance(label_list, list) else [label_list]:
    #             expanded_features.append(feat)
    #             expanded_labels.append(label)
    #
    #     return np.array(expanded_features), expanded_labels
    #
    # def plot_tsne_dualview(original_features, completed_features, labels, filename_prefix):
    #     """
    #     兼容特征维度不同的双视角 t-SNE，可用于原始稀疏+补全后稠密特征的对比。
    #     """
    #     ori_feat_exp, ori_labels_exp = expand_features_for_multilabel(original_features, labels)
    #     comp_feat_exp, comp_labels_exp = expand_features_for_multilabel(completed_features, labels)
    #
    #     # 各自独立做 t-SNE
    #     reduced_ori = TSNE(n_components=2, random_state=42).fit_transform(ori_feat_exp)
    #     reduced_comp = TSNE(n_components=2, random_state=42).fit_transform(comp_feat_exp)
    #
    #     # 绘图
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #
    #     sns.scatterplot(
    #         x=reduced_ori[:, 0], y=reduced_ori[:, 1],
    #         hue=ori_labels_exp, palette="viridis", s=60,
    #         legend='full', ax=axes[0]
    #     )
    #     axes[0].set_title("t-SNE Visualization of Raw Input Embedding")
    #     # axes[0].set_xlabel("t-SNE 1")
    #     # axes[0].set_ylabel("t-SNE 2")
    #
    #     sns.scatterplot(
    #         x=reduced_comp[:, 0], y=reduced_comp[:, 1],
    #         hue=comp_labels_exp, palette="viridis", s=60,
    #         legend='full', ax=axes[1]
    #     )
    #     axes[1].set_title("t-SNE Visualization of Final Learned Embedding")
    #     # axes[1].set_xlabel("t-SNE 1")
    #     # axes[1].set_ylabel("t-SNE 2")
    #
    #     # 图例统一放右侧
    #     handles, labels = axes[1].get_legend_handles_labels()
    #     fig.legend(handles, labels, title="Class", loc="center right", bbox_to_anchor=(1.08, 0.5))
    #     plt.tight_layout()
    #     plt.savefig(f"embedding_Visualization/{filename_prefix}_DualView_TSNE.pdf", bbox_inches="tight")
    #     plt.close()
    #
    # plot_tsne_dualview(
    #     original_features_np,
    #     completed_features_np,
    #     original_labels,
    #     filename_prefix=f"Node_Features_Compare_Seed{args.seed}_{args.dataset}"
    # )


    '''消融实验可视化分析加路径协同图分析阶段'''
    # def analyze_path_importance_improved(model, args):
    #     """
    #     改进的路径重要性分析可视化
    #     """
    #     print("==> Analyzing learned path attention weights...")
    #     os.makedirs("Importance_visualizations", exist_ok=True)
    #
    #     all_weights = []
    #     path_names = []
    #     max_len = 0
    #
    #     # Step 1: 收集所有路径权重并确定最大长度
    #     for k, module in model.path_weight_modules1.items():
    #         weights = module().detach().cpu().numpy()
    #         max_len = max(max_len, len(weights))
    #         all_weights.append(weights)
    #         # 改进：更清晰的路径标签
    #         path_names.append(f'{k}-order meta-path group')
    #
    #     # Step 2: 用NaN补齐，使所有路径的权重向量长度一致
    #     padded_weights = []
    #     for weights in all_weights:
    #         padded = np.pad(weights, (0, max_len - len(weights)), 'constant', constant_values=np.nan)
    #         padded_weights.append(padded)
    #
    #     # Step 3: 转换为矩阵
    #     weight_matrix = np.vstack(padded_weights)
    #
    #     # Step 4: 创建改进的热图
    #     fig, ax = plt.subplots(figsize=(14, 8))  # 增大图片尺寸以便更好显示
    #
    #     # 改进：使用更好的颜色映射，提高高值区域的对比度
    #     # 使用自定义颜色映射或调整现有映射
    #     from matplotlib.colors import LinearSegmentedColormap
    #
    #     # 创建自定义颜色映射，提高高值区域的对比度
    #     colors = ['#440154', '#3b528b', '#21908c', '#5dc863', '#fde725']  # viridis的关键颜色
    #     n_bins = 100
    #     custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors, N=n_bins)
    #
    #     # 或者使用plasma色彩映射，对高值更敏感
    #     im = ax.imshow(weight_matrix, aspect='auto', cmap='plasma', interpolation='nearest')
    #
    #     # 改进：更清晰的x轴标签
    #     # 只显示有意义的x轴刻度
    #     valid_indices = []
    #     for i in range(max_len):
    #         if not np.all(np.isnan(weight_matrix[:, i])):
    #             valid_indices.append(i)
    #
    #     # 设置x轴刻度
    #     if max_len <= 20:
    #         # 如果路径数量较少，显示所有索引
    #         ax.set_xticks(range(max_len))
    #         ax.set_xticklabels([f'P{i + 1}' for i in range(max_len)], rotation=45)
    #     else:
    #         # 如果路径数量较多，只显示关键位置
    #         step = max(1, max_len // 20)
    #         tick_positions = list(range(0, max_len, step))
    #         if tick_positions[-1] != max_len - 1:
    #             tick_positions.append(max_len - 1)
    #         ax.set_xticks(tick_positions)
    #         ax.set_xticklabels([f'P{i + 1}' for i in tick_positions], rotation=45)
    #
    #     # 改进：更清晰的y轴标签
    #     ax.set_yticks(range(len(path_names)))
    #     ax.set_yticklabels(path_names, fontsize=12)
    #     ax.set_xlabel('Meta-path Index', fontsize=12)
    #     ax.set_ylabel('Meta-path Groups', fontsize=12)
    #     ax.set_title('Path Importance (Learned Attention Weights for Each Meta-path)',
    #                  fontsize=14, fontweight='bold', pad=20)
    #
    #     # 改进：更好的颜色条
    #     cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    #     cbar.set_label('Attention Weight', fontsize=12)
    #     cbar.ax.tick_params(labelsize=10)
    #
    #     # 添加数值标注（可选，当矩阵不太大时）
    #     if weight_matrix.shape[0] <= 10 and weight_matrix.shape[1] <= 20:
    #         for i in range(weight_matrix.shape[0]):
    #             for j in range(weight_matrix.shape[1]):
    #                 if not np.isnan(weight_matrix[i, j]):
    #                     text = ax.text(j, i, f'{weight_matrix[i, j]:.3f}',
    #                                    ha="center", va="center",
    #                                    color="white" if weight_matrix[i, j] < 0.1 else "black",
    #                                    fontsize=8)
    #
    #     # 改进：添加网格线以便更好区分
    #     ax.set_xticks(np.arange(-0.5, max_len, 1), minor=True)
    #     ax.set_yticks(np.arange(-0.5, len(path_names), 1), minor=True)
    #     ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    #
    #     plt.tight_layout()
    #
    #     # 保存图片
    #     plt.savefig(f'Importance_visualizations/{args.seed}_improved_path_weights_heatmap.pdf',
    #                 dpi=300, bbox_inches='tight')
    #     # plt.savefig(f'Importance_visualizations/{args.seed}_improved_path_weights_heatmap.png',
    #     #             dpi=300, bbox_inches='tight')
    #     plt.show()
    #
    #     # 额外分析：打印每个路径组的统计信息
    #     print("\n==> Path Group Statistics:")
    #     for i, (name, weights) in enumerate(zip(path_names, all_weights)):
    #         print(f"{name}:")
    #         print(f"  - Number of paths: {len(weights)}")
    #         print(f"  - Max weight: {np.max(weights):.4f}")
    #         print(f"  - Min weight: {np.min(weights):.4f}")
    #         print(f"  - Mean weight: {np.mean(weights):.4f}")
    #         print(f"  - Std weight: {np.std(weights):.4f}")
    #         print()
    #
    # # 使用方法：
    # analyze_path_importance_improved(model, args)
    #
    # def create_separate_heatmaps_by_order(model, args):
    #     """
    #     为每个阶数创建单独的热图，以便更好地观察同阶内的差异
    #     """
    #     print("==> Creating separate heatmaps for each order...")
    #     os.makedirs("Importance_visualizations/by_order", exist_ok=True)
    #
    #     for k, module in model.path_weight_modules1.items():
    #         weights = module().detach().cpu().numpy()
    #
    #         # 创建单独的热图
    #         fig, ax = plt.subplots(figsize=(16, 3))
    #
    #         # 将权重重塑为单行矩阵以便显示
    #         weight_matrix = weights.reshape(1, -1)
    #
    #         # 使用更敏感的颜色映射
    #         im = ax.imshow(weight_matrix, aspect='auto', cmap='RdYlBu_r')
    #
    #         # 设置标签（仅显示部分，避免拥挤）
    #         xtick_step = max(len(weights) // 20, 1)  # 每隔一定数量显示一个标签
    #         ax.set_xticks(range(0, len(weights), xtick_step))
    #         ax.set_xticklabels([f'P{i + 1}' for i in range(0, len(weights), xtick_step)],
    #                            rotation=45, ha='right', fontsize=12)
    #         # 强制水平对齐标签
    #         ax.tick_params(axis='x', labelrotation=0)
    #         # # 设置标签
    #         # ax.set_xticks(range(len(weights)))
    #         # ax.set_xticklabels([f'P{i + 1}' for i in range(len(weights))])
    #         ax.set_yticks([0])
    #         ax.set_yticklabels([f'{k}-order paths'])
    #         ax.set_xlabel('Meta-path Index within Group')
    #         ax.set_title(f'{k}-order Meta-path Group - Attention Weights Distribution')
    #         if int(k) <= 3:
    #             # 添加数值标注
    #             for j in range(len(weights)):
    #                 ax.text(j, 0, f'{weights[j]:.5f}', ha="center", va="center",
    #                         color="white" if weights[j] < np.mean(weights) else "black")
    #
    #         # 颜色条
    #         cbar = plt.colorbar(im, ax=ax)
    #         cbar.set_label('Attention Weight')
    #
    #         plt.tight_layout()
    #         plt.savefig(f'Importance_visualizations/by_order/{args.seed}_order_{k}_weights.pdf',
    #                     dpi=300, bbox_inches='tight')
    #         # plt.savefig(f'Importance_visualizations/by_order/{args.seed}_order_{k}_weights.png',
    #         #             dpi=300, bbox_inches='tight')
    #         plt.show()
    #
    # # 使用方法：
    # create_separate_heatmaps_by_order(model, arg

    '''1. 可视化 D 的对角元素分布'''
    # # 假设 model.D.diag 是 D 的对角元素
    # d_values = model.D_module.diag.detach().cpu().numpy()  # 转为 NumPy 数组，方便处理
    # # 创建保存文件夹
    # os.makedirs("visualizations", exist_ok=True)
    # # 绘制 D 的对角元素分布直方图
    # plt.figure(figsize=(8, 6))
    # plt.hist(d_values, bins=50, alpha=0.7, color='blue')
    # plt.title('Distribution of $\Phi$\'s Diagonal Elements')
    # plt.xlabel('Weight')#表示 D 的对角元素的值，即每个节点的权重值
    # plt.ylabel('Frequency')#表示落在某个值区间（bin）内的 节点数目
    #
    # # 保存图像
    # plt.savefig(f"visualizations/{args.dataset}_{args.seed}_d_distribution.pdf")
    # plt.close()  # 关闭当前图像
    # '''2. 保存每个节点的权重图'''
    # # 绘制每个节点的权重
    # plt.figure(figsize=(10, 6))
    # # plt.bar(range(len(d_values)), d_values, color='#1f77b4')  # 使用深蓝色，去掉透明度
    # plt.bar(range(len(d_values)), d_values, color='blue', alpha=0.7)
    # plt.title('Node-wise Weights in $\Phi$')
    # plt.xlabel('Node Index')
    # plt.ylabel('Weight')
    #
    # # 保存图像
    # plt.savefig(f"visualizations/{args.dataset}_{args.seed}_node_weights.pdf")
    # plt.close()  # 关闭当前图像
    # '''3.保存高权重节点的分析'''
    # # 找出权重最大的前10个节点
    # top_nodes = sorted(range(len(d_values)), key=lambda i: d_values[i], reverse=True)[:10]
    #
    # # 打印出这些节点的索引及其对应的权重
    # top_nodes_info = []
    # for node in top_nodes:
    #     top_nodes_info.append(f"Node {node}: Weight {d_values[node]}")
    #
    # # 将高权重节点的索引和权重保存到文本文件
    # with open(f"visualizations/{args.dataset}_{args.seed}_top_weighted_nodes.txt", "w") as f:
    #     f.write("Top 10 nodes with highest weights in D:\n")
    #     for node_info in top_nodes_info:
    #         f.write(f"{node_info}\n")




    return result_micro, result_macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCGSGCN')
    parser.add_argument("--dataset", type=str, default="IMDB", help="The dataset used.")
    parser.add_argument("--seed", type=int, default=1,help="The seed used in the training.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run", type=int, default=5,help="The run times.")

    parser.add_argument("--epochs", type=int, default=1000)#1000
    parser.add_argument("--early_stopping",type=int,default=100)#100
    parser.add_argument("--emb_dim", type=int, default=64)#原本是64
    parser.add_argument("--hidden", type=int, default=32)#原本是32
    parser.add_argument("--dropout", type=float, default=0)#0
    parser.add_argument("--input_drop", type=float, default=0)#0
    parser.add_argument("--K", type=int, default=5)#5
    parser.add_argument("--lr", type=float, default=0.01)#0.01
    parser.add_argument("--wd", type=float, default=5e-4)#1.0
    parser.add_argument("--prop_lr", type=float, default=0.01)#0.01
    parser.add_argument("--prop_wd", type=float, default=5e-4)#1.0
    parser.add_argument("--threshold", type=float, default=0.1)#默认0.1->1
    parser.add_argument("--gamma", type=float, default=1)  # 默认1,【0,1,1.5,2,3】DBLP，IMDB，AMiner设置为1，ACM设置为1.5
    args = parser.parse_args()


    with open("Finally_test.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    args.emb_dim = config[args.dataset]["emb_dim"]
    args.hidden = config[args.dataset]["hidden"]
    args.dropout = config[args.dataset]["dropout"]
    args.input_drop = config[args.dataset]["input_drop"]
    args.K = config[args.dataset]["K"]
    args.lr = config[args.dataset]["lr"]
    args.wd = config[args.dataset]["wd"]
    args.prop_lr = config[args.dataset]["prop_lr"]
    args.prop_wd = config[args.dataset]["prop_wd"]
    args.threshold = config[args.dataset]["threshold"]
    args.gamma = config[args.dataset]["gamma"]
    print(args)

    results = []
    for i in range(1,args.run+1):
        args.seed = i
        result_micro, result_macro = training(args)
        results.append([result_micro,result_macro])

    micro_mean, macro_mean = np.mean(results, axis=0) * 100
    micro_std, macro_std = np.sqrt(np.var(results, axis=0)) * 100
    print(f'SHGCN on dataset {args.dataset}, in {args.run} repeated experiment:')
    print(f'Micro mean = {micro_mean:.4f} ± {micro_std:.4f}  \t  Macro mean =  {macro_mean:.4f} ± {macro_std:.4f}')




