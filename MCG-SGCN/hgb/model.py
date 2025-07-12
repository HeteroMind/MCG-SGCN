import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_sparse import matmul as torch_sparse_matmul
import numpy as np
from collections import defaultdict
from torch_geometric.nn import GATConv
# from torch_geometric.nn import GCNConv  # 引入 GCNConv
from torch_geometric.nn import GATv2Conv
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class LearnableD(nn.Module):
    def __init__(self, num_nodes, device):
        super().__init__()
        self.diag = nn.Parameter(torch.full((num_nodes,), 1.0, device=device))  # 可学习向量

    # def forward(self):
    #     return torch.diag(F.softplus(self.diag))  # 保证正值
    #
    # def apply_to(self, x):
    #     d_pos = F.softplus(self.diag)
    #     return d_pos.unsqueeze(1) * x
    def forward(self):
        # 检查 D 的对角元素是否有非正值
        if torch.any(self.diag <= 0):  # 检查是否有元素小于或等于0
            print("Warning: D contains non-positive elements!")
        # 构造稠密对角矩阵，仅在需要完整矩阵时使用
        return torch.diag(self.diag)

    def apply_to(self, x):
        # 推荐方式：高效实现 D @ x，其中 D 是对角矩阵（不显示构造）
        return self.diag.unsqueeze(1) * x



    # def apply_to(self, x):
    #     # 先对 D 的对角元素进行 ReLU 激活
    #     relu_diag = F.relu(self.diag).unsqueeze(1)  # shape: (N, 1)
    #     return relu_diag * x  # 相当于 D @ x，逐行缩放
class MCGSGCN(nn.Module):
    def __init__(self, in_dims, num_classes, lis, lis_t, args, device,all_nodes,bns=False):#edge_index,adjm, bns=False):
        super(MCGSGCN, self).__init__()
        self.in_dims = in_dims
        self.num_classes = num_classes
        self.emb_dim = args.emb_dim
        self.h_dim = args.hidden
        self.K = args.K
        self.lis = lis
        self.lis_t = lis_t
        self.device = device
        self.args =args
        self.all_nodes =all_nodes
        # 特征投影层
        self.feat_project = nn.ModuleList([nn.Linear(in_dim, self.emb_dim, bias=bns).to(device) for in_dim in self.in_dims])
        self.lin1 = nn.Linear(self.emb_dim, self.h_dim).to(device)
        self.lin2 = nn.Linear(self.h_dim, self.num_classes).to(device)

        # Dropout层
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

        # 初始化正定非交换多项式系数
        self.W_poly = nn.Parameter(torch.tensor(self.init_coe(), device=device))
        self.D_module = LearnableD(num_nodes=sum(all_nodes), device=torch.device("cuda"))

        # 预先创建 PathWeightModule
        self.path_weight_modules1 = nn.ModuleDict({
            str(k): PathWeightModule(paths, self.emb_dim, self.h_dim, device,args.threshold,k,args.gamma,args) for k, paths in lis.items()
        }).to(device)
        self.path_weight_modules2 = nn.ModuleDict({
            str(k): PathWeightModule(paths, self.emb_dim, self.h_dim, device,args.threshold,k,args.gamma,args) for k, paths in lis_t.items()
        }).to(device)




    def init_coe(self):
        coe_num = 1 + len(self.lis[1])
        for i in range(2, self.K + 1):
            coe_num += len(self.lis[i])
        bound = np.sqrt(3.0 / coe_num)
        TEMP = np.random.uniform(-bound, bound, coe_num)
        TEMP = TEMP / np.sum(np.abs(TEMP))
        return TEMP

    def normalize(self, x):
        means = x.mean(1, keepdim=True)
        deviations = x.std(1, keepdim=True)
        x = (x - means) / deviations
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        return x

    '''消融实验，三因子相乘
    def apply_psd_nc_poly1(self, adjs, x, paths):
        res = torch.zeros_like(x, device=self.device)
        temp_lst = {}
        for k in range(self.K + 1):
            coef = self.W_poly[k]  # θ_k
            if k == 0:
                res = res + coef * x  # θ_0 * I * x
            else:
                path_weight_module = self.path_weight_modules1[str(k)]
                path_weights = path_weight_module()  # η_k^{(r)} * δ_k^{(r)}
                temp_now = {}
                for idx, path in enumerate(paths[k]):
                    path_key = f"{path}"
                    if k == 1:
                        out = torch_sparse_matmul(adjs[path], x)

                    else:
                        prev_path = path[2:]
                        out = torch_sparse_matmul(adjs[path[:2]], temp_lst[prev_path])


                    # # 关键：三因子乘法
                    res = res + coef * path_weights[idx] * out
                    temp_now[path_key] = out
                temp_lst = temp_now
        return res

    def apply_psd_nc_poly2(self, adjs, x, paths):
        res = torch.zeros_like(x, device=self.device)
        temp_lst = {}
        for k in range(self.K + 1):
            coef = self.W_poly[k]  # θ_k
            if k == 0:
                res = res + coef * x  # θ_0 * I * x
            else:
                path_weight_module = self.path_weight_modules2[str(k)]
                path_weights = path_weight_module()  # η_k^{(r)} * δ_k^{(r)}
                temp_now = {}
                for idx, path in enumerate(paths[k]):
                    path_key = f"{path}"
                    if k == 1:
                        out = torch_sparse_matmul(adjs[path], x)

                    else:
                        prev_path = path[2:]
                        out = torch_sparse_matmul(adjs[path[:2]], temp_lst[prev_path])

                    # # 关键：三因子乘法
                    res = res + coef * path_weights[idx] * out

                    temp_now[path_key] = out
                temp_lst = temp_now
        return res

    '''

    #最终版本
    def apply_psd_nc_poly1(self, adjs, x, paths):
        res = torch.zeros_like(x, device=self.device)
        temp_lst = {}
        for k in range(self.K + 1):
            coef = self.W_poly[k]
            if k == 0:
                res = res + coef * x
                # res = res +  x #对全局权重的消融
            else:
                path_weight_module = self.path_weight_modules1[str(k)]
                path_weights = path_weight_module()
                temp_now = {}
                for idx, path in enumerate(paths[k]):
                    path_key = f"{path}"
                    if k == 1:
                        out = torch_sparse_matmul(adjs[path], x)
                        res = res + path_weights[idx] * out
                    else:
                        prev_path = path[2:]
                        out = torch_sparse_matmul(adjs[path[:2]], temp_lst[prev_path])
                        res = res + path_weights[idx] * out
                    res = res + coef * out#原本

                    # res = res +out #对全局权重的消融

                    temp_now[path_key] = out
                temp_lst = temp_now
        return res

    def apply_psd_nc_poly2(self, adjs, x, paths):
        res = torch.zeros_like(x, device=self.device)
        temp_lst = {}
        for k in range(self.K + 1):
            coef = self.W_poly[k]
            if k == 0:
                res = res + coef * x
                # res = res + x
            else:
                path_weight_module = self.path_weight_modules2[str(k)]
                path_weights = path_weight_module()
                temp_now = {}
                for idx, path in enumerate(paths[k]):
                    path_key = f"{path}"
                    if k == 1:
                        out = torch_sparse_matmul(adjs[path], x)
                        res = res + path_weights[idx] * out
                    else:
                        prev_path = path[2:]
                        out = torch_sparse_matmul(adjs[path[:2]], temp_lst[prev_path])
                        res = res + path_weights[idx] * out

                    res = res + coef * out
                    # res = res +  out
                    temp_now[path_key] = out
                temp_lst = temp_now
        return res
    '''这是另外的谱卷积操作思路，不是F的转置乘以D乘以F了'''


    '''
    #这是对多项式系数做的消融，去掉self.W_poly的模块版本'''
    # def apply_psd_nc_poly2(self, adjs, x, paths):
    #     res = torch.zeros_like(x, device=self.device)
    #     temp_lst = {}
    #     for k in range(self.K + 1):
    #         # coef = self.W_poly[k]
    #         if k == 0:
    #             # res = res + coef * x
    #             res = res +  x
    #         else:
    #             path_weight_module = self.path_weight_modules2[str(k)]
    #             path_weights = path_weight_module()
    #             temp_now = {}
    #             for idx, path in enumerate(paths[k]):
    #                 path_key = f"{path}"
    #                 if k == 1:
    #                     out = torch_sparse_matmul(adjs[path], x)
    #                     res = res + path_weights[idx] * out
    #                 else:
    #                     prev_path = path[2:]
    #                     out = torch_sparse_matmul(adjs[path[:2]], temp_lst[prev_path])
    #                     res = res + path_weights[idx] * out
    #                 # res = res + coef* out
    #                 res = res +  out
    #                 temp_now[path_key] = out
    #             temp_lst = temp_now
    #     return res

    def forward(self, adjs, features_list):
        # 处理特征
        output = [self.input_drop(lin(feature)) for lin, feature in zip(self.feat_project, features_list)]
        x = torch.cat(output, 0).to(self.device)  # 移动到设备

        # 通过第一层线性变换
        x = F.relu(self.lin1(x))
        x = self.normalize(x)
        x = self.dropout(x)
        #Filter
        x_poly = self.apply_psd_nc_poly1(adjs, x, self.lis)#x_poly
        x_poly1 = self.D_module.apply_to(x_poly)
        x_legendre  = self.apply_psd_nc_poly2(adjs, x_poly1, self.lis_t)

        # 通过第二层线性变换
        res = self.lin2(x_legendre)
        return res, x_legendre



class PathEmbedding(nn.Module):
    def __init__(self, num_paths, emb_dim, device):
        super().__init__()
        self.path_emb = nn.Parameter(torch.empty(num_paths, emb_dim).to(device))
        nn.init.kaiming_normal_(self.path_emb, nonlinearity='relu')
        # self.path_emb = nn.Parameter(torch.randn(num_paths, emb_dim).to(device))

    def forward(self):
        return self.path_emb



class PathInteractionGNN(nn.Module):
    def __init__(self, num_paths, emb_dim, hidden_dim, device, num_heads=3, edge_dim=1):
        super().__init__()
        self.embedding = PathEmbedding(num_paths, emb_dim, device)

        self.sim_fusion = SimFusionModule(edge_dim)

        self.gnn = GATv2Conv(
            emb_dim,
            hidden_dim,
            heads=num_heads,
            concat=False,
            add_self_loops=False,
            edge_dim=edge_dim  # ✅ 告诉 GATv2 边的特征维度
        )
        # self.gnn = GATConv(emb_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, edge_index, edge_attr):
        fused_attr = self.sim_fusion(edge_attr)
        x = self.embedding()
        # 将边权作为 attention 权重调节项之一（GAT 不直接用 edge_attr，需要手动嵌入进 feature）
        # 可选：使用边权作为额外特征输入 GNN
        # x = self.gnn(x, edge_index)  # 暂未融合 edge_attr，但可扩展为 GATv2 或自定义 GNN
        x = self.gnn(x, edge_index, edge_attr=fused_attr)  # 直接传递 edge_attr
        scores = self.linear(x).squeeze()
        weights = torch.softmax(scores, dim=0)
        return weights

class PathWeightModule(nn.Module):
    def __init__(self, path_list, emb_dim, hidden_dim, device, threshold,k,gamma,args):
        super().__init__()
        self.path_list = path_list
        self.device = device
        self.args = args

        # 设定阶数的初始权重 (假设 path_list 是按阶数排序的),这里可根据消融实验进行分析！！说明我们的先验是对的！！
        # 模型的可视化化知道了模型去掉了k之后，模型关注二阶路径的权重上升了，但是指标结果下降了！！用这个
        path_order = self.get_path_order(path_list,k)
        #第一种：！！不知为什么DBLP不适合加上归一化操作，而ACM适合加上归一化操作！！
        if args.dataset in ["IMDB", "DBLP","AMiner"]:
            self.k_weight = nn.Parameter(torch.tensor([1.0 / (order ** gamma) for order in path_order]).to(device))
        else:
            # 第一种的变体
            weights = torch.tensor([1.0 / (order ** gamma) for order in path_order], dtype=torch.float32).to(device)
            # # 加上归一化：总和为 1  ，！！不知为什么DBLP不适合加上归一化操作，而ACM适合加上归一化操作！！
            weights = weights / weights.sum()
            # 设置为可学习参数
            self.k_weight = nn.Parameter(weights)

        # self.k_weight = nn.Parameter(torch.ones(len(path_list)).to(device))#最开始的
        # #第二种：对数，效果不太好
        # # 当前调用的路径阶数 k，例如 1、3、22
        # current_k = path_order[0]  # 因为都是一致的，取第一个即可
        # log_weight = 1.0 / torch.log(torch.tensor(current_k + 1.0))
        #
        # # 构造权重向量，维度与路径个数一致（假设有 N 个路径）
        # N = len(path_order)
        # k_weight_init = torch.full((N,), log_weight.item(), dtype=torch.float32).to(device)
        # #第三种：多项式倒数衰减
        #
        # orders = torch.tensor(path_order, dtype=torch.float32).to(device)
        # poly_decay = 1 / (orders ** gamma)
        # k_weight_init = poly_ decay / torch.sum(poly_decay)
        #
        # 设置为可学习参数
        # self.k_weight = nn.Parameter(k_weight_init).to(device)


        #第一种self.edge_index
        # self.edge_index = build_path_edge_index(path_list, threshold)
        # # 改为返回 edge_index 和 edge_attr
        # self.edge_index, self.edge_attr = build_path_edge_index(path_list, threshold=threshold)
        #第三种：
        jaccard_sim, lcs_sim, pos_sim = precompute_all_sim_matrices(path_list,device)
        self.edge_index, self.edge_attr = build_multi_sim_edge_graph(
            [jaccard_sim, lcs_sim, pos_sim],
            threshold=threshold
        )
        self.gnn_module = PathInteractionGNN(len(path_list), emb_dim, hidden_dim, device)

    def get_path_order(self, path_list,k):
        """
        根据传入的 path_list 确定每个路径的阶数。
        例如，路径 'AP' 是阶数 1，路径 'APPT' 是阶数 2，等等。
        """
        path_order = []
        for path in path_list:
            # 获取路径的阶数（这里通过路径长度减去1来判断阶数）
            # 假设路径长度是阶数+1, 即 'AP' 长度为2，则阶数为1
            ##这里有点实验上的发现，可作为消融，到底是len(path)-1（现在所测得到的数据都是用的这个），还是len(path)-k
            path_order.append(len(path)-1)
        return path_order
    def forward(self):
        edge_index = self.edge_index.to(self.device)
        edge_attr = self.edge_attr.to(self.device)
        path_weights = self.gnn_module(edge_index, edge_attr)
        # return path_weights#消融实验
        # return self.k_weight#消融实验
        return self.k_weight * path_weights


def jaccard_similarity(path1, path2):
    set1 = set(path1)
    set2 = set(path2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def lcs_similarity(path1, path2):
    m, n = len(path1), len(path2)
    dp = np.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if path1[i - 1] == path2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_length = dp[m][n]
    return 2 * lcs_length / (m + n) if m + n != 0 else 0

def position_similarity(path1, path2):
    matching_positions = sum([1 if path1[i] == path2[i] else 0 for i in range(min(len(path1), len(path2)))])
    max_len = max(len(path1), len(path2))
    return matching_positions / max_len if max_len != 0 else 0

def compute_similarities(path1, path2):
    return {
        'jaccard': jaccard_similarity(path1, path2),
        'lcs': lcs_similarity(path1, path2),
        'position': position_similarity(path1, path2)
    }

# 预先计算所有相似度矩阵（num_paths x num_paths）
def precompute_all_sim_matrices(path_list,device):
    num_paths = len(path_list)
    jaccard_sim = torch.zeros((num_paths, num_paths)).to(device)
    lcs_sim = torch.zeros((num_paths, num_paths)).to(device)
    pos_sim = torch.zeros((num_paths, num_paths)).to(device)

    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            sims = compute_similarities(path_list[i], path_list[j])  # 返回 dict
            jaccard_sim[i, j] = jaccard_sim[j, i] = sims['jaccard']
            lcs_sim[i, j] = lcs_sim[j, i] = sims['lcs']
            pos_sim[i, j] = pos_sim[j, i] = sims['position']

    return jaccard_sim, lcs_sim, pos_sim
def build_multi_sim_edge_graph(sim_matrices, threshold):
    """
    输入多个相似度矩阵（如 Jaccard、LCS、Position），构建联合边，并保留多维 edge_attr。
    """
    num_paths = sim_matrices[0].size(0)
    edges = []
    edge_feats = []

    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            sim_vec = [sim[i, j].item() for sim in sim_matrices]
            sim_score = sum(sim_vec)
            if sim_score >= threshold:
                edges.append([i, j])
                edges.append([j, i])
                edge_feats.append(sim_vec)
                edge_feats.append(sim_vec)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)  # shape: [num_edges, num_sims]
    return edge_index, edge_attr
class SimFusionModule(nn.Module):
    def __init__(self, num_sims=3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_sims)).to(device)  # 初始全1，可学习

    def forward(self, edge_attr):
        weights = torch.softmax(self.weights, dim=0)  # 保证是加权组合
        combined = (edge_attr * weights).sum(dim=1, keepdim=True)  # shape: [num_edges, 1]
        return combined
def build_path_edge_index(path_list, threshold, sim_weights=None):
    """
    构建多相似度联合加权图：
    - sim_weights: dict, 形如 {'jaccard': 0.4, 'lcs': 0.4, 'position': 0.2}
    """
    if sim_weights is None:
        sim_weights = {'jaccard': 0.4, 'lcs': 0.4, 'position': 0.2}

    num_paths = len(path_list)
    edges = []
    weights = []

    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            sims = compute_similarities(path_list[i], path_list[j])
            sim_score = sum([sims[k] * sim_weights[k] for k in sim_weights])
            if sim_score >= threshold:
                edges.append([i, j])
                edges.append([j, i])
                weights.append(sim_score)
                weights.append(sim_score)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_attr

