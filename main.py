# coding=utf-8
import networkx as nx
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from func.util import *
from sklearn.metrics import mutual_info_score
from func.synthetic import generate_evolution, generate_evolution2, \
    generate_evolution3

from tqdm import tqdm  # 进度条功能


def param_update(X, A, Y, W, alpha):
    W_apprx = X * A * X.T
    N, M = Y.shape
    X_new, A_new = np.zeros(X.shape), np.zeros(A.shape)

    print("param update...")
    for k in range(M):
        print(f"({k+1}/{M})")  # 打印进度
        for i in tqdm(range(N)):  # 添加了进度条显示
            for j in range(N):
                X_new[i, k] += W[i, j] * A[k, k] * X[j, k] / W_apprx[i, j]
                A_new[k, k] += W[i, j] * X[i, k] * X[j, k] / W_apprx[i, j]
            X_new[i, k] *= (2 * alpha * X[i, k])
            A_new[k, k] *= (alpha * A[k, k])
            X_new[i, k] += (1 - alpha) * Y[i, k]
            A_new[k, k] += (1 - alpha) * Y[i, k]
    X_new, A_new = np.matrix(X_new / np.sum(X_new, axis=0).reshape(1, M)), np.matrix(A / np.sum(A_new))
    Y = X_new * A_new
    return X_new, A_new, Y


def read_edgelist(filename, weighted=False):
    idmap = set()
    edge_cache = {}
    with open(filename) as f:
        for line in f:
            if weighted:
                u, v, w = [int(x) for x in line.strip().split()]
            else:
                tmp = [int(x) for x in line.strip().split()]
                u, v, w = tmp[0], tmp[1], 1.0
            edge_cache[(u, v)] = w
            idmap.add(u)
            idmap.add(v)
    idmap = list(idmap)  # 数组下标与结点唯一id标识的映射
    idmap_inv = {nid: i for i, nid in enumerate(idmap)}  # 结点唯一id标识与数组下标的映射
    N = len(idmap)
    adj_mat = np.zeros((N, N))
    for (u, v), w in edge_cache.items():
        adj_mat[idmap_inv[u], idmap_inv[v]] = w
    adj_mat += adj_mat.T
    return idmap, idmap_inv, adj_mat


# FacetNet with # of nodes and communities fixed
# FacetNet算法的基础形式：节点和社区固定（应该是指数量固定）
def alg(net_path, alpha, tsteps, N, M, with_truth=True):
    """
    FacetNet算法的基础形式
    :param net_path: 数据集路径（边集）
    :param alpha:
    :param tsteps: 时间片个数（与数据集的时间片个数保持一致）
    :param N: 节点的个数(fixed)?，默认的生成器配置为128
    :param M: 社区的个数(fixed)?，默认的生成器配置为4
    :param with_truth: 数据是否包含ground truth
    :return: none，结果直接输出到控制台
    """
    X, A = np.random.rand(N, M), np.diag(np.random.rand(M))  # X为 N * M 的随机数矩阵，A为 M * M 的对角随机数矩阵
    X, A = np.matrix(X / np.sum(X, axis=0).reshape(1, M)), np.matrix(A / np.sum(A))
    Y = X * A

    for t in range(tsteps):
        # G = nx.read_weighted_edgelist(net_path+"%d.edgelist" % t)
        # idmap, mapping: nodeid → array_id
        idmap, idmap_inv, adj_mat = read_edgelist(net_path + "%d.edgelist" % t, weighted=False)  # 从文件读取图数据（edgelist），这里是无权图
        if with_truth:
            with open(net_path + "%d.comm" % t) as f:  # 如果存在ground truth社区，则读取它们
                comm_map = {}  # mapping: nodeid → its community; comm_map保存从节点id到它所属的社区号的映射
                for line in f:
                    id0, comm0 = line.strip().split()
                    comm_map[int(id0)] = int(comm0)
        W = Sim(adj_mat, weighted=False)  # adj_mat是邻接矩阵
        X_new, A_new, Y = param_update(X, A, Y, W, alpha)  # todo::param_update的功能？
        D = np.zeros((N,))  # D为长度为N的零向量
        for i in range(N):
            D[i] = np.sum(Y[i, :])
        D = np.matrix(np.diag(D))
        soft_comm = D.I * X_new * A_new
        comm_pred = np.array(np.argmax(soft_comm, axis=1)).ravel()  # 社区预测
        print("time:", t)
        if with_truth:
            comm = np.array([comm_map[idmap_inv[i]] for i in range(N)])
            print("mutual_info:", mutual_info_score(comm, comm_pred))  # 计算并打印mutual info score（仅当数据集包含ground truth）
        print("soft_modularity:", soft_modularity(soft_comm, W))  # 计算并打印模块度
        # community_net = A_new * X_new.T * soft_comm
        # print("community_net")
        # print(community_net)
        # evolution_net = X.T * soft_comm
        # print("evolution_net")
        # print(evolution_net)
        X, A = X_new, A_new  # 迭代更新X和A矩阵

        yield comm_pred  # 改造为生成器


# FacetNet with # of nodes and communities changed
# FacetNet算法的扩展形式：节点和社区可变（应该是指数量）
def alg_extended(net_path, alpha, tsteps, M, with_truth=True):
    idmap0, idmap_inv0 = [], {}
    for t in range(tsteps):
        print("time:", t)
        idmap, idmap_inv, adj_mat = read_edgelist(net_path + "%d.edgelist" % t, weighted=False)
        if with_truth:
            with open(net_path + "%d.comm" % t) as f:
                comm_map = {}
                for line in f:
                    id0, comm0 = line.strip().split()
                    comm_map[int(id0)] = int(comm0)
        N = len(idmap)
        W = Sim(adj_mat, weighted=False)
        if t == 0:
            X, A = np.random.rand(N, M), np.diag(np.random.rand(M))
            X, A = np.matrix(X / np.sum(X, axis=0).reshape(1, M)), np.matrix(A / np.sum(A))
            Y = X * A
        else:  # adjustment for changing of nodes
            reserved_rows = [idmap_inv0[x] for x in idmap0 if x in idmap]
            num_new, num_old = len(set(idmap) - set(idmap0)), len(reserved_rows)
            Y = Y[reserved_rows, :]
            Y /= np.sum(Y)
            Y = np.pad(Y, ((0, num_new), (0, 0)), mode='constant', constant_values=(0, 0))
            # not mentioned on the paper, but are necessary for node changing
            X = X[reserved_rows, :]
            X = np.matrix(X / np.sum(X, axis=0).reshape(1, M))
            X *= num_old / (num_old + num_new)
            X = np.pad(X, ((0, num_new), (0, 0)), mode='constant', constant_values=(1 / num_new, 1 / num_new))

        X_new, A_new, Y = param_update(X, A, Y, W, alpha)
        D = np.zeros((N,))
        for i in range(N):
            D[i] = np.sum(Y[i, :])
        D = np.matrix(np.diag(D))
        soft_comm = D.I * X_new * A_new

        comm_pred = np.array(np.argmax(soft_comm, axis=1)).ravel()
        if with_truth:
            comm = np.array([comm_map[idmap[i]] for i in range(N)])
            print("mutual_info:", mutual_info_score(comm, comm_pred))
        s_modu = soft_modularity(soft_comm, W)
        print("soft_modularity: %f" % s_modu)
        # community_net = A_new * X_new.T * soft_comm
        # print("community_net")
        # print(community_net)
        # evolution_net = X.T * soft_comm
        # print("evolution_net")
        # print(evolution_net)
        X, A = X_new, A_new
        idmap0, idmap_inv0 = idmap, idmap_inv

        yield comm_pred  # 改造为生成器


# do experiment with network stated in 4.1.2
# 论文中4.1.2小节的实验
def exp1():
    # 生成人工图到data目录，人工网络包含ground truth的社区
    tsteps = 15  # 应该是指定人工网络中时间片的个数
    print("generating synthetic graph")
    generate_evolution("./data/syntetic1/", tsteps=tsteps)

    # 执行算法
    print("start the algorithm")
    alpha = 0.9  # todo::alpha参数含义？
    N, M = 128, 4  # todo::N和M参数含义？
    np.random.seed(0)  # 重新随机化
    alg("./data/syntetic1/", alpha, tsteps, N, M)


# do experiment with adding and removing nodes
# 增减节点的实验
def exp2():
    # 生成人工图
    tsteps = 15
    print("generating synthetic graph")
    generate_evolution2("./data/syntetic2/", tsteps=tsteps)

    # 执行探测算法
    print("start the algorithm")
    alpha = 0.5
    np.random.seed(0)
    alg_extended("./data/syntetic2/", alpha, tsteps, 4)


# do experiment with network stated in 4.1.2, adding weight
# 论文中4.1.2小节的实验，带权图
def exp3():
    # 生成人工图到data目录
    tsteps = 15
    print("generating synthetic graph")
    generate_evolution3("./data/syntetic3/", tsteps=tsteps)

    # 执行算法
    print("start the algorithm")
    alpha = 0.9
    N, M = 128, 4
    np.random.seed(0)
    alg("./data/syntetic3/", alpha, tsteps, N, M)


# 使用自收集的dblp数据集来实验
def exp_dblp():
    # 数据集路径
    # data_path = "./data/cit-DBLP/"
    data_path = "./data/fb-pages-food/"

    alpha = 0.9
    N, M = 942, 4  # 社区的数量需要指定
    np.random.seed(0)  # 随机化
    # alg(data_path, alpha=alpha, tsteps=1, N=N, M=M, with_truth=False)  # 执行算法
    for res in alg_extended(data_path, alpha, 1, 4, with_truth=False):
        print(res)


if __name__ == "__main__":
    # print("do experiment with network stated in 4.1.2")
    # exp1()
    # print("\ndo experiment with adding and removing nodes")
    # exp2()
    # print("\ndo experiment with network stated in 4.1.2, adding weight")
    # exp3()
    exp_dblp()
