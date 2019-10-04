# -*- coding: utf-8 -*-

SCALING_FACTOR = 10 #缩放画布和块料 到整数

LOOP_TIME = 1
POPULATION_SIZE = 1  # 基因组数
MUTA_RATE = 20        # 变异概率
ROTATIONS = 1   # 旋转选择， 1： 不能旋转
# 单位都是MM(毫米)
SPACING = 5*SCALING_FACTOR     # 图形间隔空间
CONTAINERSPACING = -0*SCALING_FACTOR #容器预留边距,负数表示向内
CURVETOLERANCE = 0.15*SCALING_FACTOR # polygon点之间判定为两个点的最小距离
CONVEX_WEIGHT  = 0.2   # 在寻找初始序列时，convexity的权重
AREA_WEIGHT    = 0.8   # 在寻找初始序列时，area的权重

MAX_MOVE_X = 2 * SCALING_FACTOR #压缩时，X方向最大位移
MAX_MOVE_Y = 2 * SCALING_FACTOR #压缩时，Y方向最大位移
# 不同面料尺寸
BIN_HEIGHT = 1600 * SCALING_FACTOR
BIN_WIDTH = 20000 * SCALING_FACTOR
BIN_NORMAL = [[0, 0], [0, BIN_HEIGHT], [BIN_WIDTH, BIN_HEIGHT], [BIN_WIDTH, 0]]        # 一般布是无限长
BIN_CUT_BIG = [[0, 0], [0, 1570], [2500, 1570], [2500, 0]]       # 切割机尺寸 1
BIN_CUT_SMALL = [[0, 0], [0, 1200], [1500, 1200], [1500, 0]]     # # 切割机尺寸 2