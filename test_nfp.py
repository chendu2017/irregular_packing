# -*- coding: utf-8 -*-
from nfp_function import Nester, content_loop_rate
from settings import BIN_WIDTH, BIN_NORMAL, BIN_CUT_BIG, LOOP_TIME

import ast
import pandas as pd
lingjian = pd.read_csv('.\L0002_lingjian.csv')

if __name__ == '__main__':
    n = Nester()
    s = [ast.literal_eval(contour) for contour in lingjian['外轮廓']]

    n.add_objects(
               #[ [ [0,0],[0,20],[20,0]    ],
               #  [ [20,0],[20,10],[30,10],[30,0] ],
               #  [[10,0],[20,0],[20,10],[10,10]]
               #        ]
               
               
                #[
                #[[10,0],[20,0],[20,10],[10,10]],
                #[[10,20],[20,20],[15,30]],
                #[[30,10],[50,10],[35,15],[40,30],[30,30]]
                 #]
        s[:50]#,lingjian['零件号'].values
            )

    if n.shapes_max_length > BIN_WIDTH:
        BIN_NORMAL[2][0] = n.shapes_max_length
        BIN_NORMAL[3][0] = n.shapes_max_length

    # 选择面布
    n.add_container(BIN_NORMAL)
    # 运行计算 
    n.run() #进行一次未生成子代的计算

    # 设计退出条件
    res_list = list()
    best = n.best
    # 放置在一个容器里面
    # set_target_loop(best, n)    # T6

    # 循环特定次数
    content_loop_rate(best, n, loop_time=LOOP_TIME-1)   # T7 , T4

    
