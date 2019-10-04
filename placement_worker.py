# -*- coding: utf-8 -*-
import json
from nfp_utls import almost_equal, rotate_polygon, get_polygon_bounds, polygon_area, is_convex, D_function, edge_point_distance
from settings import BIN_HEIGHT, MAX_MOVE_X, MAX_MOVE_Y, SCALING_FACTOR
import copy
import pyclipper
import time
import cvxpy as cp
TOL = 0.00001 * SCALING_FACTOR

class PlacementWorker():
    def __init__(self, bin_polygon, paths, ids, rotations, config, nfp_cache):
        self.bin_polygon = bin_polygon
        self.paths = copy.deepcopy(paths)
        self.ids = ids       # 图形原来的ID顺序
        self.rotations = rotations
        self.config = config
        self.nfpCache = copy.deepcopy(nfp_cache) or {}
    
    
    def find_constraints(self,A,B,i,j,x,y,placements):
        
        # -*- coding: utf-8 -*-
        """
        Created on Thu Sep  5 11:22:11 2019
        
        @author: chend
        """
        
        '''
        param: A is a stationary polygon
               B is the obitating polygon
               i,j 当前寻找约束的两个polygon在当前排序中的位置，从0开始，-1表示bin
               x,y LP的决策变量组，为points的第一个点的真实坐标，例如x[i]表示第i个块料的第一个点的横轴位置
        output: [] constraints的列表
        '''
        
        constraints = []
        
        key = {
                'A': A['p_id'],
                'B': B['p_id'],
                'inside': False,
                'A_rotation': A['rotation'],
                'B_rotation': B['rotation']
                } #nfp是按 B的第一个点绕出来的
        json_key = json.dumps(key)
        
        nfp = self.nfpCache[json_key][0]
        nfp_current_location = [{'x':point['x']+placements[i]['x']  , 'y':point['y']+placements[i]['y']} for point in nfp]
        ref_point_B = B['current_location'][0]
        
        # nfp逆时针，当ref_point_B在nfp上edge的右边时，便是valid_edge,找出所有valid_edge
        valid_edge = []
        valid_distance = []
        valid_point_index = []
        for k in range(0,len(nfp_current_location)):
            
            point_1 = nfp_current_location[k-1]
            point_2 = nfp_current_location[k]
            
            D_func_value = D_function(point_1,point_2,ref_point_B)
            dis          = edge_point_distance([point_1,point_2],ref_point_B)
            
            #D_func为负数说明在右边,0表示在向量上;由于精度问题，这里距离足够小的话，我们也认为是在外面
            if  (D_func_value <= 0 + TOL): #or  (dis <= 0 + TOL): 
                valid_edge.append([{'x':point_1['x'],'y':point_1['y']}, {'x':point_2['x'],'y':point_2['y']}])
                valid_distance.append( edge_point_distance(valid_edge[-1],ref_point_B) )
                valid_point_index.append([k-1,k])
        
        
        
        
        #有一点点由于精度引起的重叠，加一个constraint要求距离不能继续变大
        if len(valid_edge) == 0:
            min_dis,point_index = min( (edge_point_distance([nfp_current_location[k-1],nfp_current_location[k]] , ref_point_B),k) 
                                      for k in range(len(nfp_current_location))  )
            print(u'(%s,%s) ref_point在nfp内部了'%(A['p_id'],B['p_id']),
                  u'离最近边的距离为:', min_dis )
            print()

            constraints.append( D_function(nfp_current_location[point_index-1],nfp_current_location[point_index],
                                                  {'x':x[j],'y':y[j]}) <= 
                                D_function(nfp_current_location[point_index-1],nfp_current_location[point_index],
                                                  ref_point_B))
            return constraints
        
        elif len(valid_edge) > 0:
            if is_convex(nfp_current_location):
                
                max_distance = max(valid_distance)
                
                #目前不在nfp上
                if max_distance > 0:            
                    max_distance_index = valid_distance.index(max_distance)
                    point_1 = valid_edge[max_distance_index][0]
                    point_2 = valid_edge[max_distance_index][1]
                    constraints.append( D_function(point_1,point_2,{'x':x[j],'y':y[j]}) <= 0 )
                    
                    
                #目前在 nfp上，在nfp上的话，只可能在vertex上，或在edge上
                elif max_distance == 0:
                
                    #在vertex上
                    if len(valid_edge) == 1:
                        edge = valid_edge[0]
                        constraints.append(  D_function(edge[0],edge[1],{'x':x[j],'y':y[j]}) <= 0 )
                    
                    #在edge上
                    if len(valid_edge) == 2:
                        edge_1 = valid_edge[0]
                        if edge_1[0]['x'] == edge_1[1]['x']:
                            slope_1 = 1000000
                        else:
                            slope_1 = (edge_1[1]['y'] - edge_1[0]['y'])/(edge_1[1]['x']-edge_1[0]['x'])                    
                        #slope_2
                        edge_2 = valid_edge[1]
                        if edge_2[0]['x'] == edge_2[1]['x']:
                            slope_2 = 1000000
                        else:
                            slope_2 = (edge_2[1]['y'] - edge_2[0]['y'])/(edge_2[1]['x']-edge_2[0]['x'])                    
                        
                        if abs(slope_1) <= abs(slope_2):
                            constraints.append( D_function(edge_1[0],edge_1[1],{'x':x[j],'y':y[j]}) <= 0 )
                        else:
                            constraints.append( D_function(edge_2[0],edge_2[1],{'x':x[j],'y':y[j]}) <= 0 )
                
            if not is_convex(nfp_current_location):
                
                max_distance = max(valid_distance)
                edge = valid_edge[ valid_distance.index(max_distance) ]
                point_1_index,point_2_index = valid_point_index[valid_distance.index(max_distance)][0],valid_point_index[valid_distance.index(max_distance)][1]
                
                #加入edge构成的约束
                constraints.append( D_function(edge[0],edge[1],{'x':x[j],'y':y[j]}) <= 0 )
                
                if len(edge) is not 0:
                    ON_RIGHT = True
                else:
                    ON_RIGHT = False
                
                #逆时针寻找，下一条边的终点在edge右侧的，则加入下一条边构成的约束
                while ON_RIGHT:
                    if point_2_index < len(nfp_current_location)-1:
                        point_3_index = point_2_index + 1
                    if point_2_index == len(nfp_current_location)-1:
                        point_3_index = 0
                    
                    point_1 = nfp_current_location[point_1_index]
                    point_2 = nfp_current_location[point_2_index]
                    point_3 = nfp_current_location[point_3_index]
                        
                    ON_RIGHT =  D_function( point_1,point_2,point_3) <= 0 
                    
                    point_1_index = point_2_index
                    point_1 = nfp_current_location[point_1_index]
                    
                    point_2_index = point_3_index
                    point_2 = nfp_current_location[point_2_index]
                    
                    if point_3_index == len(nfp_current_location)-1:
                        point_3_index = 0
                        point_3 = nfp_current_location[point_3_index]
                    
                    if ON_RIGHT:
                        constraints.append( D_function(point_1,point_2,{'x':x[j],'y':y[j]}) <= 0 )
                
                
                #顺时针寻找，下一条边的终点在上一条右侧的，则加入这条边构成的约束
                point_1_index = valid_point_index[valid_distance.index(max_distance)][0]
                point_2_index = valid_point_index[valid_distance.index(max_distance)][1]
                point_0_index = point_1_index - 1
                
                point_0 = nfp_current_location[point_0_index]
                point_1 = nfp_current_location[point_1_index]
                point_2 = nfp_current_location[point_2_index]
                
                if len(edge) is not 0:
                    ON_RIGHT = True
                while ON_RIGHT:
                    
                    ON_RIGHT = D_function( point_0,point_1,point_2) <= 0 
                         
                    point_0_index = point_0_index - 1
                    point_1_index = point_1_index - 1
                    point_2_index = point_2_index - 1
                    
                    point_0 = nfp_current_location[point_0_index]
                    point_1 = nfp_current_location[point_1_index]
                    point_2 = nfp_current_location[point_2_index]
                    
                    if ON_RIGHT:
                        constraints.append( D_function(point_1,point_2,{'x':x[j],'y':y[j]}) <= 0 )
                        
            return constraints

    def compact_layout(self,current_layout):
        # -*- coding: utf-8 -*-
        """
        Created on Thu Sep  5 11:22:11 2019
        
        @author: chend
        """
        
        '''
        Param: current_layout = {'placed':[], 'placements':[]}   其中placed是经旋转的块料
        output: placement: [{'x': ,'y':},{},....]    
        '''
        placed = copy.deepcopy(current_layout['placed'])
        placements = copy.deepcopy(current_layout['placements'])
        
        if len(placed)<2 and len(placements)<2:
            return placements
        
        
        for i in range(0,len(placed)):
            
            placed[i]['current_location'] = [{'x':point['x']+placements[i]['x'],
                                              'y':point['y']+placements[i]['y']} for point in placed[i]['points']]
        
            # Linear Programming 
        
        #variable
        x = cp.Variable(shape=len(placed))#integer=True
        y = cp.Variable(shape=len(placed))
        z = cp.Variable(shape=1)
        #obj
        obj = cp.Minimize(z)
        #constraints
        constraints = []
        constraints += [x[i]>=0 for i in range(len(placed))]
        constraints += [y[i]>=0 for i in range(len(placed))]
        constraints += [z>=0]
        
        # i是固定的块， j是待移动方块
        for j in range(0,len(placed)):
            
            B = placed[j]
            
            # 每个块的移动不能超过一定距离
            ref_point = B['points'][0]
            distance_from_ref_point_to_part_right = B['x'] + B['width'] - ref_point['x']
            distance_from_ref_point_to_part_left  = ref_point['x'] - B['x']
            distance_from_ref_point_to_part_top   = B['y'] + B['width'] - ref_point['y']
            distance_from_ref_point_to_part_bottom= ref_point['y'] - B['y']
            # x-axis
            current_x = ref_point['x'] + placements[j]['x']
            constraints.append( x[j] - current_x <= MAX_MOVE_X )
            constraints.append( x[j] - current_x >= -MAX_MOVE_X )
            # y-axis 
            current_y = ref_point['y'] + placements[j]['y']
            constraints.append( y[j] - current_y <= MAX_MOVE_Y )   
            constraints.append( y[j] - current_y >= -MAX_MOVE_Y )
            
            # 目标函数 z >= x_i + W_i, W_i是当前ref点到该块料最右侧距离
            constraints.append( z >= x[j] + distance_from_ref_point_to_part_right )
            
            # 每个块料都需要在 bin_offset里面
            bin_offset = get_polygon_bounds(self.bin_polygon['points_offset'])
            constraints.append( x[j] - distance_from_ref_point_to_part_left >= bin_offset['x'] )
            constraints.append( y[j] + distance_from_ref_point_to_part_top <= bin_offset['y'] + bin_offset['width'] )
            constraints.append( y[j] - distance_from_ref_point_to_part_bottom >= bin_offset['y'] )
            

            for i in range(0,j):
                
                A = placed[i]
                
                constraints += self.find_constraints(A,B,i,j,x,y,placements)
        
        # compaction problem
        prob = cp.Problem(obj,constraints)
        prob.solve(solver = cp.CVXOPT)#,**{'cplex_filename':'a.lp'}
        print('x:',x.value)
        print('y:',y.value)
        
        if prob.status == 'optimal':
            best_location = [{'x':x_loc,'y':y_loc}  for x_loc,y_loc in zip(x.value,y.value)]
            shift_vector  = [{'x': best_location[k]['x'] - placed[k]['current_location'][0]['x'],
                             'y': best_location[k]['y'] - placed[k]['current_location'][0]['y']}
                              for k in range(0,len(placed))]
            
            placements = [{'x':placements[k]['x'] + shift_vector[k]['x'],
                           'y':placements[k]['y'] + shift_vector[k]['y'],
                           'p_id':placements[k]['p_id'],
                           'rotation':placements[k]['rotation']} 
                           for k in range(0,len(placements))]
            print(u'找到最优解，已压缩一次\n')
            
        elif prob.status == 'infeasible':
            print(u'压缩模型无解,无法压缩\n')
        
        else:
            print(u'压缩模型出错\n')
        
        return placements

    def place_paths(self):
        start = time.clock()
        # 排列图形
        if self.bin_polygon is None:
            return None

        # rotate paths by given rotation
        rotated = list()
        for i in range(0, len(self.paths)):
            
            r = copy.deepcopy(self.paths[i][1])
            
            #旋转本体
            rot_return = rotate_polygon(self.paths[i][1]['points'], self.paths[i][2])
            r['points'] = rot_return['points']
            r['x'] = rot_return['x']
            r['y'] = rot_return['y']
            r['width'] = rot_return['width']
            r['height'] = rot_return['height']
            
            #旋转 offset 和 convex_hull
            r['points_offset'] = rotate_polygon(self.paths[i][1]['points_offset'], self.paths[i][2])['points']
            r['convex_hull'] = rotate_polygon(self.paths[i][1]['convex_hull'], self.paths[i][2])['points']
            
            #其他信息
            r['rotation'] = self.paths[i][2]
            r['source'] = self.paths[i][1]['p_id']
            r['p_id'] = self.paths[i][0]
            rotated.append(r)

        paths = rotated
        # 保存所有转移数据
        all_placements = list()
        orders = list()
        rotations = list()
        # 基因组的适应值
        fitness = []
        bin_area = abs(polygon_area(self.bin_polygon['points']))
        min_width = None
        while len(paths) > 0:
            placed = list()
            placements = list()
            # add 1 for each new bin opened (lower fitness is better)
            #fitness += 1
            for i in range(0, len(paths)):
                path = paths[i]
                # 图形的坐标
                key = json.dumps({
                    'A': '-1',
                    'B': path['p_id'],
                    'inside': True,
                    'A_rotation': 0,
                    'B_rotation': path['rotation']
                })

                binNfp = self.nfpCache.get(key)
                if binNfp is None or len(binNfp) == 0:
                    continue

                # part unplaceable, skip
                error = False

                
                # ensure all necessary NFPs exist
                for p in placed:
                    key = json.dumps({
                        'A': p['p_id'],
                        'B': path['p_id'],
                        'inside': False,
                        'A_rotation': p['rotation'],
                        'B_rotation': path['rotation']
                    })
                    nfp = self.nfpCache.get(key)
                    if nfp is None:
                        error = True
                        break

                # part unplaceable, skip
                if error:
                    continue
                
                #把第一个图形放到布料左下角
                position = None
                if len(placed) == 0:
                    for j in range(0, len(binNfp)):
                        for k in range(0, len(binNfp[j])):
                            if position is None or (binNfp[j][k]['x']-path['points'][0]['x'] < position['x']):
                                position = {
                                    'x': binNfp[j][k]['x'] - path['points'][0]['x'],
                                    'y': binNfp[j][k]['y'] - path['points'][0]['y'],
                                    'p_id': path['p_id'],
                                    'rotation': path['rotation']
                                }

                    placements.append(position)
                    placed.append(path)
                    continue

                clipper_bin_nfp = list()
                for j in range(0, len(binNfp)):
                    clipper_bin_nfp.append([[p['x'], p['y']] for p in binNfp[j]])

                # 找出 待放置方块 和 已放置方块 的 combined_NFP
                clipper = pyclipper.Pyclipper()
                for j in range(0, len(placed)):
                    p = placed[j]
                    key = json.dumps({
                        'A': p['p_id'],
                        'B': path['p_id'],
                        'inside': False,
                        'A_rotation': p['rotation'],
                        'B_rotation': path['rotation']
                    })
                    nfp = self.nfpCache.get(key)

                    if nfp is None:
                        continue
                    for k in range(0, len(nfp)):
                        clone = [[np['x'] + placements[j]['x'], np['y'] + placements[j]['y']] for np in nfp[k]]
                        clone = pyclipper.CleanPolygon(clone)
                        if len(clone) > 2:
                            clipper.AddPath(clone, pyclipper.PT_SUBJECT, True)
                combine_nfp = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                if len(combine_nfp) == 0:
                    continue
                
                # 以 combined_NFP 为 夹子(clip) 夹掉 待放方块和画布的内部NFP
                clipper = pyclipper.Pyclipper()
                clipper.AddPaths(combine_nfp, pyclipper.PT_CLIP, True)
                try:
                    clipper.AddPaths(clipper_bin_nfp, pyclipper.PT_SUBJECT, True)
                except:
                    print(u'图形坐标出错', clipper_bin_nfp)
                # choose placement that results in the smallest bounding box
                finalNfp = clipper.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                if len(finalNfp) == 0:
                    continue
                
                #这里clean nfp后会有很多重叠的地方，不能clean
                #finalNfp = pyclipper.CleanPolygons(finalNfp)

                for j in range(len(finalNfp)-1, -1, -1):
                    if len(finalNfp[j]) < 3:
                        finalNfp.pop(j)
                if len(finalNfp) == 0:
                    continue

                finalNfp = [[{'x': p[0], 'y': p[1]}for p in polygon] for polygon in finalNfp]

                min_width = None
                min_area = None
                min_x = None
                
                #all_points 是已摆好的方块的转移后摆好的坐标点集合
                all_points = list() 
                for m in range(0, len(placed)):
                    for p in placed[m]['points']:
                        all_points.append({
                            'x': p['x']+placements[m]['x'],
                            'y': p['y']+placements[m]['y']
                        })
                
                # 生成nfp多边形
                for nf in finalNfp:

                    if abs(polygon_area(nf)) < 2:
                        continue

                    for p_nf in nf:
                        
                        all_points_deepcopy = copy.deepcopy(all_points)
                        
                        # path 坐标
                        shift_vector = {
                            'x': p_nf['x'] - path['points'][0]['x'],
                            'y': p_nf['y'] - path['points'][0]['y'],
                            'p_id': path['p_id'],
                            'rotation': path['rotation'],
                        }

                        # 找新坐标后的最小矩形
                        for m in range(0, len(path['points'])):
                            all_points_deepcopy.append({
                                'x': path['points'][m]['x'] + shift_vector['x'],
                                'y': path['points'][m]['y'] + shift_vector['y']
                            })
                        
                        all_points_deepcopy.append({'x':0,'y':0})
                        rect_bounds = get_polygon_bounds(all_points_deepcopy)
                        # weigh width more, to help compress in direction of gravity 
                        # ???  width*2 不就是对width做更多的惩罚吗？这样图形会尽量往左下边放
                        area = rect_bounds['height'] + rect_bounds['width']
                        
                        if (min_area is None or area < min_area or almost_equal(min_area, area)) and (
                                        min_x is None or shift_vector['x'] <= min_x):
                            min_area = area
                            min_width = rect_bounds['width']
                            position = shift_vector
                            min_x = shift_vector['x']

                if position:
                    placed.append(path)
                    placements.append(position)

            for p in placed:
                p_id = paths.index(p)
                if p_id >= 0:
                    paths.pop(p_id)
            
            if placements and len(placements) > 0:
                all_placements.append(placements)

            else:
                # something went wrong
                break
            

        placed_area = 0
        all_points = []
        for k,path in enumerate(placed):
            placed_area += abs(polygon_area(path['points']))
            for point in path['points']:
                all_points.append({
                        'x': point['x'] + placements[k]['x'],
                        'y': point['y'] + placements[k]['y']
                            })
        all_points.append({'x':0,'y':0})
        min_rect_area = get_polygon_bounds(all_points)['width']*BIN_HEIGHT
        
        #保存第一次heuristic排列得到的结果
        all_placements.append(placements)
        orders.append([ self.paths[i][0] for i in range(len(self.paths))])
        rotations.append([self.rotations ] )
        fitness.append(placed_area/min_rect_area)
        
        print('由Heuristic得到的最小包络矩阵利用率: %.4f'%fitness[0])
        print('排列一次耗时:%.1f'%(time.clock()-start),'\n-----------\n' )
        print('排列得到：',all_placements)
        
        #压缩
        IMPROVED = True
        while IMPROVED:
            start = time.clock()
            
            placements = self.compact_layout({'placed':placed,'placements':placements}) 
            
            #new_fitness 
            placed_area = 0
            all_points = []
            for k,path in enumerate(placed):
                placed_area += abs(polygon_area(path['points']))
                for point in path['points']:
                    all_points.append({
                            'x': point['x'] + placements[k]['x'],
                            'y': point['y'] + placements[k]['y']
                                })
            all_points.append({'x':0,'y':0})
            min_rect_area = get_polygon_bounds(all_points)['width']*BIN_HEIGHT
            new_fitness = placed_area/min_rect_area
        
            if new_fitness > fitness[-1]:
                print(u'已压缩现有排列方式,新排列方式的利用率为: %.4f' %new_fitness)
                print('压缩一次用时: %.2f' %(time.clock()-start),'\n-------------\n')
                fitness.append(new_fitness)
                all_placements.append(placements)
            else:
                IMPROVED = False
                print('压缩完成，已无法继续压缩')
            
        print('压缩总用时:%.2f' %(time.clock()-start))
        print('全部placements:',all_placements)
        
        
        return {'placements': all_placements, 'fitness': fitness, 'paths': paths, 'area': bin_area}
