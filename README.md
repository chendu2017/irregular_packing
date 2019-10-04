# irregular_packing

This is the codes for the competition https://tianchi.aliyun.com/competition/entrance/231749/introduction?spm=5176.12281949.1003.3.493e2448UnwztK

## About codes

Most of these codes come from @liangxu chen (https://github.com/liangxuCHEN), including the genetic algorithm, worker class. I am not sure whether I correctly upload the LICENSE (This is my first time to heavily use others codes. So please let me know if I didn't upload enough information about copyright.)

For the purpose of efficiency, I changed some details to reduce the computation time by removing input part and output part, which are two important parts in an engineering project. Also, some unnecessary iterations are removed for efficiency. That may lead to the situation that these codes are just suitable for this competition and learning on irregular packing. Not very useful for engineer purpose.

## Irregular Packing

For starters, SVGnest (https://github.com/Jack000/SVGnest), a free and open-source software, may give you a full picture of irregular packing and no-fit polygons.

In my revised codes, I introduced an compaction algorithm to PlacementWorker class in placement_worker.py. This algorithm can compact the existing layout by moving pieces without changing theirs relative positions.
The Compaction algorithm is proposed by [1]. A linear programming is used to describe the moving problem, and the feasible locations for each pieces are generated from the current feasible layout. 

Why this compaction algorithm is needed? Because the initial layout is obtained by the genetic algorithm, in which the local objective is minimizing the length of used textile. However, getting a local optimal solution does not mean the whole layout is a global optimum. The compaction algorithm provides a possibility of compacting the current layout, although it is also
possible that constraints generated from the current layout are conflicting. That is because the pieces are non-convex, so the feasible region described by these generated constraints is much smaller then truly feasible region. But this algorithm still has the potential!

For reading materials are listed:

[2] provides a heuristic to find out no-fit polygons when there is concavity, by "concavity" I mean polygons with shapes like letter "C". You need to put another piece into the letter "C". It's not a easy work, especially for the computer.

[3] provides a tutorial, which is easy to follow.

[4] provides a heuristic, allowing the worker to look forward. So the worker can compare whether the work should put the one on hand or put the next piece, 
so that the worker can get a better placement layout.

[5] provides another stream to deal with this problem, without genetic algorithm framework. The tabu search used in this paper achieves much more better results than other best results in literature.






## Reference
[1] Gomes A M, Oliveira J F. Solving irregular strip packing problems by hybridising simulated annealing and linear programming[J]. 
European Journal of Operational Research, 2006, 171(3): 811-829.

[2] Burke E K, Hellier R S R, Kendall G, et al. Complete and robust no-fit polygon generation for the irregular stock cutting problem[J]. European Journal of Operational Research, 2007, 179(1): 27-49.

[3] Bennell J A, Oliveira J F. The geometry of nesting problems: A tutorial[J]. European Journal of Operational Research, 2008, 184(2): 397-415.

[4] Gomes A M, Oliveira J F. A 2-exchange heuristic for nesting problems[J]. European Journal of Operational Research, 2002, 141(2): 359-370.

[5] Bennell J A, Dowsland K A. Hybridising tabu search with optimisation techniques for irregular stock cutting[J]. Management Science, 2001, 47(8): 1160-1172.


