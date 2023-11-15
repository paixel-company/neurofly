import numpy as np
from brightest_path_lib.algorithm import NBAStarSearch


def seg2skel(img,seg,roi):
    skel = []
    if len(seg)<2:
        return None
    for n1,n2 in zip(seg[:-1],seg[1:]):
        p1=n1['pos']
        p2=n2['pos']
        p1 = [i-j for i,j in zip(p1,roi[0:3])]
        p2 = [i-j for i,j in zip(p2,roi[0:3])]
        sa = NBAStarSearch(img, start_point=p1, goal_point=p2)
        try:
            path = sa.search()
            skel.append(path)
        except:
            print('one seg failed to interp')

    return np.vstack(skel)


