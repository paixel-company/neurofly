from tqdm import tqdm
from scipy.spatial import KDTree
from ntools.dbio import get_size, segs2db, points2db
import numpy as np


class SegsTree():
    def __init__(self, segs):
        # endpoints 
        esids = []
        ecoords = []
        ecids = [] # point index in its segment
        for seg in segs:
            ecoords.append(seg['points'][0])
            ecids.append(0)
            ecoords.append(seg['points'][-1])
            ecids.append(len(seg['points'])-1)
            esids.append(seg['sid'])
            esids.append(seg['sid'])
        self.etree = KDTree(np.array(ecoords)) # kdtree of endpoints
        self.endpoints = []
        for i, coord in enumerate(ecoords):
            self.endpoints.append(
                {
                    'coord': coord,
                    'sid': esids[i],
                    'cid': ecids[i]
                }
            )

        # all sampled points
        sids = []
        coords = []
        cids = []
        for seg in segs:
            for i, coord in enumerate(seg['points']):
                coords.append(coord)
                sids.append(seg['sid'])
                cids.append(i)

        self.stree = KDTree(np.array(coords))
        self.spoints = []
        for i, coord in enumerate(coords):
            self.spoints.append(
                {
                    'coord': coord,
                    'sid': sids[i],
                    'cid': cids[i]
                }
            )

        self.segs = {seg['sid']: seg for seg in segs}



    def query_segs(self,coord,dis):
        dis, nbr_indices = self.etree.query(coord,k=50,distance_upper_bound=dis)
        nbr_indices = [j for i,j in zip(dis,nbr_indices) if i!=np.inf]
        nbr_points = [self.endpoints[i] for i in nbr_indices]
        return nbr_points



    def get_nbr_segs(self,coord,dis):
        nbr_indices = self.stree.query_ball_point(coord,dis,p=np.inf)
        nbr_points = [self.spoints[i] for i in nbr_indices]
        sids = list(set([p['sid'] for p in nbr_points]))
        return sids


    def get_length(self):
        length = []
        for point in self.endpoints:
            nbr_points = self.query_segs(point['coord'],dis=15)
            n_sids = list(set([point['sid'] for point in nbr_points]))
            if len(n_sids)==1:
                length.append(0)
                continue
            total_length = 0
            for sid in n_sids:
                total_length += len(self.segs[sid]['points'])
            length.append(total_length)

        return length
        # sorted_index = np.argsort(np.array(length))
        # sorted_index = sorted_index.tolist()
        # sorted_index.reverse()
        # print(sorted_index)


    def segs_list(self):
        return list(self.segs.values()) 
    

    def write_db(self,db_path):
        '''
        point:
            {
                coord: [x,y,z],
                sid: int,
                cid: int,
                length: int,
                checked: int,
                PRIMARY KEY (sid,cid)
            }
        segment:
            {
                sid: int,
                points: [head,...,tail],
                sampled_points: points[::interval],
                nbrs = [[index_of_point,sid],...],
            }
        '''
        offset = get_size(db_path)

        segs = []
        for i, seg in enumerate(self.segs_list()):
            s = seg.copy()
            s['sid'] = s['sid']+offset
            segs.append(s)

        segs2db(segs,db_path)



def cat_segs(segs):
    # for seg ends at x and y splices, find k nearest ends, if there's only one neibhgor, calculte weather their orientations align. Concatenate two segs together by adding seg_id to neighbors. 

    # first build kdtree with all head and tail coordinates, each coordinate have its corresponding sid. The segs should be saved in a dict so that can be indexed by sid.

    '''
    segment:
        {
            sid: int,
            points: [head,...,tail],
            sampled_points: points[::interval],
            nbrs = [[index_of_point,sid],...],
            checked: int
        }
    '''
    for seg in segs:
        print(seg['sid'],seg['head'],seg['tail'],len(seg['points']),len(seg['sampled_points']))



if __name__ == '__main__':
    db_path = 'tests/z002.db'
    from ntools.dbio import read_segs
    segs = read_segs(db_path)
    stree = SegsTree(segs)
    # stree.write_db(db_path)
    # cat_segs(segs)

