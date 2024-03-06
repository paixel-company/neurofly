from datetime import datetime
import sqlite3
import os
import time


def segs2db(segs,path):
    '''
    given a list of segs, add all nodes and edges to the datebase.
    seg:
        {
            points: [head,...,tail],
            sampled_points: points[::interval],
        }
    node:
        {
            nid: int, PRIMARY KEY
            coord: str,
            nbrs: str,
            checked: int
        }
    edge
        {
            src: int, 
            des: int,
            date: str, TIMESTAMP
            creator: str,
            PRIMARY KEY: (src,des)
        }

    the graph is undirected, thus edges exist in pairs
    '''

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS segs(
                sid INTEGER PRIMARY KEY,
                points TEXT,
                sampled_points TEXT
            )
            '''
        )
    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS nodes(
                nid INTEGER PRIMARY KEY,
                coord TEXT,
                nbrs TEXT,
                checked INTEGER
            )
            '''
        )

    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS edges(
                src INTEGER,
                des INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                creator TEXT,
                PRIMARY KEY (src,des)
            )
            '''
        )
    
    query = f"SELECT COUNT(*) FROM segs"
    cursor.execute(query)
    result = cursor.fetchone()
    count = result[0]
    
    for seg in segs:
        count+=1
        cursor.execute(f"INSERT INTO segs (sid, points, sampled_points) VALUES (?, ?, ?)",
                    (count, sqlite3.Binary(str(seg['points']).encode()), sqlite3.Binary(str(seg['sampled_points']).encode())))

    print(f'Number of segs in database: {count}, {len(segs)} newly added.')

    query = f"SELECT COUNT(*) FROM nodes"
    cursor.execute(query)
    result = cursor.fetchone()
    count = result[0]

    # assign unique nid for each node in segs according to index
    nodes = [] # [nid,coord,nbr_ids]
    edges = [] # [source_id,target_id]

    # TODO: append nbrs to node[]
    for seg in segs:
        points = seg['sampled_points']
        count+=1
        nodes.append([count,points[0],[count+1]])
        edges.append([count,count+1])
        edges.append([count+1,count])

        for c in points[1:-1]:
            count+=1
            nodes.append([count,c,[count-1,count+1]])
            edges.append([count,count+1])
            edges.append([count+1,count])

        count+=1
        nodes.append([count,points[-1],[count-1]])
    
    # add nodes and edges to the database
    for node in nodes:
        cursor.execute(f"INSERT INTO nodes (nid, coord, nbrs, checked) VALUES (?, ?, ?, ?)",
                    (node[0], sqlite3.Binary(str(node[1]).encode()), sqlite3.Binary(str(node[2]).encode()), 0))

    for edge in edges:
        cursor.execute(f"INSERT INTO edges (src, des, date, creator) VALUES (?, ?, ?, ?)",
                    (edge[0], edge[1], datetime.now(), 'seger'))

    conn.commit()
    conn.close()


def read_segs(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM segs ORDER BY sid")
    rows = cursor.fetchall()
    segs = []
    for row in rows:
        data = {
            'sid': row[0],
            'points': eval(row[1]),
            'sampled_points': eval(row[2]),
        }
        segs.append(data)
    conn.close()
    return segs


def read_nodes(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nodes ORDER BY nid")
    rows = cursor.fetchall()
    points = []
    for row in rows:
        data = {
            'nid': row[0],
            'coord': eval(row[1]),
            'nbrs': eval(row[2]),
            'checked': row[3],
        }
        points.append(data)
    conn.close()
    return points


def read_edges(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM edges")
    rows = cursor.fetchall()
    points = []
    for row in rows:
        data = {
            'src': row[0],
            'des': row[1],
            'date': row[2],
            'creator': row[3],
        }
        points.append(data)
    conn.close()
    return points



def get_one_point(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT * FROM points WHERE checked = 0 ORDER BY length DESC LIMIT 1;"
    cursor.execute(query)
    rows = cursor.fetchall()
    points = []
    for row in rows:
        data = {
            'coord': eval(row[0]),
            'sid': row[1],
            'cid': row[2],
            'length': row[3],
            'checked': row[4],
        }
        points.appenk(data)
    conn.close()
    return points[0]


def get_random_point(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT * FROM points WHERE checked = 0 and length > 100 ORDER BY RANDOM() DESC LIMIT 1;"
    cursor.execute(query)
    rows = cursor.fetchall()
    points = []
    for row in rows:
        data = {
            'coord': eval(row[0]),
            'sid': row[1],
            'cid': row[2],
            'length': row[3],
            'checked': row[4],
        }
        points.append(data)
    conn.close()
    return points[0]


def get_points(db_path,ids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    points = []
    for [sid,cid] in ids:
        cursor.execute("SELECT * FROM points WHERE sid = ? AND cid = ?", (sid,cid))
        rows = cursor.fetchall()
        for row in rows:
            data = {
                'coord': eval(row[0]),
                'sid': row[1],
                'cid': row[2],
                'length': row[3],
                'checked': row[4],
            }
            points.append(data)
    conn.close()
    return points



def points2db(points,path):
    '''
    point:
        {
            coord: [x,y,z],
            sid: int,
            cid: int,
            length: int,
            checked: int
        }
    '''

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS points(
                        coord TEXT,
                        sid INTEGER,
                        cid INTEGER,
                        length INTEGER,
                        checked INTEGER,
                        PRIMARY KEY (sid, cid)
                    )''')

    query = f"SELECT COUNT(*) FROM points"
    cursor.execute(query)
    result = cursor.fetchone()
    count = result[0]

    for point in points:
        cursor.execute(f"INSERT INTO points (coord, sid, cid, length, checked) VALUES (?, ?, ?, ?, ?)",
                    (sqlite3.Binary(str(point['coord']).encode()), point['sid'], point['cid'], point['length'], point['checked']))

    print(f'Number of points in database: {count}, {len(points)} newly added.')
    conn.commit()
    conn.close()


def get_size(path):
    if not os.path.exists(path):
        return 0
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    query = f"SELECT COUNT(*) FROM segs"
    cursor.execute(query)
    result = cursor.fetchone()
    count = result[0]
    return count


def segs_from_sids(path,sids):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    segs = []
    for sid in sids:
        cursor.execute("SELECT * FROM segs WHERE sid=?", (sid,))
        rows = cursor.fetchall()
        for row in rows:
            data = {
                'sid': row[0],
                'points': eval(row[1]),
                'sampled_points': eval(row[2]),
                'nbrs': eval(row[3]),
            }
            segs.append(data)
    conn.close()
    return segs



if __name__ == '__main__':
    from ntools.segs import SegsTree
    db_path = 'tests/test.db'
    segs = read_segs(db_path)
    tree = SegsTree(segs)
    point = get_one_point(db_path)
    sids = tree.get_nbr_segs(point['coord'],dis=32)
    print(sids)
    segs = segs_from_sids(db_path,sids)
    pids = []
    for seg in segs:
        pids.append([seg['sid'],0])
        pids.append([seg['sid'],len(seg['points'])-1])

    print(pids)
    points = get_points(db_path,pids)
    for point in points:
        print(point['sid'],point['length'])


