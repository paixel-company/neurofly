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
            type: int,
            checked: int
        }
    edge:
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
                type INTEGER,
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
        cursor.execute(f"INSERT INTO nodes (nid, coord, type, checked) VALUES (?, ?, ?, ?)",
                    (node[0], sqlite3.Binary(str(node[1]).encode()), 0, 0))

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
            'type': row[2],
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
    edges = []
    for row in rows:
        data = {
            'src': row[0],
            'des': row[1],
            'date': row[2],
            'creator': row[3],
        }
        edges.append(data)
    conn.close()
    return edges


def augment_nodes(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    table_name = 'nodes'
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    column_names = [col_info[1] for col_info in columns_info]
    conn.close()
    if 'type' in column_names:
        return
        
    # replace nbr column with type column of nodes table
    nodes = read_nodes(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    drop_command = f"DROP TABLE IF EXISTS nodes;"
    cursor.execute(drop_command)

    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS nodes(
                nid INTEGER PRIMARY KEY,
                coord TEXT,
                type INTEGER,
                checked INTEGER
            )
            '''
        )
    for node in nodes:
        cursor.execute(f"INSERT INTO nodes (nid, coord, type, checked) VALUES (?, ?, ?, ?)",(node['nid'], sqlite3.Binary(str(node['coord']).encode()), 0, node['checked'])) 

    conn.commit()
    conn.close()


def delete_nodes(path,node_ids):
    # given a list of node_ids, delete nodes from nodes table and edges from edges table
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM nodes WHERE nid IN ({})".format(','.join(map(str, node_ids))))
    # Remove edges where either source or destination node is in the given list
    cursor.execute("DELETE FROM edges WHERE src IN ({}) OR des IN ({})".format(','.join(map(str, node_ids)), ','.join(map(str, node_ids))))
    conn.commit()
    conn.close()


def add_nodes(path,nodes):
    # given a list of nodes, write them to node table
    # nodes: [{'nid','coord','type','checked'}]
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    for node in nodes:
        cursor.execute(f"INSERT OR IGNORE INTO nodes (nid, coord, type, checked) VALUES (?, ?, ?, ?)",
                    (node['nid'], sqlite3.Binary(str(node['coord']).encode()), node['type'], 0))
    conn.commit()
    conn.close()


def add_edges(path, edges, user_name='seger'):
    # given list of edges, write them to edges table
    # edges: [[src,tar]]
    undirected_edges = []
    for [src,tar] in edges:
        undirected_edges.append([src,tar])
        undirected_edges.append([tar,src])

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    for edge in undirected_edges:
        cursor.execute(f"INSERT OR IGNORE INTO edges (src, des, date, creator) VALUES (?, ?, ?, ?)",
                    (edge[0], edge[1], datetime.now(), 'seger'))
    conn.commit()
    conn.close()


def check_node(path,nid):
    # given list of edges, write them to edges table
    # edges: [[src,tar]]
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("UPDATE OR IGNORE nodes SET checked = 1 WHERE nid = ?", (nid,))

    conn.commit()
    conn.close()


def uncheck_nodes(path,nids):
    # given list of node ids, label them as unchecked
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("UPDATE OR IGNORE nodes SET checked = -1 WHERE nid IN ({})".format(','.join(map(str, nids))))

    conn.commit()
    conn.close()


def change_type(path,nid,type):
    # given list of node ids, label them as unchecked
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("UPDATE nodes SET type = ? WHERE nid = ?", (type, nid))

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





if __name__ == '__main__':
    # db_path = 'tests/z002_re.db'
    # nodes = read_nodes(db_path)
    # import numpy as np
    # coords = []
    # for node in nodes:
    #     coords.append(node['coord'])
    # coords = np.array(coords)
    # for i in range(3):
    #     print(np.min(coords[:,i]),np.max(coords[:,i]-np.min(coords[:,i])))

    # db_path = 'tests/z002.db'
    # augment_nodes(db_path)
    # nodes = read_nodes(db_path)
    # print(nodes)

    db_path = 'tests/z002_final.db'
    save_lyp(db_path,template_path='src/weights/template.lyp')
