from ntools.dbio import read_nodes, read_edges
import networkx as nx
import json
import os


def save_lyp(path,template_path,size_thres=300):
    '''
    load lyp template
    get all large connected components, save each as a single .lyp file
    save undirected graph, i.e. add all neighbours to 'parent_ids' except soma node
    for soma node, add "#Soma" to 'message'
    node template:
    {
        "creator_id": 0,
        "group_id": "1",
        "id": "5030239",
        "message": "", # if node is soma node, "#Soma" else ""
        "parent_ids": "5030238",
        "position": "49722.937 32652.702 44400.602",
        "timestamp": "1693814131" #unix timestamp
    }
    '''
    with open(template_path, "r") as json_file:
        template_dict = json.load(json_file)

    nodes = read_nodes(path)
    edges = read_edges(path)
    G = nx.Graph()

    for node in nodes:
        G.add_node(node['nid'], nid = node['nid'],coord = node['coord'], type = node['type'], checked = node['checked'])
    for edge in edges:
        G.add_edge(edge['src'],edge['des'],creator = edge['creator'])
    
    # find all connected components larger than threshold
    connected_components = list(nx.connected_components(G))

    count = 0
    for cc in connected_components:
        if len(cc)>size_thres:
            count+=1
            nodes = [G.nodes[i] for i in cc]
            n_list = []
            for node in nodes:
                position = ' '.join(str(i) for i in node['coord'])
                message = "#Soma" if node['type']==1 else ""
                p_ids = ' '.join(str(i) for i in list(G.neighbors(node['nid'])))
                node_dict = {
                    'creator_id': 0,
                    'group_id': "1",
                    'id': str(node['nid']),
                    'message': message,
                    'parent_ids': p_ids,
                    'position': position,
                    'timestamp': str(1628497783)
                }
                n_list.append(node_dict)
            lyp_dict = template_dict.copy()
            lyp_dict['nodes'] = n_list

            output_dir = 'tests/lyps/'
            file_name = str(count)+'.lyp'
            file_path = os.path.join(output_dir,file_name)
            with open(file_path, "w") as json_file:
                json.dump(lyp_dict, json_file, indent=4)



if __name__ == '__main__':
    db_path = 'tests/z002_final.db'
    save_lyp(db_path,template_path='src/weights/template.lyp')
