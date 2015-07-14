__author__ = 'sturaroa'

from decimal import *
import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q


def find_neighboring_subs(G, v):
    subs = list()
    q = Q.Queue()
    q.put(v)
    discovered = list()
    discovered.append(v)
    while not q.empty():
        u = q.get()
        # print('G.neighbors(u) = {}'.format(G.neighbors(u)))  # debug
        for w in G.neighbors(u):
            if w not in discovered:
                discovered.append(w)
                # print(G.node[w]['type'])
                if len(G.node[w]['sub_ids']) > 0:
                    subs.append(w)  # you reached a neighboring substation, don't search beyond it
                else:
                    q.put(w)  # keep searching
    # print('subs ' + str(subs))  #debug
    return subs


subs_G = nx.Graph()
final_G = nx.Graph()
subs = dict()
lines_by_id = dict()
point_to_id = dict()

elec_subs_fpath = 'datasets/ElecSubs_epsg_4326.geojson'
elec_lines_fpath = 'datasets/ElecLine_epsg_4326.geojson'

with open(elec_subs_fpath) as elec_subs_file, open(elec_lines_fpath) as elec_lines_file:
    # add nodes to the graph
    # elec_subs = json.load(elec_subs_file, parse_float=Decimal)
    elec_subs = json.load(elec_subs_file)

    print('sub_cnt features ' + str(len(elec_subs['features'])))  # debug

    for sub in elec_subs['features']:

        if sub['geometry'] is None:
            print('Missing geometry for sub with OBJECTID ' + str(sub['properties']['OBJECTID']))  # debug
            continue

        sub_id = sub['properties']['OBJECTID']  # these ids probably start from 1 and may not be continuous
        if sub_id in subs:
            print('Duplicated sub OBJECTID ' + str(sub_id))  # debug
            continue

        sub_attrs = dict()
        sub_attrs['COMPANY'] = sub['properties']['COMPANY']
        sub_attrs['COMP_ID'] = sub['properties']['COMP_ID']
        sub_attrs['SUB_TYPE'] = sub['properties']['SUB_TYPE']

        # remember coordinates as a tuple (lat, long)
        point = tuple(sub['geometry']['coordinates'])
        sub_attrs['coordinates'] = point

        # save the properties of the substation indexing with by the substation OBJECTID
        subs[sub_id] = sub_attrs

        # remember that this substation is found at this point
        # there may be more than 1 substation in the same point

        if point not in point_to_id:
            point_id = len(point_to_id)
            point_to_id[point] = point_id
            subs_G.add_node(point_id, attr_dict={'x': point[0], 'y': point[1], 'sub_ids': list()})

        point_id = point_to_id[point]
        subs_G.node[point_id]['sub_ids'].append(sub_id)

    print('len(subs) {}'.format(len(subs)))  # debug

    final_G.add_nodes_from(subs_G.nodes(data=True))  # copy nodes representing substation locations to a multigraph

    for node in subs_G.nodes():
        if len(subs_G.node[node]['sub_ids']) > 1:
            print('Group of substations with the same coords ' + str(subs_G.node[node]['sub_ids']))

    # elec_lines = json.load(elec_lines_file, parse_float=Decimal)
    voltages = list()
    elec_lines = json.load(elec_lines_file)
    for line in elec_lines['features']:

        if line['geometry'] is None:
            print('Missing geometry for line with OBJECTID ' + str(line['properties']['OBJECTID']))  # debug
            continue

        line_id = line['properties']['OBJECTID']  # these ids probably start from 1 and may not be continuous
        if line_id in lines_by_id:
            print('Duplicated line OBJECTID ' + str(line_id))  # debug
            continue

        line_attrs = dict()
        line_attrs['COMPANY'] = line['properties']['COMPANY']
        line_attrs['COMP_ID'] = line['properties']['COMP_ID']
        line_attrs['ACDC'] = line['properties']['ACDC']

        voltage = line['properties']['VOLTAGE']
        line_attrs['VOLTAGE'] = voltage
        if voltage not in voltages:
            voltages.append(voltage)

        # remember the list of coordinates as a list of tuples (lat, long)
        line_attrs['points'] = list()
        for coords in line['geometry']['coordinates']:
            point = tuple(coords)
            line_attrs['points'].append(point)

        lines_by_id[line_id] = line_attrs

    print('len(lines) {}'.format(len(lines_by_id)))  # debug

for voltage in voltages:

    # make a graph consisting of the points that make up the electric lines

    temp_G = subs_G.copy()  # start by copying substation positions

    for line_id in lines_by_id:
        line_attrs = lines_by_id[line_id]

        if line_attrs['VOLTAGE'] != voltage:
            continue

        # remember that this line junction is found at this point
        # there may be more than 1 line junction in the same point

        for point in line_attrs['points']:

            if point not in point_to_id:
                point_id = len(point_to_id)
                point_to_id[point] = point_id
            else:
                point_id = point_to_id[point]

            if point_id not in temp_G.nodes():
                node_attrs = {'x': point[0], 'y': point[1], 'sub_ids': []}
                temp_G.add_node(point_id, attr_dict=node_attrs)

    # connect nodes that appear as consecutive points on the same transmission line

    for line_id in lines_by_id:
        line_attrs = lines_by_id[line_id]

        if line_attrs['VOLTAGE'] != voltage:
            continue

        # proceed linking consecutive points: 0 with 1, 1 with 2, etc.
        line_points = line_attrs['points']
        for idx in range(0, len(line_points)):
            if idx <= len(line_points) - 2:
                node = point_to_id[line_points[idx]]
                other_node = point_to_id[line_points[idx + 1]]
                if not temp_G.has_edge(node, other_node):
                    temp_G.add_edge(node, other_node, attr_dict={'line_ids': list()})
                temp_G.edge[node][other_node]['line_ids'].append(line_id)

    print('Consecutive line points linked')  # debug

    # add substation nodes to a new, simpler graph

    voltage_G = nx.Graph()
    sub_cnt = 0  # debug
    for node in temp_G.nodes():
        if len(temp_G.node[node]['sub_ids']) > 0:  # if this node is also a substation
            sub_cnt += 1  # debug
            voltage_G.add_node(node, attr_dict=dict(temp_G.node[node]))  # deep copy of the node attributes
            # print('G.node[node] = ' + str(G.node[node]))  # debug

    print('sub_cnt ' + str(sub_cnt))  # debug

    second_hits = 0
    for node in voltage_G.nodes():
        neighboring_subs = find_neighboring_subs(temp_G, node)  # search in the other graph
        for neighbor in neighboring_subs:
            if not voltage_G.has_edge(node, other_node):
                voltage_G.add_edge(node, neighbor)  # TODO: think about some data for the edge (like voltage)
            else:
                second_hits += 1

    print('Voltage = {}, second_hits = {}'.format(voltage, second_hits))  # debug

    # small test
    point_sub_24 = subs[24]['coordinates']
    node_sub_24 = point_to_id[point_sub_24]
    point_sub_1112 = subs[1112]['coordinates']
    node_sub_1112 = point_to_id[point_sub_1112]

    if node_sub_24 in voltage_G.neighbors(node_sub_1112):
        print('subs 24 and 1112 correctly connected')

    # another small test
    point_sub_93 = subs[93]['coordinates']
    node_sub_93 = point_to_id[point_sub_93]
    point_sub_105 = subs[105]['coordinates']
    node_sub_105 = point_to_id[point_sub_105]

    if node_sub_93 in voltage_G.neighbors(node_sub_105):
        print('subs 93 and 105 correctly connected')

    # copy graph edges (if they are already there, no problem)
    final_G.add_edges_from(voltage_G.edges())  # TODO: find a way to add edge data to a list

# throw away isolated components (this step is optional)
components = sorted(nx.connected_components(final_G), key=len, reverse=True)
for component_idx in range(1, len(components)):
    print('isolated component {} = {}'.format(component_idx, components[component_idx]))
    final_G.remove_nodes_from(components[component_idx])
print('node count without isolated components = {}'.format(final_G.number_of_nodes()))

# since GraphML does not support attributes with list values, we convert them to strings
for node in final_G.nodes():
    final_G.node[node]['sub_ids'] = str(final_G.node[node]['sub_ids'])

# export graph in GraphML format
nx.write_graphml(final_G, 'MN_pow.graphml')

# draw the final graph

pos = dict()
first_it = True
for node in final_G.nodes():
    x = final_G.node[node]['x']
    y = final_G.node[node]['y']
    pos[node] = [x, y]

    if first_it is True:
        x_min = x
        y_min = y
        x_max = x
        y_max = y
        first_it = False
    else:
        if x > x_max:
            x_max = x
        elif x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        elif y < y_min:
            y_min = y

margin = 0.2
delta_x = abs(x_max - x_min)
delta_y = abs(y_max - y_min)

print('x_min = {}\nx_max = {}\ny_min = {}\ny_max = {}'.format(x_min, x_max, y_min, y_max))
# plt.xlim(x_min - margin * delta_x, x_max + margin * delta_x)
# plt.xlim(y_min - margin * delta_y, y_max + margin * delta_y)
nx.draw_networkx(final_G, pos, with_labels=False, node_size=2, linewidths=0.0)
plt.show()
plt.close()
