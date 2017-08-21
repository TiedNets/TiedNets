__author__ = 'Agostino Sturaro'

import os
import json
import networkx as nx
import matplotlib.pyplot as plt

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

subs_G = nx.Graph()
final_G = nx.Graph()
lines_by_id = dict()
point_to_id = dict()

com_lines_fpath = os.path.normpath('datasets/ComLines.geojson')
parsed_graph_fpath = os.path.normpath('MN_pow.graphml')

this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)

if not os.path.isabs(com_lines_fpath):
    com_lines_fpath = os.path.abspath(com_lines_fpath)
if not os.path.isabs(parsed_graph_fpath):
    parsed_graph_fpath = os.path.abspath(parsed_graph_fpath)

with open(com_lines_fpath) as com_lines_file:
    # elec_lines = json.load(com_lines_file, parse_float=Decimal)
    com_lines = json.load(com_lines_file)
    for line in com_lines['features']:
        # these ids may not start from 1 and may not be continuous, but they should be unique
        # AFAIK, GeoJSON does not mandate a specific id property, so we just use our own
        line_id = line['properties']['id']

        if line['geometry'] is None:
            print('Missing geometry for line {}'.format(line_id))  # debug
            continue

        if line_id in lines_by_id:
            print('Duplicated line id {}, this line will be skipped!'.format(line_id))  # warning
            continue

        line_attrs = dict()

        # remember the list of coordinates as a list of tuples (lat, long)
        line_attrs['points'] = list()
        for coords in line['geometry']['coordinates']:
            point = tuple(coords)
            line_attrs['points'].append(point)

        lines_by_id[line_id] = line_attrs

for line_id in lines_by_id:
    line_attrs = lines_by_id[line_id]
    line_points = line_attrs['points']

    # add line points as nodes to the graph
    for point in line_points:
        if point not in point_to_id:
            point_id = len(point_to_id)
            point_to_id[point] = point_id
            final_G.add_node(point_id, attr_dict={'x': point[0], 'y': point[1]})

    # connect consecutive line points to form arcs in the graph
    for idx in range(0, len(line_points) - 1):
        node = point_to_id[line_points[idx]]
        other_node = point_to_id[line_points[idx + 1]]
        final_G.add_edge(node, other_node)

    lines_by_id[line_id] = line_attrs

print('len(lines) {}'.format(len(lines_by_id)))  # debug

# throw away isolated components (this step is optional)
components = sorted(nx.connected_components(final_G), key=len, reverse=True)
for component_idx in range(1, len(components)):
    print('isolated component {} = {}'.format(component_idx, components[component_idx]))
    final_G.remove_nodes_from(components[component_idx])
print('node count without isolated components = {}'.format(final_G.number_of_nodes()))

# export graph in GraphML format
nx.write_graphml(final_G, 'MN_com.graphml')

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

margin = 0.01
delta_x = abs(x_max - x_min)
delta_y = abs(y_max - y_min)

print('x_min = {}\nx_max = {}\ny_min = {}\ny_max = {}'.format(x_min, x_max, y_min, y_max))
plt.xlim(x_min - margin * delta_x, x_max + margin * delta_x)
plt.ylim(y_min - margin * delta_y, y_max + margin * delta_y)
nx.draw_networkx(final_G, pos, with_labels=False, node_size=2, linewidths=0.0)
plt.show()
plt.close()
