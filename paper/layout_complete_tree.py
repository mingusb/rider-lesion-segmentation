"""Render every node of a saved ordinary binary tree as compact, editable TikZ."""
from pathlib import Path
import json, subprocess, shlex, sys
from decimal import Decimal

source = Path(sys.argv[1])
out = Path(sys.argv[2])
tree = json.loads(source.read_text())
nodes, edges, thresholds = [], [], {}
predicates = {}
large = tree['nodes'] > 50
vertical = large or (len(sys.argv) > 4 and sys.argv[4] == 'portrait')
maximum_width = float(sys.argv[5]) if len(sys.argv) > 5 else 6.65
font_points = float(sys.argv[6]) if len(sys.argv) > 6 else (7.5 if large else 8.5)
optical_offsets = json.loads(Path(sys.argv[7]).read_text()) if len(sys.argv) > 7 else {}
feature_names = {0: 'x', 1: 'u', 4: 's', 19: 'v', 20: 'm', 21: 'p', 43: 'a_{21}', 44: 'a_{22}', 50: 'a_{28}'}
used_features, missing_true = set(), []

def visit(node):
    ident = node['id']
    if 'class' in node:
        label = r'\mathbf{L}' if node['class'] == 1 else r'\mathbf{B}'
        color = 'leafteal' if node['class'] == 1 else 'leafgray'
        plain = 'L' if node['class'] == 1 else 'B'
    else:
        used_features.add(node['feature'])
        feature = feature_names.get(node['feature'], 'f_{%d}' % node['feature'])
        if node['threshold'] == 1 and node['feature'] == 20:
            threshold = '1'
        else:
            key = node['raw_threshold_bits']
            if key not in thresholds:
                thresholds[key] = (len(thresholds)+1, node['threshold_text'])
            threshold = r'\tau_{%d}' % thresholds[key][0]
        label = feature + '<' + threshold
        predicate_key = (node['feature'], node['raw_threshold_bits'], node['missing_goes_left'])
        if predicate_key not in predicates:
            predicates[predicate_key] = (len(predicates)+1, feature, node['threshold_text'])
        predicate_index = predicates[predicate_key][0]
        if large: label = r'P_{%d}' % predicate_index
        if node['missing_goes_left']:
            missing_true.append(ident)
            label += r'\;{}^{\star}'
        color = 'rootpurple' if node is tree['root'] else 'treeink'
        plain = feature.replace('{','').replace('}','') + '<t' + str(len(thresholds))
        if large: plain = 'P'+str(predicate_index)
        for side in ('left', 'right'):
            edges.append((ident, node[side]['id'], side))
            visit(node[side])
    nodes.append((ident, label, color, plain))

visit(tree['root'])
assert len(nodes) == tree['nodes'] and len(edges) == len(nodes)-1
assert len(set(n[0] for n in nodes)) == len(nodes)
dot = ['digraph T {', ('graph [rankdir=TB, ordering=out, nodesep=0.035, ranksep=0.10, margin=0];' if vertical else 'graph [rankdir=LR, ordering=out, nodesep=0.16, ranksep=0.24, margin=0];'),
       'node [shape=plain, fontname="Times-Roman", fontsize=%d];' % (8 if large else 9)]
dot += [f'n{i} [label="{plain}"];' for i, _, _, plain in nodes]
dot += [f'n{a} -> n{b};' for a,b,_ in edges]
dot += ['}']
out.with_suffix('.dot').write_text('\n'.join(dot)+'\n')
plain = subprocess.check_output(['dot','-Tplain'],input='\n'.join(dot).encode()).decode()
positions = {}
for line in plain.splitlines():
    words = shlex.split(line)
    if words[0] == 'graph': W,H = float(words[2]),float(words[3])
    if words[0] == 'node': positions[int(words[1][1:])] = (float(words[2]),float(words[3]))
scale = min(1., maximum_width/W)
target_height = float(sys.argv[3]) if len(sys.argv) > 3 else (3.5 if large else 2.0)
yscale = min(1., target_height/H)
font = r'\fontsize{%g}{%g}\selectfont' % (font_points, font_points + 1)
lines = [r'\begin{tikzpicture}[x=1in,y=1in, every node/.style={inner sep=.6pt,outer sep=0pt,font='+font+'},',
         r'trueedge/.style={draw=treeblue,line width=.6pt,preaction={draw=treeblue,opacity=.08,line width=1.4pt}},falseedge/.style={draw=treeorange,densely dashed,line width=.6pt,preaction={draw=treeorange,opacity=.08,line width=1.4pt}}]']
for ident,label,color,_ in nodes:
    x,y = positions[ident]
    dx = optical_offsets.get('horizontal_points', {}).get(str(ident), 0) / 72.27
    lines.append(r'\node[text=%s] (n%d) at (%.5f,%.5f) {$%s$};' % (color,ident,x*scale+dx,y*yscale,label))
for a,b,side in edges:
    style = 'trueedge' if side == 'left' else 'falseedge'
    anchors = ('south','north') if vertical else ('east','west')
    lines.append(r'\draw[%s] (n%d.%s) -- (n%d.%s);' % (style,a,anchors[0],b,anchors[1]))
lines.append(r'\end{tikzpicture}')
nan_legend = r'NaN follows true at $\star$, false elsewhere.' if missing_true else 'NaN follows false.'
lines += [r'\par\smallskip', r'{\footnotesize\color{treeink}',
          r'\textcolor{treeblue}{Solid: true}\quad\textcolor{treeorange}{Dashed: false}\quad $\mathbf B$: background\quad $\mathbf L$: lesion. '+nan_legend+r'\par']
descriptions = {0:'normalized CT intensity',1:'local CT mean (2-mm radius)',4:'local CT standard deviation (4-mm radius)',19:'CT intensity at the supplied point',20:'CNN mask',21:'CNN foreground probability',43:'decoder activation 21',44:'decoder activation 22',50:'decoder activation 28'}
legend = []
for feature in sorted(used_features):
    label = feature_names.get(feature, 'f_{%d}' % feature)
    desc = descriptions.get(feature, 'input coordinate %d' % feature)
    legend.append('$'+label+'$: '+desc)
if large and used_features.issubset(feature_names.keys()):
    lines.append(r'$x$: normalized CT intensity; $u,s$: local mean (2-mm radius) and standard deviation (4-mm radius); $v$: intensity at the supplied point; $m,p$: CNN mask and probability; $a_k$: decoder channel $k$.\par')
else:
    lines.append('; '.join(legend)+r'.\par')
values = []
if large:
    for index,feature,value in predicates.values():
        val = format(Decimal(value),'f')
        values.append(r'$P_{%d}:\;%s<%s$' % (index,feature,val))
else:
    for index,value in thresholds.values():
        val = format(Decimal(value),'f')
        values.append(r'$\tau_{%d}=%s$' % (index,val))
for i in range(0,len(values),4):
    lines.append(r'\quad '.join(values[i:i+4]) + r'\par')
lines.append('}')
out.write_text('\n'.join(lines)+'\n')
out.with_suffix('.layout.json').write_text(json.dumps({'source':str(source),'nodes':len(nodes),'edges':len(edges),'unscaled_width_inches':W,'unscaled_height_inches':H,'width_inches':W*scale,'height_inches':H*yscale,'threshold_count':len(thresholds),'predicate_count':len(predicates),'missing_true_nodes':missing_true},indent=2)+'\n')
print(json.dumps({'nodes':len(nodes),'edges':len(edges),'width_inches':W*scale,'height_inches':H*yscale}))
