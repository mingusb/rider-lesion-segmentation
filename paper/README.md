# Paper and complete tree figures

[Read the PDF](rider-neural-decision-programs.pdf). The manuscript uses the ICML 2026 style with named authors and no proceedings notice; use of that style does not imply conference acceptance. It has two body pages followed by references.

Page 1 shows both complete ordinary binary trees side by side: 21 nodes for the earlier refit and 181 nodes for the current refit. Shared continuations are expanded, with no omitted branches. The predicate key is on page 2. The tree panels use neon cyan (`#00E5FF`) and lemon yellow (`#FFEE32`) on deep navy (`#081A32`), with crisp vector strokes and a small glow. The rest of the paper remains white. Leaf labels **B** (background) and **L** (lesion), plus solid true edges and dashed false edges, make the meaning readable without relying on color. The caption explains missing-value routing.

## Build the manuscript

The main TeX file is self-contained: it embeds the required style material, figure content and references. With LuaLaTeX and its standard packages installed, run twice from this directory:

```sh
lualatex -interaction=nonstopmode -halt-on-error rider-neural-decision-programs.tex
lualatex -interaction=nonstopmode -halt-on-error rider-neural-decision-programs.tex
```

No patient data, CNN checkpoint or model training is needed to build the paper. Embedded third-party style notices remain applicable.

## Regenerate the editable tree fragments

`layout_complete_tree.py` uses Python 3 and Graphviz `dot`. Its positional arguments are:

```text
python3 layout_complete_tree.py SOURCE_TREE_JSON OUTPUT_TEX [TARGET_HEIGHT_IN] [ORIENTATION] [MAX_WIDTH_IN] [FONT_POINTS]
```

The final figure invocations, from this directory, are:

```sh
python3 layout_complete_tree.py ../models/earlier/binary-tree.json earlier-complete-tree.tex 2.7 portrait 1.90 7
python3 layout_complete_tree.py ../models/current/binary-tree.json latest-complete-tree.tex 2.2 portrait 4.42 6.5 current-layout-adjustments.json
```

The renderer writes an editable TikZ fragment and layout metadata. It preserves every tree node, the saved FP32 predicate and its missing direction. Layout dimensions affect presentation only. The standalone fragments are supplied for editing; after changing one, update the corresponding embedded figure in the self-contained manuscript before recompiling.

Performance numbers describe cross-validation procedures, while the figures depict final all-patient refits. See [results](../results/README.md), [model cards](../models/README.md) and [license boundaries](../THIRD_PARTY_NOTICES.md).
