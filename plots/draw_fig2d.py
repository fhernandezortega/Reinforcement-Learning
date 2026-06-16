import json
import matplotlib.pyplot as plt
import networkx as nx


# ==========================================
# Cargar árbol
# ==========================================

with open("tree_data_structured.json", "r") as f:

    tree = json.load(f)


# ==========================================
# Grafo dirigido
# ==========================================

G = nx.DiGraph()

node_counter = 0

# posiciones manuales tipo paper
positions = {}

# separación horizontal/vertical
DX = 1.8
DY = 1.3

# podar ramas pequeñas
MIN_BRANCH_PROB = 0.05


# ==========================================
# Función recursiva
# ==========================================

def add_tree(
    node,
    parent=None,
    edge_label="",
    x=0,
    y=0
):

    global node_counter

    prob = node["probability"]

    # --------------------------------------
    # Ignorar ramas muy pequeñas
    # --------------------------------------

    if prob < MIN_BRANCH_PROB:
        return None

    current = node_counter
    node_counter += 1

    # --------------------------------------
    # Nodo terminal
    # --------------------------------------

    if node["type"] == "terminal":

        label = f"{node['terminal_state']}"

        node_color = "#D9D9D9"

        edge_color = "gray"

        size = 900

    # --------------------------------------
    # Nodo normal
    # --------------------------------------

    else:

        label = f"{node['action']}"

        node_color = "white"

        edge_color = "red"

        size = 700

    # --------------------------------------
    # Guardar nodo
    # --------------------------------------

    G.add_node(
        current,
        label=label,
        color=node_color,
        edge=edge_color,
        prob=prob,
        size=size
    )

    positions[current] = (x, y)

    # --------------------------------------
    # Conectar con padre
    # --------------------------------------

    if parent is not None:

        G.add_edge(
            parent,
            current,
            label=edge_label,
            prob=prob
        )

    # --------------------------------------
    # Hijos
    # --------------------------------------

    if node["type"] == "node":

        children = node.get("children", {})

        # k = 0 (negro, arriba)
        if "k=0" in children:

            add_tree(
                children["k=0"],
                current,
                "k=0",
                x + DX,
                y + DY
            )

        # k = 1 (azul, abajo)
        if "k=1" in children:

            add_tree(
                children["k=1"],
                current,
                "k=1",
                x + DX,
                y - DY
            )

    return current


# ==========================================
# Construir árbol
# ==========================================

add_tree(
    tree,
    x=0,
    y=0
)


# ==========================================
# Figura
# ==========================================

plt.figure(figsize=(13, 8))


# ==========================================
# Dibujar edges
# ==========================================

for u, v, data in G.edges(data=True):

    color = (
        "black"
        if data["label"] == "k=0"
        else "deepskyblue"
    )

    nx.draw_networkx_edges(
        G,
        positions,
        edgelist=[(u, v)],
        edge_color=color,
        width=1.8,
        arrows=True,
        arrowsize=18
    )


# ==========================================
# Dibujar nodos
# ==========================================

for n in G.nodes():

    nx.draw_networkx_nodes(

        G,
        positions,

        nodelist=[n],

        node_color=G.nodes[n]["color"],

        edgecolors=G.nodes[n]["edge"],

        linewidths=1.5,

        node_size=G.nodes[n]["size"]
    )


# ==========================================
# Labels nodos
# ==========================================

labels = {

    n: G.nodes[n]["label"]

    for n in G.nodes()
}

nx.draw_networkx_labels(

    G,
    positions,

    labels,

    font_size=10
)


# ==========================================
# Probabilidades
# ==========================================

for u, v, data in G.edges(data=True):

    x1, y1 = positions[u]
    x2, y2 = positions[v]

    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2

    color = (
        "black"
        if data["label"] == "k=0"
        else "deepskyblue"
    )

    plt.text(
        xm,
        ym + 0.12,
        f"{data['prob']:.2f}",
        fontsize=10,
        color=color,
        fontweight="bold"
    )


# ==========================================
# Leyenda estilo paper
# ==========================================

plt.text(
    -0.5,
    4.5,
    r"$k=0$",
    fontsize=13,
    color="black"
)

plt.text(
    0.2,
    4.5,
    "⟶",
    fontsize=13,
    color="black"
)

plt.text(
    1.0,
    4.5,
    "termination",
    fontsize=11,
    color="black"
)

plt.text(
    -0.5,
    3.9,
    r"$k=1$",
    fontsize=13,
    color="deepskyblue"
)

plt.text(
    0.2,
    3.9,
    "⟶",
    fontsize=13,
    color="deepskyblue"
)


# ==========================================
# Estilo final
# ==========================================

plt.title(
    "Fig. 2(d) RL-QLS Decision Tree",
    fontsize=16
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "fig2d_tree.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Figura guardada como fig2d_tree.png"
)

plt.show()