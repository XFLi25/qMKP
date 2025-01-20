

import os
import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_random_graph(n, m):
    
    if m > n*(n-1)//2:
        raise ValueError("Too many edges requested for the given number of vertices.")

    G = nx.Graph()
    G.add_nodes_from(range(1, n+1))
    edges_set = set()
    while len(edges_set) < m:
        u = random.randint(1, n)
        v = random.randint(1, n)
        if u != v:
            if (u, v) not in edges_set and (v, u) not in edges_set:
                edges_set.add((u, v))

    for (u, v) in edges_set:
        G.add_edge(u, v)

    return G

def write_graph_adjacency_list(G, n, m, folder_path="/Users/xiaofanli/Desktop/Code/qMKP/qaMKP"):
    
    actual_n = G.number_of_nodes()
    actual_m = G.number_of_edges()
    filename = f"n_{actual_n}_m_{actual_m}.txt"
    filepath = os.path.join(folder_path, filename)

    with open(filepath, "w") as f:
        for node in sorted(G.nodes()):
            neighbors = sorted(list(G.neighbors(node)))
            neighbors_str = " ".join(str(nb) for nb in neighbors)
            f.write(f"{node}: {neighbors_str}\n")

    print(f"Graph adjacency list saved to: {filepath}")

def is_k_plex(G, vertices_subset, k):
    
    for v in vertices_subset:
        non_neighbors_in_P = 0
        for w in vertices_subset:
            if w == v:
                continue
            if not G.has_edge(v, w):
                non_neighbors_in_P += 1

            if non_neighbors_in_P > (k - 1):
                return False
    return True
def find_maximum_k_plex(G, k, max_iter=1000, random_starts=10):
    
    neighbors = {}
    for u in G.nodes():
        neighbors[u] = set(G.neighbors(u))

    def is_valid_k_plex(subset):
        
        for v in subset:
            non_neighbor_count = 0
            for w in subset:
                if w == v:
                    continue
                if w not in neighbors[v]:
                    non_neighbor_count += 1
                    if non_neighbor_count > (k - 1):
                        return False
        return True

    def fix_violations(subset):
        
        changed = True
        subset = set(subset)
        while changed:
            changed = False
            to_remove = []
            for v in subset:
                non_neighbor_count = 0
                for w in subset:
                    if w == v:
                        continue
                    if w not in neighbors[v]:
                        non_neighbor_count += 1
                        if non_neighbor_count > (k - 1):
                            to_remove.append(v)
                            break
            if to_remove:
                for v in to_remove:
                    subset.remove(v)
                changed = True
        return subset

    def local_search(subset):
        
        current = fix_violations(subset)   # fix it first
        best_local = set(current)
        improved = True
        iteration = 0

        while improved and iteration < max_iter:
            iteration += 1
            improved = False
            for v in list(current):
                if v not in current:
                    continue
                current.remove(v)
                current = fix_violations(current)
                candidates = list(set(G.nodes()) - current)
                random.shuffle(candidates)
                for c in candidates:
                    current.add(c)
                    if not is_valid_k_plex(current):
                        current.remove(c)
                if len(current) > len(best_local):
                    best_local = set(current)
                    improved = True
                if improved:
                    break
                else:
                    current.add(v)

        return best_local

    best_kplex_vertices = set()
    best_size = 0
    all_nodes = list(G.nodes())
    for _ in range(random_starts):
        init_size = random.randint(1, max(1, len(all_nodes) // 10))
        initial_subset = set(random.sample(all_nodes, init_size))
        candidate = local_search(initial_subset)
        candidate_size = len(candidate)
        if candidate_size > best_size:
            best_size = candidate_size
            best_kplex_vertices = candidate

    return best_size, list(best_kplex_vertices)


def visualize_kplex(G, kplex_vertices, k, n, m, folder_path="/Users/xiaofanli/Desktop/Code/qMKP/qaMKP"):
    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=300)
    nx.draw_networkx_labels(G, pos, font_color="black")
    subG = G.subgraph(kplex_vertices)
    nx.draw_networkx_edges(subG, pos, edge_color="red", width=2.0)
    nx.draw_networkx_nodes(subG, pos, node_color="red", node_size=300)
    nx.draw_networkx_labels(subG, pos, font_color="white")

    actual_n = G.number_of_nodes()
    actual_m = G.number_of_edges()
    filename = f"n_{actual_n}_m_{actual_m}_k_{k}.png"
    filepath = os.path.join(folder_path, filename)
    plt.title(f"Maximum k-Plex (k={k})")
    plt.axis("off")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Figure saved: {filepath}")

def main():
    n = 30   # number of vertices
    m = 300  # number of edges
    G = generate_random_graph(n, m)
    write_graph_adjacency_list(G, n, m)
    for k_val in [2, 3, 4, 5]:
        size_kplex, kplex_vertices = find_maximum_k_plex(G, k_val)
        print(f"k={k_val}, maximum k-plex size = {size_kplex}, vertices: {kplex_vertices}")
        visualize_kplex(G, kplex_vertices, k_val, n, m)
if __name__ == "__main__":
    main()
