

import os
os.environ["DWAVE_API_TOKEN"] = "yourtoken"
import math
import time
import numpy as np
import networkx as nx
import dimod
from dwave.system import DWaveSampler, LeapHybridSampler

def load_graph(n, m, folder_path="/Users/xiaofanli/Desktop/Code/qMKP/qaMKP"):
    
    target_prefix = f"n_{n}_m_{m}"
    chosen_file = None

    for fname in os.listdir(folder_path):
        if fname.startswith(target_prefix) and fname.endswith(".txt"):
            chosen_file = fname
            break

    if not chosen_file:
        raise FileNotFoundError(f"No file matching 'n_{n}_m_{m}' found in {folder_path}")

    filepath = os.path.join(folder_path, chosen_file)
    print(f"Loading graph from: {filepath}")

    G = nx.Graph()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            node_str = parts[0].strip()
            if len(parts) < 2:
                neighbors_str = ""
            else:
                neighbors_str = parts[1].strip()

            if not node_str.isdigit():
                continue
            node = int(node_str)

            if not G.has_node(node):
                G.add_node(node)

            if neighbors_str:
                neighbor_list = neighbors_str.split()
                for nb in neighbor_list:
                    nb_int = int(nb)
                    G.add_edge(node, nb_int)

    return G

def build_qubo_for_mkp(G, k, R):
    
    n = G.number_of_nodes()
    if k > n:
        raise ValueError("k cannot exceed the number of vertices.")
    if R <= 1.0:
        raise ValueError("R must be > 1.0")
    M = n - k + 1
    L = int(math.floor(math.log2(n))) if n>1 else 1
    bqm = dimod.BinaryQuadraticModel('BINARY')
    alpha = 1.0
    x_index = {}
    for i in sorted(G.nodes()):
        var_name = f"x_{i}"
        x_index[i] = var_name
        bqm.add_variable(var_name, 0.0)  # add to BQM with initial bias=0
    s_index = {}
    for i in sorted(G.nodes()):
        for r in range(L):
            var_name = f"s_{i}_{r}"
            s_index[(i, r)] = var_name
            bqm.add_variable(var_name, 0.0)
    for i in G.nodes():
        bqm.set_linear(x_index[i], bqm.get_linear(x_index[i]) - alpha)
    neighbors = {}
    for i in G.nodes():
        neighbors[i] = set(G.neighbors(i))

    for i in G.nodes():
        non_neighbors = []
        for nd in G.nodes():
            if nd == i:
                continue
            if nd not in neighbors[i]:
                non_neighbors.append(nd)
        constant_part = - (k - 1) - M
        linear_terms = {x_index[i]: float(M)}
        for j in non_neighbors:
            linear_terms[x_index[j]] = linear_terms.get(x_index[j], 0.0) + 1.0
        for r in range(L):
            var = s_index[(i, r)]
            coeff = 2.0**r
            linear_terms[var] = linear_terms.get(var, 0.0) + coeff
        bqm_offset = bqm.offset
        bqm.offset = bqm_offset + R*(constant_part**2)
        for var, c_v in linear_terms.items():
            lin_bias = bqm.get_linear(var)
            lin_bias += R * (2.0 * constant_part * c_v)
            bqm.set_linear(var, lin_bias)
        all_vars = sorted(linear_terms.keys())
        for idx1 in range(len(all_vars)):
            v1 = all_vars[idx1]
            c1 = linear_terms[v1]
            diag_bias = bqm.get_linear(v1)
            diag_bias += R * (c1**2)
            bqm.set_linear(v1, diag_bias)

            for idx2 in range(idx1+1, len(all_vars)):
                v2 = all_vars[idx2]
                c2 = linear_terms[v2]
                quad_coeff = R * (2.0 * c1 * c2)
                bqm.add_quadratic(v1, v2, quad_coeff)

    return bqm, x_index, s_index

def is_k_plex(G, subset, k):
    
    subset_set = set(subset)
    for v in subset_set:
        non_neighbors_in_subset = 0
        for w in subset_set:
            if w == v:
                continue
            if not G.has_edge(v, w):
                non_neighbors_in_subset += 1
                if non_neighbors_in_subset > (k - 1):
                    return False
    return True

def solve_qubo_on_dwave(bqm, use_hybrid=True, token="yourtoken", sampler_params=None):
    
    import dwave.cloud
    dwave.cloud.Client.from_config(token=token)

    start_time = time.time()

    if use_hybrid:
        sampler = LeapHybridSampler()
        if sampler_params:
            sampleset = sampler.sample(bqm, **sampler_params)
        else:
            sampleset = sampler.sample(bqm)
    else:
        sampler = DWaveSampler()
        if sampler_params:
            sampleset = sampler.sample(bqm, **sampler_params)
        else:
            sampleset = sampler.sample(bqm)

    end_time = time.time()
    runtime_seconds = end_time - start_time

    return sampleset, runtime_seconds

def main():
    n = 8   # number of vertices (must match the file in /Users/xiaofanli/Desktop/Code/qMKP/qaMKP)
    m = 10  # number of edges (also must match the file naming: n_8_m_10.txt or similar)
    k = 3   # k in k-plex
    R = 2.0 # Must be > 1
    use_hybrid_solver = True   # set False to use quantum hardware
    token = "yourtoken"
    sampler_params = {'time_limit': 5}  # example for hybrid
    G = load_graph(n, m)
    bqm, x_index, s_index = build_qubo_for_mkp(G, k, R)
    sampleset, runtime_sec = solve_qubo_on_dwave(
        bqm=bqm,
        use_hybrid=use_hybrid_solver,
        token=token,
        sampler_params=sampler_params
    )

    print("QUBO solved. D-Wave run time: {:.4f} seconds".format(runtime_sec))
    best = sampleset.first.sample
    energy = sampleset.first.energy
    print(f"Best energy found: {energy}")
    chosen_vertices = []
    for i in G.nodes():
        if best[x_index[i]] == 1:
            chosen_vertices.append(i)
    valid = is_k_plex(G, chosen_vertices, k)
    if valid:
        print(f"Solution subgraph is a valid k-plex with size = {len(chosen_vertices)}")
    else:
        print(f"Solution subgraph is NOT a valid k-plex. size = {len(chosen_vertices)}")
    print("Chosen vertices:", chosen_vertices)


if __name__ == "__main__":
    main()
