

import os
os.environ["DWAVE_API_TOKEN"] = "yourtoken"
import math
import time
import numpy as np
import networkx as nx
import dimod
from dwave.system import DWaveSampler, LeapHybridSampler, EmbeddingComposite
from dimod.reference.samplers import ExactSolver
from dimod import SimulatedAnnealingSampler
from minorminer import find_embedding


def load_graph(n, m, folder_path="/Users/xiaofanli/Desktop/Code/qMKP/qaMKP"):
    
    prefix = f"n_{n}_m_{m}"
    chosen_file = None

    for fname in os.listdir(folder_path):
        if fname.startswith(prefix) and fname.endswith(".txt"):
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
            neighbors_str = parts[1].strip() if len(parts) > 1 else ""
            if node_str.isdigit():
                node = int(node_str)
                if not G.has_node(node):
                    G.add_node(node)
                if neighbors_str:
                    for nb in neighbors_str.split():
                        nb_int = int(nb)
                        G.add_edge(node, nb_int)

    return G


def build_qubo_for_mkp(G, k, R):
    
    n = G.number_of_nodes()
    if k > n:
        raise ValueError("k cannot be larger than the number of vertices.")
    if R <= 1:
        raise ValueError("R must be > 1.")
    M = n - k + 1
    L = int(math.ceil(math.log2(n))) if n > 1 else 1
    alpha = 1.0

    bqm = dimod.BinaryQuadraticModel('BINARY')
    var_x_index = {}
    var_s_index = {}
    for i in sorted(G.nodes()):
        x_name = f"x_{i}"
        var_x_index[i] = x_name
        bqm.add_variable(x_name, 0.0)
    for i in sorted(G.nodes()):
        for r in range(L):
            s_name = f"s_{i}_{r}"
            var_s_index[(i, r)] = s_name
            bqm.add_variable(s_name, 0.0)
    for i in G.nodes():
        bqm.set_linear(var_x_index[i], bqm.get_linear(var_x_index[i]) - alpha)
    neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}
    for i in G.nodes():
        non_neighbors = []
        for nd in G.nodes():
            if nd == i:
                continue
            if nd not in neighbors[i]:
                non_neighbors.append(nd)
        constant_part = -(k - 1) - M
        linear_terms = {}  # dict var -> coefficient
        for j in non_neighbors:
            linear_terms[var_x_index[j]] = linear_terms.get(var_x_index[j], 0.0) + 1.0
        linear_terms[var_x_index[i]] = linear_terms.get(var_x_index[i], 0.0) + float(M)
        for r in range(L):
            s_name = var_s_index[(i, r)]
            coeff = (2.0**r)
            linear_terms[s_name] = linear_terms.get(s_name, 0.0) + coeff
        bqm.offset += R*(constant_part**2)

        for v1, c1 in linear_terms.items():
            diag_contrib = R*(2.0*constant_part*c1 + c1**2)
            bqm.set_linear(v1, bqm.get_linear(v1) + diag_contrib)
        var_list = sorted(linear_terms.keys())
        for idx_a in range(len(var_list)):
            vA = var_list[idx_a]
            cA = linear_terms[vA]
            for idx_b in range(idx_a + 1, len(var_list)):
                vB = var_list[idx_b]
                cB = linear_terms[vB]
                coeff_AB = R*(2.0*cA*cB)
                bqm.add_quadratic(vA, vB, coeff_AB)

    return bqm, var_x_index, var_s_index


def is_k_plex(G, subset, k):
    
    subset_set = set(subset)
    for v in subset_set:
        non_neighbors = 0
        for w in subset_set:
            if w == v:
                continue
            if not G.has_edge(v, w):
                non_neighbors += 1
                if non_neighbors > (k - 1):
                    return False
    return True


def show_available_solvers(token):
    
    import dwave.cloud
    print("Listing D-Wave solvers (requires valid token):\n")
    with dwave.cloud.Client.from_config(token=token) as client:
        solvers = client.get_solvers()
        for s in solvers:
            solver_type = "QPU" if s.qpu else "HYBRID" if s.hybrid else "UNKNOWN"
            num_qubits = s.properties.get('num_qubits', 'N/A')
            couplers = s.properties.get('couplers', 'N/A')
            num_couplers = len(couplers) if hasattr(couplers, '__len__') else couplers
            print(f"Solver ID: {s.id}, Type: {solver_type}, Qubits: {num_qubits}, Couplers: {num_couplers}")

    print("\nClassical solvers (local, no qubit limit):")
    print("- ExactSolver: enumerates all solutions, limited to small BQM.")
    print("- SimulatedAnnealingSampler: Monte Carlo approach, can handle moderate BQM size.\n")


def solve_qubo(
    bqm,
    sampler_type="classical",
    token="yourtoken",
    sampler_params=None,
    use_embedding=False
):
    
    import dwave.cloud
    start_time = time.time()
    detailed_timing = {
        "solver_type": sampler_type,
        "embed_used": use_embedding,
        "qpu_access_time": 0,
        "qpu_anneal_time_per_sample": 0,
        "total_real_time": 0
    }

    if sampler_type == "classical":
        sampler = SimulatedAnnealingSampler()
        if sampler_params and "num_reads" in sampler_params:
            num_reads = sampler_params["num_reads"]
            num_sweeps = sampler_params["num_sweeps"]
            sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
        else:
            sampleset = sampler.sample(bqm)

        end_time = time.time()
        detailed_timing["total_real_time"] = end_time - start_time
        return sampleset, detailed_timing

    elif sampler_type == "quantum":
        dwave.cloud.Client.from_config(token=token)
        base_sampler = DWaveSampler()
        if use_embedding:
            source_edges = list(bqm.quadratic.keys())
            adj = base_sampler.adjacency
            target_edges = []
            for u, neighbors in adj.items():
                for v in neighbors:
                    if u < v:
                        target_edges.append((u, v))
            embedding = find_embedding(source_edges, target_edges)
            used_physical_qubits = set()
            for chain in embedding.values():
                used_physical_qubits.update(chain)
            print("Computed embedding:", embedding)
            print("Used physical qubits:", len(used_physical_qubits))

            sampler = EmbeddingComposite(base_sampler)
        else:
            sampler = base_sampler
        if sampler_params:
            sampleset = sampler.sample(bqm, **sampler_params)
        else:
            sampleset = sampler.sample(bqm)

        end_time = time.time()
        detailed_timing["total_real_time"] = end_time - start_time
        if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
            qpu_timing = sampleset.info['timing']
            if 'qpu_access_time' in qpu_timing:
                detailed_timing["qpu_access_time"] = qpu_timing['qpu_access_time']
            if 'qpu_anneal_time_per_sample' in qpu_timing:
                detailed_timing["qpu_anneal_time_per_sample"] = qpu_timing['qpu_anneal_time_per_sample']

        return sampleset, detailed_timing

    elif sampler_type == "hybrid":
        dwave.cloud.Client.from_config(token=token)
        sampler = LeapHybridSampler()
        if sampler_params and 'time_limit' in sampler_params:
            print('time_limit:', sampler_params['time_limit'])
            sampleset = sampler.sample(bqm, time_limit=sampler_params['time_limit'])
        else:
            sampleset = sampler.sample(bqm)

        end_time = time.time()
        detailed_timing["total_real_time"] = end_time - start_time
        return sampleset, detailed_timing

    else:
        raise ValueError("sampler_type must be one of ['classical', 'quantum', 'hybrid'].")

# # --------------------------------------------------
# #original main
def main():
    
    # --------------------------------------------------
    # User Settings
    # --------------------------------------------------
    n = 30   # Must match the file n_8_m_*.txt in the folder
    m = 300  # for example n_8_m_10.txt
    k = 3
    R = 2  # must be > 1
    # token = "yourtoken"
    token = "yourtoken"

    # Solver selection: "classical", "quantum", or "hybrid"
    # solver_type = "classical"
    # solver_type = "hybrid"
    solver_type = "quantum"
    # If solver_type="quantum" and you want an automatic or manual embedding
    use_embedding = True

    # Sampler params example:

    # For "quantum": {'num_reads':100, 'annealing_time':50, 'chain_strength':2}
    # sampler_params = {'num_reads':5, 'annealing_time':20}
    sampler_params = {'num_reads': 1, 'annealing_time': 1}

    # For "hybrid":  {'time_limit':5}
    #'time_limit' controls the max duration of total time of each run, in second
    # sampler_params = {'num_reads': 1, 'time_limit':30}

    # For "classical": {'num_reads':10000} (SimulatedAnnealingSampler)
    #num_sweeps constrol the duration of annealing
    # sampler_params = {'num_reads': 100000, 'num_sweeps': 2}

    # --------------------------------------------------
    # Load Graph
    # --------------------------------------------------
    G = load_graph(n, m)
    actual_n = G.number_of_nodes()
    actual_m = G.number_of_edges()
    print(f"Graph loaded. Actual vertices = {actual_n}, edges = {actual_m}")

    # --------------------------------------------------
    # Build QUBO
    # --------------------------------------------------
    bqm, x_index, s_index = build_qubo_for_mkp(G, k, R)

    # Print total # of binary variables
    total_binaries = len(bqm.variables)
    print(f"QUBO constructed. # of binary variables = {total_binaries}. "
          "These will map to qubits on quantum hardware (if chosen).")


    # --------------------------------------------------
    # Optional: Show available D-Wave solvers
    # --------------------------------------------------
    # show_available_solvers(token)

    # --------------------------------------------------
    # Solve QUBO
    # --------------------------------------------------
    sampleset, timing_info = solve_qubo(
        bqm=bqm,
        sampler_type=solver_type,
        token=token,
        sampler_params=sampler_params,
        use_embedding=use_embedding
    )

    # Print solver timing info
    print("Solver timing breakdown:", timing_info)

    # --------------------------------------------------
    # Extract best solution
    # --------------------------------------------------
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    print(f"Best energy found: {best_energy}")

    # Convert solution back to bits for x_i
    chosen_vertices = []
    for i in G.nodes():
        if best_sample[x_index[i]] == 1:
            chosen_vertices.append(i)

    # --------------------------------------------------
    # Check if chosen_vertices is a valid k-plex
    # --------------------------------------------------
    valid_kplex = is_k_plex(G, chosen_vertices, k)
    if valid_kplex:
        print(f"Solution is a VALID k-plex, size = {len(chosen_vertices)}")
    else:
        print(f"Solution is NOT a valid k-plex, size = {len(chosen_vertices)}")

    print("Chosen vertices:", chosen_vertices)
# # --------------------------------------------------


if __name__ == "__main__":
    main()
