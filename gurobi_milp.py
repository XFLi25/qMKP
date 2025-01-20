

import os
import math
import time
import networkx as nx

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise ImportError("You need to have gurobipy (Gurobi) installed.")

def load_graph(n, m, folder_path="/Users/xiaofanli/Desktop/Code/qMKP/qaMKP"):
    
    prefix = f"n_{n}_m_{m}"
    chosen_file = None
    for fname in os.listdir(folder_path):
        if fname.startswith(prefix) and fname.endswith(".txt"):
            chosen_file = fname
            break
    if not chosen_file:
        raise FileNotFoundError(f"No file starting with 'n_{n}_m_{m}' in {folder_path}")

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
        raise ValueError("k cannot exceed the number of vertices.")
    if R <= 1:
        raise ValueError("R must be > 1.")

    M = n - k + 1
    L = int(math.ceil(math.log2(n))) if n>1 else 1
    alpha = 1.0
    qubo_dict = {}
    offset = 0.0
    x_vars = [f"x_{i}" for i in sorted(G.nodes())]
    s_vars = []
    for i in sorted(G.nodes()):
        for r in range(L):
            s_vars.append(f"s_{i}_{r}")
    def add_to_qubo(u, v, val):
        
        if u > v:
            u, v = v, u
        qubo_dict[(u, v)] = qubo_dict.get((u, v), 0.0) + val
    def add_linear(u, val):
        add_to_qubo(u, u, val)
    neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}
    for i in G.nodes():
        var_x = f"x_{i}"
        add_linear(var_x, -alpha)
    for i in G.nodes():
        const_part = -(k-1) - M
        linear_terms = {}  # var -> coefficient
        non_nbr_list = []
        for nd in G.nodes():
            if nd != i and (nd not in neighbors[i]):
                non_nbr_list.append(nd)

        for nd in non_nbr_list:
            var_j = f"x_{nd}"
            linear_terms[var_j] = linear_terms.get(var_j, 0.0) + 1.0
        var_xi = f"x_{i}"
        linear_terms[var_xi] = linear_terms.get(var_xi, 0.0) + float(M)
        for r in range(L):
            var_sr = f"s_{i}_{r}"
            coeff = (2.0**r)
            linear_terms[var_sr] = linear_terms.get(var_sr, 0.0) + coeff
        offset += R * (const_part**2)
        for var1, c1 in linear_terms.items():
            diag_val = R * (2.0*const_part*c1 + c1**2)
            add_linear(var1, diag_val)
        varkeys = sorted(linear_terms.keys())
        for idxA in range(len(varkeys)):
            vA = varkeys[idxA]
            cA = linear_terms[vA]
            for idxB in range(idxA+1, len(varkeys)):
                vB = varkeys[idxB]
                cB = linear_terms[vB]
                cross_val = R*(2.0 * cA * cB)
                add_to_qubo(vA, vB, cross_val)

    return qubo_dict, x_vars, s_vars, offset

def is_k_plex(G, subset, k):
    
    sub_set = set(subset)
    for v in sub_set:
        nonnbr_count = 0
        for w in sub_set:
            if w == v:
                continue
            if not G.has_edge(v, w):
                nonnbr_count += 1
                if nonnbr_count > (k-1):
                    return False
    return True

def build_milp_from_qubo(qubo_dict, offset, x_vars, s_vars):
    
    model = gp.Model("mkp_milp")
    model.setParam("OutputFlag", 0)  # turn off default logging if desired
    x_milp_vars = {}
    all_qubo_vars = set(x_vars).union(s_vars)

    for qv in sorted(all_qubo_vars):
        x_milp_vars[qv] = model.addVar(vtype=GRB.BINARY, name=qv)
    y_milp_vars = {}
    linear_terms = []  # to accumulate (coeff, var) for objective
    const_in_obj = offset

    for (u,v), coeff in qubo_dict.items():
        if u == v:
            linear_terms.append((coeff, x_milp_vars[u]))
        else:
            y_name = f"y_{u}__{v}"
            y_var = model.addVar(vtype=GRB.BINARY, name=y_name)
            y_milp_vars[(u, v)] = y_var
            model.addConstr(y_var <= x_milp_vars[u], name=f"c_{y_name}_1")
            model.addConstr(y_var <= x_milp_vars[v], name=f"c_{y_name}_2")
            model.addConstr(y_var >= x_milp_vars[u] + x_milp_vars[v] - 1, name=f"c_{y_name}_3")
            linear_terms.append((coeff, y_var))
    obj_expr = gp.LinExpr()
    obj_expr.addConstant(const_in_obj)
    for (co, var) in linear_terms:
        obj_expr.addTerms(co, var)

    model.setObjective(obj_expr, GRB.MINIMIZE)

    model.update()
    return model, x_milp_vars

def main():
    n = 20
    m = 100
    k = 3
    R = 2  # must be > 1
    time_limit_in_seconds = 0.03
    G = load_graph(n, m)
    actual_n = G.number_of_nodes()
    actual_m = G.number_of_edges()
    print(f"Graph loaded. #vertices = {actual_n}, #edges = {actual_m}")
    qubo_dict, x_vars, s_vars, offset = build_qubo_for_mkp(G, k, R)
    qubo_num_binaries = len(x_vars) + len(s_vars)
    print(f"QUBO built. #QUBO binary variables = {qubo_num_binaries}")
    milp_model, milp_vars = build_milp_from_qubo(qubo_dict, offset, x_vars, s_vars)
    milp_num_binaries = milp_model.NumVars
    print(f"MILP built. #MILP binary variables = {milp_num_binaries}")
    if time_limit_in_seconds > 0:
        milp_model.setParam("TimeLimit", time_limit_in_seconds)
    start_time = time.time()
    milp_model.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    if milp_model.SolCount > 0:
        best_obj_value = milp_model.objVal
        print(f"Solution found with objective = {best_obj_value:.4f}")
        chosen_vertices = []
        for xv in x_vars:
            if milp_model.getVarByName(xv).X > 0.5:
                i_str = xv.split("_")[1]
                i = int(i_str)
                chosen_vertices.append(i)
        valid = is_k_plex(G, chosen_vertices, k)
        sub_size = len(chosen_vertices)
        print(f"Chosen subset = {chosen_vertices}")
        if valid:
            print(f"The subgraph is a VALID k-plex, size = {sub_size}")
        else:
            print(f"The subgraph is NOT a valid k-plex, size = {sub_size}")
    else:
        print("No feasible solution found by Gurobi.")
        best_obj_value = None
    print(f"Runtime = {runtime:.3f} seconds")
    print("(Hint) Adjust TimeLimit or other parameters to get different solutions.\n")



if __name__ == "__main__":
    main()
