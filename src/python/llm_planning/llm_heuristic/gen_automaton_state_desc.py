from collections import defaultdict
from heapq import *
import spot
import numpy as np
from llm_planning.natural_to_temporal_logic.visualization import show_svg


def dijkstra(g, f, t):
    q, seen, mins = [(0, f, ())], set(), {f: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t:
                return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen:
                    continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf"), None


def custom_print(aut):
    bdict = aut.get_dict()
    print("Acceptance:", aut.get_acceptance())
    print("Number of sets:", aut.num_sets())
    print("Number of states: ", aut.num_states())
    print("Initial states: ", aut.get_init_state_number())
    print("Atomic propositions:", end="")
    for ap in aut.ap():
        print(" ", ap, " (=", bdict.varnum(ap), ")", sep="", end="")
    print()
    # Templated methods are not available in Python, so we cannot
    # retrieve/attach arbitrary objects from/to the automaton.  However the
    # Python bindings have get_name() and set_name() to access the
    # "automaton-name" property.
    name = aut.get_name()
    if name:
        print("Name: ", name)
    print("Deterministic:", aut.prop_universal() and aut.is_existential())
    print("Unambiguous:", aut.prop_unambiguous())
    print("State-Based Acc:", aut.prop_state_acc())
    print("Terminal:", aut.prop_terminal())
    print("Weak:", aut.prop_weak())
    print("Inherently Weak:", aut.prop_inherently_weak())
    print("Stutter Invariant:", aut.prop_stutter_invariant())

    for s in range(0, aut.num_states()):
        print("State {}: accepting = {}".format(s, aut.state_is_accepting(s)))
        for t in aut.out(s):
            print("  edge({} -> {})".format(t.src, t.dst))
            # bdd_print_formula() is designed to print on a std::ostream, and
            # is inconvenient to use in Python.  Instead we use
            # bdd_format_formula() as this simply returns a string.
            print("    label =", spot.bdd_format_formula(bdict, t.cond))
            print("    acc sets =", t.acc)


def build_graph_from_spot(aut, num_aps):
    bdict = aut.get_dict()
    start_state = aut.get_init_state_number()
    edges = {}
    accepting_states = []
    for s in range(0, aut.num_states()):
        print("State {}:".format(s))
        if aut.state_is_accepting(s):
            accepting_states.append(s)
        for t in aut.out(s):
            print("  edge({} -> {})".format(t.src, t.dst))
            edge_vals = parse_edge_dfa(spot.bdd_format_formula(bdict, t.cond), num_aps)
            edges[(t.src, t.dst)] = (1, edge_vals)

    g = defaultdict(list)
    for e in edges:
        l = e[0]
        r = e[1]
        c = edges[e][0]
        g[l].append((c, r))

    return start_state, edges, g, accepting_states


def parse_edge_dfa(edge, num_aps):
    edge = edge.strip()
    aps = edge.split("|")[0]
    aps = aps.lstrip(" (").rstrip(") ")
    aps = aps.split("&")
    print(aps)
    aps = [ap.strip() for ap in aps]
    # print(aps)
    edge_vals = []
    for i in range(num_aps):
        if "p" + str(i + 1) in aps:
            edge_vals.append(True)
        elif "!p" + str(i + 1) in aps:
            edge_vals.append(False)
        else:
            edge_vals.append(None)
    return edge_vals


def parse_ap_func(ap_func):
    function = ap_func.strip()
    ret = None
    try:
        if function[:5] == "enter":
            room = function[6:-1].strip()
            ret = "visit {}".format(room)
        elif function[:5] == "reach":
            object = function[6:-1]
            ret = "reach {}".format(object)
    except Exception as e:
        print("An exception has happened with parsing {}: \n {}".format(function, str(e)))
    return ret


def load_ap_desc(edge):
    ap_func = np.load(edge)
    print("AP descriptions:\n")
    ap_desc = []
    for ap in ap_func.keys():
        next_ap_desc = parse_ap_func(str(ap_func[ap]).replace("_", " with ID "))
        ap_desc.append(next_ap_desc)
        print("{} = {}".format(ap, next_ap_desc))
    return ap_desc


def gen_dfa_state_desc(g, edges, start_state, automaton_state, ap_desc):
    path = dijkstra(g, start_state, automaton_state)[1]
    aps = [None for i in range(len(ap_desc))]
    while len(path[1]) == 2:
        v2 = path[0]
        v1 = path[1][0]
        ap_edge = edges[(v1, v2)][1]
        for i in range(len(aps)):
            if aps[i] is None:
                aps[i] = ap_edge[i]
        path = path[1]
    print("AP values for state {} is {}".format(automaton_state, aps))
    desc = ""
    for i in range(len(aps)):
        if aps[i] is True:
            desc = desc + ap_desc[i] + ". "
    print("Description of current automaton state: {}".format(desc))
    return desc


def test():
    task_id = 5
    env = "Benevolence"
    dir = "nl2ltl"
    ap_desc = load_ap_desc("../../../" + dir + "/" + env + "/" + str(task_id) + "/ap_desc.npz")
    spot_aut = spot.automaton("../../../" + dir + "/" + env + "/" + str(task_id) + "/automaton.aut")
    task_desc_file = "../../../" + dir + "/" + env + "/" + str(task_id) + "/NL_instructions_uuid"
    ltl_desc_file = "../../../" + dir + "/" + env + "/" + str(task_id) + "/GPT_LTL_formula"
    with open(task_desc_file) as f:
        task_desc = f.readlines()
    print(task_desc)
    with open(ltl_desc_file) as f:
        ltl_desc = f.readlines()
    print(ltl_desc)
    start_state, edges, g, accepting_states = build_graph_from_spot(spot_aut, num_aps=len(ap_desc))
    print("start_state = {}".format(start_state))
    print("AP values for each edge: {}".format(edges))
    automaton_state = 1
    desc = gen_dfa_state_desc(g, edges, start_state, automaton_state, ap_desc)
    print(desc)
    show_svg(spot_aut.show().data)


if __name__ == "__main__":
    test()
