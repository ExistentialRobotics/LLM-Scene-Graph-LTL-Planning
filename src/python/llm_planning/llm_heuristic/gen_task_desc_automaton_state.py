from collections import defaultdict
from heapq import heappush
from heapq import heappop
import spot
from llm_planning.llm_heuristic.gen_automaton_state_desc import load_ap_desc
from llm_planning.natural_to_temporal_logic.visualization import show_svg


def dijkstra(g, f, t):
    q, seen, mins = [(0, f, ())], set(), {f: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t:
                return cost, path

            for c, v2 in g.get(v1, ()):
                if v2 in seen:
                    continue
                prev_cost = mins.get(v2, None)
                next_cost = cost + c
                if prev_cost is None or next_cost < prev_cost:
                    mins[v2] = next_cost
                    heappush(q, (next_cost, v2, path))

    return float("inf"), None


def custom_print(aut):
    bdict = aut.get_dict()
    print("Acceptance:", aut.get_acceptance())
    print("Number of sets:", aut.num_sets())
    print("Number of states:", aut.num_states())
    print("Initial states:", aut.get_init_state_number())
    print("Atomic propositions:", end="")
    for ap in aut.ap():
        print(" ", ap, " (=", bdict.varnum(ap), ")", sep="", end="")
    print()
    # Templated methods are not available in Python, so we cannot
    # retrieve/attach arbitrary objects from/to the automaton. However, the
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
            # is inconvenient to use in Python. Instead, we use
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
            edge_ap_vals = parse_edge_dfa(spot.bdd_format_formula(bdict, t.cond), num_aps)
            edges[(t.src, t.dst)] = (1, edge_ap_vals)

    graph = defaultdict(list)
    for e in edges:
        v1 = e[0]
        v2 = e[1]
        cost = edges[e][0]
        graph[v1].append((cost, v2))

    return start_state, edges, graph, accepting_states


def parse_edge_dfa(edge, num_aps):
    edge = edge.strip()
    aps = edge.split("|")[0]  # multiple labels exist, take the first one only.
    aps = aps.lstrip(" (").rstrip(") ")
    aps = aps.split("&")
    aps = [ap.strip() for ap in aps]
    edge_vals = []
    for i in range(num_aps):
        if "!p" + str(i + 1) in aps:  # ap index starts from 1
            edge_vals.append(False)
        elif "p" + str(i + 1) in aps:  # we care about positive ap only.
            edge_vals.append(True)
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
            obj = function[6:-1]
            ret = "reach {}".format(obj)
    except Exception as e:
        print("An exception has happened with parsing {}: \n {}".format(function, str(e)))
    return ret


def gen_task_state_desc(graph, edges, begin, destination, ap_desc):
    path = dijkstra(graph, begin, destination)[1]
    if path is None:
        return ""
    aps = [None for _ in range(len(ap_desc))]
    while len(path[1]) == 2:
        v2 = path[0]
        v1 = path[1][0]
        ap_edge = edges[(v1, v2)][1]
        for i in range(len(aps)):
            if aps[i] is None:
                aps[i] = ap_edge[i]
        path = path[1]
    print("The remaining AP values for state {} is {}".format(begin, aps))
    desc = ", and ".join([ap_desc[i] for i in range(len(aps)) if aps[i] is True])
    print("Description of the remaining tasks: {}".format(desc))
    return desc


def test():
    ap_desc = load_ap_desc("../natural_to_temporal_logic/ap_desc.npz")
    spot_aut = spot.automaton("../natural_to_temporal_logic/automaton.aut")
    start_state, edges, graph, accepting_states = build_graph_from_spot(spot_aut, num_aps=len(ap_desc))
    print("start_state = {}".format(start_state))
    print("AP values for each edge: {}".format(edges))
    automaton_state = 1
    desc = gen_task_state_desc(graph, edges, automaton_state, accepting_states[0], ap_desc)
    print(desc)
    show_svg(spot_aut.show().data)


if __name__ == "__main__":
    test()
