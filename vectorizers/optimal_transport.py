import numpy as np
import numba
from warnings import warn
from collections import namedtuple


EPSILON = 2.2204460492503131e-15
_EPSILON = 1e-8
INFINITY = np.finfo(np.float32).max
INF = np.iinfo(np.int64).max
MAX = np.finfo(np.float32).max
OPTIMAL = 0
MAX_ITER_REACHED = -1
UNBOUNDED = -2
INFEASIBLE = -3

STATE_UPPER = 1
STATE_TREE = 0
STATE_LOWER = -1

SpanningTree = namedtuple("SpanningTree", [])
PivotBlock = namedtuple("PivotBlock", [])

# locals: c, min, e, cnt, a
# modifies _in_arc, _next_arc,
def findEnteringArc(
    _next_arc,
    _search_arc_num,
    _block_size,
    _state,
    _cost,
    _pi,
    _source,
    _target,
    _in_arc,
):
    min = 0
    cnt = _block_size
    for e in range(_next_arc, _search_arc_num):  # (e = _next_arc; e !=
        # _search_arc_num; ++e) {
        c = _state[e] * (_cost[e] + _pi[_source[e]] - _pi[_target[e]])
        if c < min:
            min = c
            _in_arc = e

        cnt -= 1
        if cnt == 0:
            if np.fabs(_pi[_source[_in_arc]]) > np.fabs(_pi[_target[_in_arc]]):
                a = np.fabs(_pi[_source[_in_arc]])
            else:
                a = np.fabs(_pi[_target[_in_arc]])

            if a <= np.fabs(_cost[_in_arc]):
                a = np.fabs(_cost[_in_arc])

            if min < -(EPSILON * a):
                _next_arc = e
                return True
            else:
                cnt = _block_size
            # a=fabs(_pi[_source[_in_arc]])>fabs(_pi[_target[_in_arc]]) ? fabs(_pi[_source[_in_arc]]):fabs(_pi[_target[_in_arc]]);
            # a=a>fabs(_cost[_in_arc])?a:fabs(_cost[_in_arc]);
            # if (min <  -EPSILON*a) goto search_end;
            # cnt = _block_size;

    for e in range(_next_arc):  # (e = 0; e != _next_arc; ++e) {
        c = _state[e] * (_cost[e] + _pi[_source[e]] - _pi[_target[e]])
        if c < min:
            min = c
            _in_arc = e

        cnt -= 1
        if cnt == 0:
            if np.fabs(_pi[_source[_in_arc]]) > np.fabs(_pi[_target[_in_arc]]):
                a = np.fabs(_pi[_source[_in_arc]])
            else:
                a = np.fabs(_pi[_target[_in_arc]])

            if a <= np.fabs(_cost[_in_arc]):
                a = np.fabs(_cost[_in_arc])

            if min < -(EPSILON * a):
                _next_arc = e
                return True
            else:
                cnt = _block_size
            # a=fabs(_pi[_source[_in_arc]])>fabs(_pi[_target[_in_arc]]) ? fabs(_pi[_source[_in_arc]]):fabs(_pi[_target[_in_arc]]);
            # a=a>fabs(_cost[_in_arc])?a:fabs(_cost[_in_arc]);
            # if (min <  -EPSILON*a) goto search_end;
            # cnt = _block_size;

    if np.fabs(_pi[_source[_in_arc]]) > np.fabs(_pi[_target[_in_arc]]):
        a = np.fabs(_pi[_source[_in_arc]])
    else:
        a = np.fabs(_pi[_target[_in_arc]])

    if a <= np.fabs(_cost[_in_arc]):
        a = np.fabs(_cost[_in_arc])

    if min >= -(EPSILON * a):
        return False
    # a=fabs(_pi[_source[_in_arc]])>fabs(_pi[_target[_in_arc]]) ? fabs(_pi[_source[_in_arc]]):fabs(_pi[_target[_in_arc]]);
    # a=a>fabs(_cost[_in_arc])?a:fabs(_cost[_in_arc]);
    # if (min >=  -EPSILON*a) return false;

    # _next_arc = e # This is not required as we don't use the goto
    return True


# Find the join node
# Operates with graph (_source, _target) and MST (_succ_num, _parent, in_arc) data
# locals: u, v
# modifies: join
def findJoinNode(_source, _target, _succ_num, _parent, in_arc):
    u = _source[in_arc]
    v = _target[in_arc]
    while u != v:
        if _succ_num[u] < _succ_num[v]:
            u = _parent[u]
        else:
            v = _parent[v]

    join = u

    return join


# Find the leaving arc of the cycle and returns true if the
# leaving arc is not the same as the entering arc
# locals: first, second, result, d, e
# modifies: u_in, v_in, u_out, delta
def findLeavingArc(
    join, in_arc, _state, _source, _target, _flow, _pred, _parent, _forward
):
    # Initialize first and second nodes according to the direction
    # of the cycle
    if _state[in_arc] == STATE_LOWER:
        first = _source[in_arc]
        second = _target[in_arc]
    else:
        first = _target[in_arc]
        second = _source[in_arc]

    delta = INF
    result = 0

    # Search the cycle along the path form the first node to the root
    # for (int u = first; u != join; u = _parent[u]) {
    u = first
    while u != join:
        e = _pred[u]
        if _forward[u]:
            d = _flow[e]
        else:
            d = INF

        if d < delta:
            delta = d
            u_out = u
            result = 1

        u = _parent[u]

    # Search the cycle along the path form the second node to the root
    # for (int u = second; u != join; u = _parent[u]) {
    u = second
    while u != join:
        e = _pred[u]
        if _forward[u]:
            d = INF
        else:
            d = _flow[e]

        if d <= delta:
            delta = d
            u_out = u
            result = 2

        u = _parent[u]

    if result == 1:
        u_in = first
        v_in = second
    else:
        u_in = second
        v_in = first

    return result != 0


# Change _flow and _state vectors
# locals: val, u
# modifies: _state, _flow
def changeFlow(
    change,
    join,
    delta,
    u_out,
    _state,
    _flow,
    _source,
    _target,
    _forward,
    _pred,
    _parent,
    in_arc,
):
    # Augment along the cycle
    if delta > 0:
        val = _state[in_arc] * delta
        _flow[in_arc] += val
        # for (int u = _source[in_arc]; u != join; u = _parent[u]) {
        u = _source[in_arc]
        while u != join:
            if _forward[u]:
                _flow[_pred[u]] -= val
            else:
                _flow[_pred[u]] += val

            u = _parent[u]

        # for (int u = _target[in_arc]; u != join; u = _parent[u]) {
        u = _source[in_arc]
        while u != join:
            if _forward[u]:
                _flow[_pred[u]] += val
            else:
                _flow[_pred[u]] -= val

            u = _parent[u]

    # Update the state of the entering and leaving arcs
    if change:
        _state[in_arc] = STATE_TREE
        if _flow[_pred[u_out]] == 0:
            _state[_pred[u_out]] = STATE_LOWER
        else:
            _state[_pred[u_out]] = STATE_UPPER
    else:
        _state[in_arc] = -_state[in_arc]


# Update the tree structure
# locals: u, w, old_rev_thread, old_succ_num, old_last_succ, tmp_sc, tmp_ls
# more locals: up_limit_in, up_limit_out
# modifies: v_out, _thread, _dirty_revs, _rev_thread, _parent, _last_succ,
# modifies: _pred, _forward, _succ_num
def updateTreeStructure(
    v_in,
    u_in,
    u_out,
    join,
    in_arc,
    _thread,
    _dirty_revs,
    _rev_thread,
    _parent,
    _last_succ,
    _pred,
    _forward,
    _succ_num,
    _source,
):
    old_rev_thread = _rev_thread[u_out]
    old_succ_num = _succ_num[u_out]
    old_last_succ = _last_succ[u_out]
    v_out = _parent[u_out]

    u = _last_succ[u_in]  # the last successor of u_in
    right = _thread[u]  # the node after it

    # Handle the case when old_rev_thread equals to v_in
    # (it also means that join and v_out coincide)
    if old_rev_thread == v_in:
        last = _thread[_last_succ[u_out]]
    else:
        last = _thread[v_in]

    # Update _thread and _parent along the stem nodes (i.e. the nodes
    # between u_in and u_out, whose parent have to be changed)
    _thread[v_in] = stem = u_in
    _dirty_revs = []
    _dirty_revs.append(v_in)
    par_stem = v_in
    while stem != u_out:
        # Insert the next stem node into the thread list
        new_stem = _parent[stem]
        _thread[u] = new_stem
        _dirty_revs.append(u)

        # Remove the subtree of stem from the thread list
        w = _rev_thread[stem]
        _thread[w] = right
        _rev_thread[right] = w

        # Change the parent node and shift stem nodes
        _parent[stem] = par_stem
        par_stem = stem
        stem = new_stem

        # Update u and right
        if _last_succ[stem] == _last_succ[par_stem]:
            u = _rev_thread[par_stem]
        else:
            u = _last_succ[stem]

        right = _thread[u]

    _parent[u_out] = par_stem
    _thread[u] = last
    _rev_thread[last] = u
    _last_succ[u_out] = u

    # Remove the subtree of u_out from the thread list except for
    # the case when old_rev_thread equals to v_in
    # (it also means that join and v_out coincide)
    if old_rev_thread != v_in:
        _thread[old_rev_thread] = right
        _rev_thread[right] = old_rev_thread

    # Update _rev_thread using the new _thread values
    # for (int i = 0; i != int(_dirty_revs.size()); ++i) {
    for i in range(len(_dirty_revs)):
        u = _dirty_revs[i]
        _rev_thread[_thread[u]] = u

    # Update _pred, _forward, _last_succ and _succ_num for the
    # stem nodes from u_out to u_in
    tmp_sc = 0
    tmp_ls = _last_succ[u_out]
    u = u_out
    while u != u_in:
        w = _parent[u]
        _pred[u] = _pred[w]
        _forward[u] = not _forward[w]
        tmp_sc += _succ_num[u] - _succ_num[w]
        _succ_num[u] = tmp_sc
        _last_succ[w] = tmp_ls
        u = w

    _pred[u_in] = in_arc
    _forward[u_in] = u_in == _source[in_arc]
    _succ_num[u_in] = old_succ_num

    # Set limits for updating _last_succ form v_in and v_out
    # towards the root
    up_limit_in = -1
    up_limit_out = -1
    if _last_succ[join] == v_in:
        up_limit_out = join
    else:
        up_limit_in = join

    # Update _last_succ from v_in towards the root
    # for (u = v_in; u != up_limit_in && _last_succ[u] == v_in;
    #      u = _parent[u]) {
    u = v_in
    while u != up_limit_in and _last_succ[u] == v_in:
        _last_succ[u] = _last_succ[u_out]
        u = _parent[u]

    # Update _last_succ from v_out towards the root
    if join != old_rev_thread and v_in != old_rev_thread:
        # for (u = v_out; u != up_limit_out && _last_succ[u] == old_last_succ;
        #      u = _parent[u]) {
        u = v_out
        while u != up_limit_out and _last_succ[u] == old_last_succ:
            _last_succ[u] = old_rev_thread
            u = _parent[u]

    else:
        # for (u = v_out; u != up_limit_out && _last_succ[u] == old_last_succ;
        #      u = _parent[u]) {
        u = v_out
        while u != up_limit_out and _last_succ[u] == old_last_succ:
            _last_succ[u] = _last_succ[u_out]
            u = _parent[u]

    # Update _succ_num from v_in to join
    # for (u = v_in; u != join; u = _parent[u]) {
    u = v_in
    while u != join:
        _succ_num[u] += old_succ_num
        u = _parent[u]

    # Update _succ_num from v_out to join
    # for (u = v_out; u != join; u = _parent[u]) {
    u = v_out
    while u != join:
        _succ_num[u] -= old_succ_num
        u = _parent[u]


# Update potentials
# locals: sigma, end
# modifies: _pi
def updatePotential(u_in, v_in, _forward, _pi, _thread, _last_succ, _cost, _pred):
    if _forward[u_in]:
        sigma = _pi[v_in] - _pi[u_in] - _cost[_pred[u_in]]
    else:
        sigma = _pi[v_in] - _pi[u_in] + _cost[_pred[u_in]]

    # Update potentials in the subtree, which has been moved
    # for (int u = u_in; u != end; u = _thread[u]) {
    end = _thread[_last_succ[u_in]]
    u = u_in
    while u != end:
        _pi[u] += sigma
        u = _thread[u]


# Heuristic initial pivots
# locals: curr, total, supply_nodes, demand_nodes, u
# modifies:
def initialPivots():
    curr = 0
    total = 0
    supply_nodes = []
    demand_nodes = []

    # Node u; _graph.first(u);
    # for (; u != INVALIDNODE; _graph.next(u)) {
    for u in range(n_nodes, -1, -1):
        curr = _supply[_node_id(u)]
        if curr > 0:
            total += curr
            supply_nodes.append(u)
        elif curr < 0:
            demand_nodes.append(u)

    if _sum_supply > 0:
        total -= _sum_supply

    if total <= 0:
        return True

    arc_vector = []
    if _sum_supply >= 0:
        if len(supply_nodes) == 1 and len(demand_nodes) == 1:
            # Perform a reverse graph search from the sink to the source
            # typename GR::template NodeMap<bool> reached(_graph, false);
            reached = np.zeros(_node_num, dtype=np.bool)
            s = supply_nodes[0]
            t = demand_nodes[0]
            stack = []
            reached[t] = True
            stack.append(t)
            while len(stack) > 0:
                u = stack[-1]
                v = stack[-1]
                stack.pop(-1)
                if v == s:
                    break

                # Arc a; _graph.firstIn(a, v);
                # for (; a != INVALID; _graph.nextIn(a)) {
                first_arc = n_arcs + v - n_nodes if v >= _n1 else -1
                for a in range(first_arc, -1, -_n2):
                    u = _graph.source(a)
                    if reached[u]:
                        continue

                    j = getArcID(a)
                    if INF >= total:
                        arc_vector.append(j)
                        reached[u] = True
                        stack.append(u)

        else:
            # Find the min. cost incomming arc for each demand node
            # for (int i = 0; i != int(demand_nodes.size()); ++i) {
            for i in range(len(demand_nodes)):
                v = demand_nodes[i]
                c = MAX
                min_cost = MAX
                min_arc = INVALID
                # Arc a; _graph.firstIn(a, v);
                # for (; a != INVALID; _graph.nextIn(a)) {
                first_arc = n_arcs + v - n_nodes if v >= _n1 else -1
                for a in range(first_arc, -1, -_n2):
                    c = _cost[getArcID(a)]
                    if c < min_cost:
                        min_cost = c
                        min_arc = a

                if min_arc != INVALID:
                    arc_vector.append(getArcID(min_arc))

    else:
        # Find the min. cost outgoing arc for each supply node
        # for (int i = 0; i != int(supply_nodes.size()); ++i) {
        for i in range(len(supply_nodes)):
            u = supply_nodes[i]
            c = MAX
            min_cost = MAX
            min_arc = INVALID
            # Arc a; _graph.firstOut(a, u);
            # for (; a != INVALID; _graph.nextOut(a)) {
            a = (u + 1) * _n2 - 1 if u <= _n1 else -1
            while a % _n2 != 0 and a >= 0:
                c = _cost[getArcID(a)]
                if c < min_cost:
                    min_cost = c
                    min_arc = a

                a -= 1

            if min_arc != INVALID:
                arc_vector.append(getArcID(min_arc))

    # Perform heuristic initial pivots
    # for (int i = 0; i != int(arc_vector.size()); ++i) {
    for i in range(len(arc_vector)):
        in_arc = arc_vector[i]
        # l'erreur est probablement ici...
        if (
            _state[in_arc]
            * (_cost[in_arc] + _pi[_source[in_arc]] - _pi[_target[in_arc]])
            >= 0
        ):
            continue

        findJoinNode()
        change = findLeavingArc()
        if delta >= MAX:
            return False

        changeFlow(change)
        if change:
            updateTreeStructure()
            updatePotential()

    return True


def start(
    _pi,
    _flow,
    _cost,
    _state,
    _sum_supply,
    _node_num,
    _source,
    _target,
    _search_arc_num,
    _all_arc_num,
    _stype,
    _block_size,
    _next_arc,
    _in_arc,
    _succ_num,
    _parent,
    delta,
    max_iter,
):
    pivot_data = pivot(*this)
    prevCost = -1.0
    retVal = OPTIMAL

    # Perform heuristic initial pivots
    if not initialPivots():
        return UNBOUNDED

    iter_number = 0
    # pivot.setDantzig(true);
    # Execute the Network Simplex algorithm
    while findEnteringArc(
        _next_arc,
        _search_arc_num,
        _block_size,
        _state,
        _cost,
        _pi,
        _source,
        _target,
        _in_arc,
    ):
        iter_number += 1
        if max_iter > 0 and iter_number >= max_iter and max_iter > 0:
            warn(
                f"RESULT MIGHT BE INACURATE\nMax number of "
                f"iteration reached, currently {iter_number}. Sometimes iterations"
                " go on in "
                "cycle even though the solution has been reached, to check if it's the case here have a look at the minimal reduced cost. If it is very close to machine precision, you might actually have the correct solution, if not try setting the maximum number of iterations a bit higher\n"
            )
            retVal = MAX_ITER_REACHED
            break

        # #if DEBUG_LVL>0
        #                 if(iter_number>MAX_DEBUG_ITER)
        #                     break;
        #                 if(iter_number%1000==0||iter_number%1000==1){
        #                     double curCost=totalCost();
        #                     double sumFlow=0;
        #                     double a;
        #                     a= (fabs(_pi[_source[in_arc]])>=fabs(_pi[_target[in_arc]])) ? fabs(_pi[_source[in_arc]]) : fabs(_pi[_target[in_arc]]);
        #                     a=a>=fabs(_cost[in_arc])?a:fabs(_cost[in_arc]);
        #                     for (int i=0; i<_flow.size(); i++) {
        #                         sumFlow+=_state[i]*_flow[i];
        #                     }
        #                     std::cout << "Sum of the flow " << std::setprecision(20) << sumFlow << "\n" << iter_number << " iterations, current cost=" << curCost << "\nReduced cost=" << _state[in_arc] * (_cost[in_arc] + _pi[_source[in_arc]] -_pi[_target[in_arc]]) << "\nPrecision = "<< -EPSILON*(a) << "\n";
        #                     std::cout << "Arc in = (" << _node_id(_source[in_arc]) << ", " << _node_id(_target[in_arc]) <<")\n";
        #                     std::cout << "Supplies = (" << _supply[_source[in_arc]] << ", " << _supply[_target[in_arc]] << ")\n";
        #                     std::cout << _cost[in_arc] << "\n";
        #                     std::cout << _pi[_source[in_arc]] << "\n";
        #                     std::cout << _pi[_target[in_arc]] << "\n";
        #                     std::cout << a << "\n";
        #                 }
        # #endif

        join = findJoinNode(_source, _target, _succ_num, _parent, _in_arc)
        change = findLeavingArc()
        if delta >= MAX:
            return UNBOUNDED

        changeFlow(change)

        if change:
            updateTreeStructure()
            updatePotential()

    # #if DEBUG_LVL>0
    #                 else{
    #                     std::cout << "No change\n";
    #                 }
    # #endif
    # #if DEBUG_LVL>1
    #                 std::cout << "Arc in = (" << _source[in_arc] << ", " << _target[in_arc] << ")\n";
    # #endif

    # #if DEBUG_LVL>0
    #                 double curCost=totalCost();
    #                 double sumFlow=0;
    #                 double a;
    #                 a= (fabs(_pi[_source[in_arc]])>=fabs(_pi[_target[in_arc]])) ? fabs(_pi[_source[in_arc]]) : fabs(_pi[_target[in_arc]]);
    #                 a=a>=fabs(_cost[in_arc])?a:fabs(_cost[in_arc]);
    #                 for (int i=0; i<_flow.size(); i++) {
    #                     sumFlow+=_state[i]*_flow[i];
    #                 }
    #
    #                 std::cout << "Sum of the flow " << std::setprecision(20) << sumFlow << "\n" << niter << " iterations, current cost=" << curCost << "\nReduced cost=" << _state[in_arc] * (_cost[in_arc] + _pi[_source[in_arc]] -_pi[_target[in_arc]]) << "\nPrecision = "<< -EPSILON*(a) << "\n";
    #
    #                 std::cout << "Arc in = (" << _node_id(_source[in_arc]) << ", " << _node_id(_target[in_arc]) <<")\n";
    #                 std::cout << "Supplies = (" << _supply[_source[in_arc]] << ", " << _supply[_target[in_arc]] << ")\n";

    # endif

    # #if DEBUG_LVL>1
    #             sumFlow=0;
    #             for (int i=0; i<_flow.size(); i++) {
    #                 sumFlow+=_state[i]*_flow[i];
    #                 if (_state[i]==STATE_TREE) {
    #                     std::cout << "Non zero value at (" << _node_num+1-_source[i] << ", " << _node_num+1-_target[i] << ")\n";
    #                 }
    #             }
    #             std::cout << "Sum of the flow " << sumFlow << "\n"<< niter <<" iterations, current cost=" << totalCost() << "\n";
    # #endif
    # Check feasibility
    if retVal == OPTIMAL:
        for e in range(_search_arc_num, _all_arc_num):
            if _flow[e] != 0:
                if abs(_flow[e]) > EPSILON:
                    return INFEASIBLE
                else:
                    _flow[e] = 0

    # Shift potentials to meet the requirements of the GEQ/LEQ type
    # optimality conditions
    if _sum_supply == 0:
        if _stype == "GEQ":
            max_pot = -INFINITY
            for i in range(_node_num):
                if _pi[i] > max_pot:
                    max_pot = _pi[i]
            if max_pot > 0:
                for i in range(_node_num):
                    _pi[i] -= max_pot

        else:
            min_pot = INFINITY
            for i in range(_node_num):
                if _pi[i] < min_pot:
                    min_pot = _pi[i]
            if min_pot < 0:
                for i in range(_node_num):
                    _pi[i] -= min_pot

    return retVal
