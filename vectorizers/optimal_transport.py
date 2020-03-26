import numpy as np
import numba
from warnings import warn


EPSILON = 2.2204460492503131e-15
_EPSILON = 1e-8
INFINITY = np.finfo(np.float32).max
MAX = np.finfo(np.float32).max
OPTIMAL = 0
MAX_ITER_REACHED = -1
UNBOUNDED = -2
INFEASIBLE = -3

# locals: c, min, e, cnt, a
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
    c = 0
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


def start(
    _pi,
    _flow,
    _sum_supply,
    _node_num,
    _search_arc_num,
    _all_arc_num,
    _stype,
    delta,
    max_iter,
):
    pivot(*this)
    prevCost = -1.0
    retVal = OPTIMAL

    # Perform heuristic initial pivots
    if not initialPivots():
        return UNBOUNDED

    iter_number = 0
    # pivot.setDantzig(true);
    # Execute the Network Simplex algorithm
    while pivot.findEnteringArc():
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

        findJoinNode()
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
