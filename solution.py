"""Student entry point. See README.md for the API and scoring rules."""

import time
from collections import deque
from sokoban import SokobanState, sokoban_goal_state

def solve(initial_state, timebound=120):
    if timebound <= 0:
        return False

    start_time = time.perf_counter()  #ghi lại thời điểm bắt đầu chạy

    frontier = deque([initial_state])
    visited = {initial_state.hashable_state()}
    while frontier:
        if time.perf_counter() - start_time >= timebound:  #lấy thời gian hiện tại trừ thời gian bắt đầu
            return False     
         
        state = frontier.popleft()
        if sokoban_goal_state(state):
            return state
        for succ in state.successors():
            key = succ.hashable_state()
            if key not in visited:
                visited.add(key)
                frontier.append(succ)
    return False