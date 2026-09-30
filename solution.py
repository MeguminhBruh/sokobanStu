"""Student entry point. See README.md for the API and scoring rules."""


from collections import deque
from sokoban import SokobanState, sokoban_goal_state

def solve(initial_state, timebound=120):
    if timebound <= 0:
        return False
    frontier = deque([initial_state])
    visited = {initial_state.hashable_state()}
    while frontier:
        state = frontier.popleft()
        if sokoban_goal_state(state):
            return state
        for succ in state.successors():
            key = succ.hashable_state()
            if key not in visited:
                visited.add(key)
                frontier.append(succ)
    return False