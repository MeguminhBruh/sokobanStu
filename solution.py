"""Student entry point. See README.md for the API and scoring rules."""

import time
import heapq
from collections import deque
from sokoban import SokobanState, sokoban_goal_state

def heuristic(state):
    total = 0  # tổng h(n) của cả state

    for box in state.boxes:  # duyệt từng thùng
        min_distance = float("inf")  # khoảng cách nhỏ nhất từ box này tới 1 storage

        for storage in state.storage:  # thử tất cả ô đích
            distance = abs(box[0] - storage[0]) + abs(box[1] - storage[1]) # tính khoảng cách Manhattan giữa box và storage

            min_distance = min(min_distance, distance) # giữ lại storage gần box này nhất

        total += min_distance # cộng khoảng cách gần nhất của box này vào tổng h(n)

    return total  # trả về heuristic h(n)

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