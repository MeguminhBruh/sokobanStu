"""Student entry point. See README.md for the API and scoring rules."""

import time
import heapq
from collections import deque
from sokoban import SokobanState, sokoban_goal_state

def build_neighbors(state): #ính sẵn, cho mỗi ô trên bàn cờ, 4 ô liền kề (lên/phải/xuống/trái) nếu đi được hoặc -1 nếu bị chặn (tường/ra ngoài biên), để lúc search không phải tính lại tọa độ hay tra frozenset mỗi lần.
    w = state.width
    h = state.height
    blocked = {y * w + x for (x, y) in state.obstacles}
    nb = [[-1, -1, -1, -1] for _ in range(w * h)]
    for y in range(h):
        for x in range(w):
            c = y * w + x
            if c in blocked:
                continue
            for d, (dx, dy) in enumerate(DELTAS):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (ny * w + nx) not in blocked:
                    nb[c][d] = ny * w + nx
    return nb

def gen_successors(nb, robots, boxes): #sinh ra tất cả các trạng thái (robots, boxes) kế tiếp hợp lệ từ một trạng thái hiện tại, bằng cách thử cho từng robot đi 1 trong 4 hướng (và đẩy thùng nếu ô kế có thùng, miễn ô sau thùng trống).
    out = []
    for r in range(len(robots)):
        pos = robots[r]
        for d in range(4):
            np_ = nb[pos][d]
            if np_ < 0 or np_ in robots:
                continue
            nbx = boxes
            if (boxes >> np_) & 1:                # ô kế có thùng -> thử đẩy
                t = nb[np_][d]
                if t < 0 or (boxes >> t) & 1 or t in robots:
                    continue
                nbx = boxes ^ (1 << np_) ^ (1 << t)
            new_robots = robots[:r] + (np_,) + robots[r + 1:]
            out.append((new_robots, nbx, (r, d)))
    return out


def bfs_core(nb, robots0, boxes0, storage_mask, deadline):
    """Trả về (goal_key, came_from) nếu tìm thấy, hoặc (None, came_from) nếu hết giờ/không có lời giải."""
    start = (robots0, boxes0)
    frontier = deque([start])
    came_from = {start: None}      # key -> (parent_key, action) hoặc None cho state đầu
    checks = 0
    while frontier:
        checks += 1
        if checks % 1000 == 0 and time.perf_counter() > deadline:
            return None, came_from

        robots, boxes = frontier.popleft()
        if boxes & ~storage_mask == 0:          # mọi thùng nằm trong storage_mask
            return (robots, boxes), came_from

        for nr, nbx, action in gen_successors(nb, robots, boxes):
            key = (nr, nbx)
            if key not in came_from:
                came_from[key] = ((robots, boxes), action)
                frontier.append(key)
    return None, came_from
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
    if timebound is None or timebound <= 0:
        return False

    w, h = initial_state.width, initial_state.height
    nb = build_neighbors(initial_state)

    robots0 = tuple(y * w + x for (x, y) in initial_state.robots)
    boxes0 = 0
    for (x, y) in initial_state.boxes:
        boxes0 |= 1 << (y * w + x)
    storage_mask = 0
    for (x, y) in initial_state.storage:
        storage_mask |= 1 << (y * w + x)

    if boxes0 & ~storage_mask == 0:             # đã là goal ngay từ đầu
        return initial_state

    deadline = time.perf_counter() + timebound
    goal_key, came_from = bfs_core(nb, robots0, boxes0, storage_mask, deadline)
    if goal_key is None:
        return False

    # dựng ngược chuỗi action từ goal_key về start
    actions = []
    key = goal_key
    while came_from[key] is not None:
        parent_key, action = came_from[key]
        actions.append(action)
        key = parent_key
    actions.reverse()

    # replay qua successors() thật để có SokobanState hợp lệ
    current = initial_state
    for r, d in actions:
        wanted = "{} {}".format(r, DIR_NAMES[d])
        current = next(s for s in current.successors() if s.action == wanted)
    return current