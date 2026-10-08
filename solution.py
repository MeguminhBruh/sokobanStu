"""Student entry point. See README.md for the API and scoring rules."""

import time
import heapq
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

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

def heuristic_mask(boxes, storage_mask, w):  #thêm heuristic_mask để phù hợp với code search hiện tại
    total = 0  # tổng h(n)

    # lấy danh sách vị trí các storage từ storage_mask
    storages = []

    temp_storage = storage_mask

    while temp_storage:
        bit = temp_storage & -temp_storage
        pos = bit.bit_length() - 1

        storages.append(pos)

        temp_storage ^= bit

    # duyệt từng box
    temp_boxes = boxes

    while temp_boxes:
        bit = temp_boxes & -temp_boxes
        box_pos = bit.bit_length() - 1

        # đổi vị trí box sang tọa độ (x, y)
        box_x = box_pos % w
        box_y = box_pos // w

        min_distance = float("inf")

        # tìm storage gần box này nhất
        for storage_pos in storages:

            storage_x = storage_pos % w
            storage_y = storage_pos // w

            distance = abs(box_x - storage_x) + abs(box_y - storage_y)
            # khoảng cách Manhattan

            min_distance = min(min_distance, distance)

        total += min_distance
        # cộng khoảng cách gần nhất của box này vào h(n)

        temp_boxes ^= bit
        # bỏ box vừa xét để chuyển sang box tiếp theo

    return total

def is_deadlock(nb, boxes, storage_mask):
    """
    Deadlock góc cơ bản:
    nếu một box không nằm trên storage và bị chặn ở 2 hướng vuông góc thì state đó không thể giải tiếp.
    """
    temp_boxes = boxes

    while temp_boxes:
        bit = temp_boxes & -temp_boxes
        box_pos = bit.bit_length() - 1

        # box đang ở storage thì không coi là deadlock
        if bit & storage_mask:
            temp_boxes ^= bit
            continue

        # 4 hướng quanh box
        up = nb[box_pos][0]
        right = nb[box_pos][1]
        down = nb[box_pos][2]
        left = nb[box_pos][3]

        # nb = -1 nghĩa là bị tường hoặc ra ngoài biên
        up_blocked = up == -1
        right_blocked = right == -1
        down_blocked = down == -1
        left_blocked = left == -1

        # box mắc ở một trong 4 góc
        if up_blocked and left_blocked:
            return True

        if up_blocked and right_blocked:
            return True

        if down_blocked and left_blocked:
            return True

        if down_blocked and right_blocked:
            return True

        # chuyển sang box tiếp theo
        temp_boxes ^= bit

    return False

def bfs_core(nb, robots0, boxes0, storage_mask, w, deadline):
    """Trả về (goal_key, came_from) nếu tìm thấy, hoặc (None, came_from) nếu hết giờ/không có lời giải."""
    start = (robots0, boxes0)
    counter = 0
    h0 = heuristic_mask(boxes0, storage_mask, w) #h của state đầu
    frontier =[(h0, counter, 0, start)] # (f, tie, g, key)
    came_from = {start: None}   # key -> (parent_key, action) hoặc None cho state đầu
    best_g = {start: 0}
    checks = 0
    while frontier:
        checks += 1
        if checks % 1000 == 0 and time.perf_counter() > deadline:
            return None, came_from

        f, _, g, (robots, boxes) = heapq.heappop(frontier)   # lấy state có f nhỏ nhất

        if g > best_g[(robots, boxes)]:
            continue

        if boxes & ~storage_mask == 0:          # mọi thùng nằm trong storage_mask
            return (robots, boxes), came_from

        for nr, nbx, action in gen_successors(nb, robots, boxes):
             # bỏ state nếu có box bị kẹt ở góc
            if is_deadlock(nb, nbx, storage_mask):
                continue

            key = (nr, nbx)
            ng = g + 1
            if key not in best_g or ng < best_g[key]:       # chưa gặp, hoặc đường mới tốt hơn
                 nh = heuristic_mask(nbx, storage_mask, w)   # h mới
                 best_g[key] = ng
                 came_from[key] = ((robots, boxes), action)
                 counter += 1
                 heapq.heappush(frontier, (ng + nh, counter, ng, key))   # f = g + h
    return None, came_from
def heuristic(state):
    total = 0  # tổng h(n) của cả state

    for box in state.boxes:  # duyệt từng thùng
        min_distance = float("inf")  # khoảng cách nhỏ nhất từ box này tới 1 storage

        for storage in state.storage:  # thử tất cả ô đích
            distance = abs(box[0] - storage[0]) + abs(
                box[1] - storage[1])  # tính khoảng cách Manhattan giữa box và storage

            min_distance = min(min_distance, distance)  # giữ lại storage gần box này nhất

        total += min_distance  # cộng khoảng cách gần nhất của box này vào tổng h(n)

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

    if boxes0 & ~storage_mask == 0:  # đã là goal ngay từ đầu
        return initial_state

    deadline = time.perf_counter() + timebound
    goal_key, came_from = bfs_core(nb, robots0, boxes0, storage_mask, w, deadline)
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
        x, y = current.robots[r]
        dx, dy = DELTAS[d]
        target = (x + dx, y + dy)
        current = next(s for s in current.successors() if s.robots[r] == target)
    return current
