"""Student entry point. See README.md for the API and scoring rules."""

import time
import heapq
from collections import deque
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

def gen_macro_successors(nb, robots, boxes):
    """Mỗi successor = 1 robot đi bộ tới sát một thùng rồi đẩy nó 1 ô.
    Trả về (robots mới, boxes mới, (r, dãy hướng), chi phí = số bước đi + 1)."""
    out = []
    for r in range(len(robots)):
        start = robots[r]
        others = set(robots[:r] + robots[r + 1:])   # robot khác là vật cản
        parent = {start: None}                      # ô -> (ô trước, hướng đã đi)
        dist = {start: 0}
        queue = deque([start])
        while queue:
            c = queue.popleft()
            for d in range(4):
                nxt = nb[c][d]
                if nxt < 0 or nxt in others:
                    continue
                if (boxes >> nxt) & 1:                       # ô kế là thùng -> thử đẩy
                    t = nb[nxt][d]
                    if t < 0 or (boxes >> t) & 1 or t in others:
                        continue
                    path = [d]                                # đường đi bộ start -> c, rồi đẩy
                    cur = c
                    while parent[cur] is not None:
                        cur, pd = parent[cur]
                        path.append(pd)
                    path.reverse()
                    new_robots = robots[:r] + (nxt,) + robots[r + 1:]
                    new_boxes = boxes ^ (1 << nxt) ^ (1 << t)
                    out.append((new_robots, new_boxes, (r, tuple(path)), dist[c] + 1))
                    continue
                if nxt not in parent:                         # ô trống -> đi tiếp (BFS)
                    parent[nxt] = (c, d)
                    dist[nxt] = dist[c] + 1
                    queue.append(nxt)
    return out

def gen_single(nb, robots, boxes):
    """Bọc gen_successors cũ về cùng định dạng với macro (mỗi bước chi phí 1)."""
    return [(nr, nbx, (r, (d,)), 1) for nr, nbx, (r, d) in gen_successors(nb, robots, boxes)]

def heuristic_mask(boxes, storages, w):  
    # storages đã được tính 1 lần trong solve(),
    # không cần tạo lại mỗi lần heuristic được gọi

    total = 0  # tổng h(n)

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

def is_corner_deadlock(nb, box_pos, storage_mask):
    """
    Chỉ kiểm tra box vừa được đẩy.
    Nếu box không nằm trên storage nhưng bị kẹt ở góc
    thì đây là deadlock.
    """

    bit = 1 << box_pos

    # box đã nằm trên storage thì không coi là deadlock
    if bit & storage_mask:
        return False

    # 4 ô xung quanh box
    up = nb[box_pos][0]
    right = nb[box_pos][1]
    down = nb[box_pos][2]
    left = nb[box_pos][3]

    # -1 nghĩa là tường hoặc ra ngoài biên
    up_blocked = up == -1
    right_blocked = right == -1
    down_blocked = down == -1
    left_blocked = left == -1

    # 4 trường hợp box bị mắc góc
    if up_blocked and left_blocked:
        return True

    if up_blocked and right_blocked:
        return True

    if down_blocked and left_blocked:
        return True

    if down_blocked and right_blocked:
        return True

    return False

def astar_core(nb, robots0, boxes0, storage_mask, storages, w, deadline,gen):
    """Trả về (goal_key, came_from) nếu tìm thấy, hoặc (None, came_from) nếu hết giờ/không có lời giải."""
    start = (robots0, boxes0)
    counter = 0
    h0 = heuristic_mask(boxes0, storages, w) #h của state đầu
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

        for nr, nbx, action, cost in gen(nb, robots, boxes):
            # Nếu boxes thay đổi thì vừa có một box bị đẩy
            if nbx != boxes:

                # bit mới xuất hiện chính là vị trí mới của box vừa bị đẩy
                moved_box_bits = nbx & ~boxes

                if moved_box_bits:
                    moved_box_pos = moved_box_bits.bit_length() - 1

                    # nếu box vừa đẩy bị mắc góc thì bỏ state này
                    if is_corner_deadlock(nb, moved_box_pos, storage_mask):
                        continue

            key = (nr, nbx)
            ng = g + cost
            if key not in best_g or ng < best_g[key]:       # chưa gặp, hoặc đường mới tốt hơn
                 nh = heuristic_mask(nbx, storages, w)   # h mới
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
    
    # lưu thời điểm bắt đầu toàn bộ solve()
    if timebound is None or timebound <= 0:
        return False
    solve_start = time.perf_counter()
    # ghi lại thời điểm bắt đầu toàn bộ solve()
    w, h = initial_state.width, initial_state.height
    nb = build_neighbors(initial_state)

    robots0 = tuple(y * w + x for (x, y) in initial_state.robots)
    boxes0 = 0
    for (x, y) in initial_state.boxes:
        boxes0 |= 1 << (y * w + x)
    storage_mask = 0
    storages = []  # lưu vị trí các ô đích để heuristic dùng lại

    for (x, y) in initial_state.storage:
        pos = y * w + x

        storage_mask |= 1 << pos
        storages.append(pos)

    if boxes0 & ~storage_mask == 0:  # đã là goal ngay từ đầu
        return initial_state

    deadline = time.perf_counter() + timebound
    macro_deadline = time.perf_counter() + 0.7 * timebound      # dành 70% thời gian cho macro

    # thử macro move trước
    goal_key, came_from = astar_core(nb, robots0, boxes0, storage_mask, storages, w,
                                   macro_deadline, gen_macro_successors)

    # macro không ra lời giải -> thử lại bằng đi từng bước với thời gian còn lại
    if goal_key is None and time.perf_counter() < deadline:
        goal_key, came_from = astar_core(nb, robots0, boxes0, storage_mask, storages, w,
                                       deadline, gen_single)
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
    # mỗi action là (robot r, dãy hướng dirs): đi bộ rồi đẩy
    current = initial_state
    for r, dirs in actions:
        for d in dirs:
            x, y = current.robots[r]
            dx, dy = DELTAS[d]
            target = (x + dx, y + dy)
            current = next(s for s in current.successors() if s.robots[r] == target)
    return current
