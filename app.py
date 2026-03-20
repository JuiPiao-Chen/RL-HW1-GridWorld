from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__)

# Actions: 0=up, 1=down, 2=left, 3=right
ACTIONS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
}
ACTION_SYMBOLS = {0: '\u2191', 1: '\u2193', 2: '\u2190', 3: '\u2192'}

GAMMA = 0.9
STEP_REWARD = -1

# Global grid state
grid_data = {
    'n': None,
    'start': None,
    'end': None,
    'obstacles': [],
    'max_obstacles': 0,
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    n = int(request.json.get('n', 5))
    if n < 5 or n > 9:
        return jsonify({'error': 'n must be between 5 and 9'}), 400

    grid_data['n'] = n
    grid_data['start'] = None
    grid_data['end'] = None
    grid_data['obstacles'] = []
    grid_data['max_obstacles'] = n - 2

    return jsonify({
        'n': n,
        'max_obstacles': n - 2,
        'grid': _build_grid_info()
    })


@app.route('/api/click', methods=['POST'])
def click_cell():
    row = int(request.json['row'])
    col = int(request.json['col'])
    cell = (row, col)
    n = grid_data.get('n')
    if not n or row < 0 or row >= n or col < 0 or col >= n:
        return jsonify({'error': 'Invalid cell'}), 400

    # Toggle off if clicking existing special cell
    if grid_data['start'] == cell:
        grid_data['start'] = None
        return jsonify({'grid': _build_grid_info()})
    if grid_data['end'] == cell:
        grid_data['end'] = None
        return jsonify({'grid': _build_grid_info()})
    if cell in grid_data['obstacles']:
        grid_data['obstacles'].remove(cell)
        return jsonify({'grid': _build_grid_info()})

    # Assign in order: start -> end -> obstacles
    if grid_data['start'] is None:
        grid_data['start'] = cell
    elif grid_data['end'] is None:
        grid_data['end'] = cell
    elif len(grid_data['obstacles']) < grid_data['max_obstacles']:
        grid_data['obstacles'].append(cell)
    else:
        return jsonify({'grid': _build_grid_info(), 'message': 'Maximum obstacles reached'})

    return jsonify({'grid': _build_grid_info()})


@app.route('/api/random_policy', methods=['POST'])
def random_policy():
    n = grid_data.get('n')
    if not n or grid_data.get('start') is None or grid_data.get('end') is None:
        return jsonify({'error': 'Please set grid size, start, and end first'}), 400

    # Generate a random deterministic policy for each non-terminal, non-obstacle cell
    policy = {}
    for r in range(n):
        for c in range(n):
            if (r, c) == grid_data['end'] or (r, c) in grid_data['obstacles']:
                continue
            policy[f"{r},{c}"] = random.randint(0, 3)

    values = _policy_evaluation(n, policy)

    return jsonify({
        'policy': {k: ACTION_SYMBOLS[v] for k, v in policy.items()},
        'values': {k: round(v, 2) for k, v in values.items()},
        'grid': _build_grid_info()
    })


@app.route('/api/value_iteration', methods=['POST'])
def value_iteration():
    n = grid_data.get('n')
    if not n or grid_data.get('start') is None or grid_data.get('end') is None:
        return jsonify({'error': 'Please set grid size, start, and end first'}), 400

    end = grid_data['end']

    # Initialize values
    values = {}
    for r in range(n):
        for c in range(n):
            values[f"{r},{c}"] = 0.0

    # Value iteration
    for _ in range(1000):
        delta = 0
        for r in range(n):
            for c in range(n):
                key = f"{r},{c}"
                if (r, c) == end or (r, c) in grid_data['obstacles']:
                    continue
                best = max(
                    STEP_REWARD + GAMMA * values[_next_key((r, c), a, n)]
                    for a in range(4)
                )
                delta = max(delta, abs(best - values[key]))
                values[key] = best
        if delta < 1e-4:
            break

    # Derive optimal policy
    policy = {}
    for r in range(n):
        for c in range(n):
            if (r, c) == end or (r, c) in grid_data['obstacles']:
                continue
            best_a, best_v = 0, float('-inf')
            for a in range(4):
                v = STEP_REWARD + GAMMA * values[_next_key((r, c), a, n)]
                if v > best_v:
                    best_v = v
                    best_a = a
            policy[f"{r},{c}"] = best_a

    # Trace optimal path
    path = _trace_path(n, policy)

    return jsonify({
        'policy': {k: ACTION_SYMBOLS[v] for k, v in policy.items()},
        'values': {k: round(v, 2) for k, v in values.items()},
        'path': path,
        'grid': _build_grid_info()
    })


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_grid_info():
    n = grid_data.get('n', 5)
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            cell_type = 'normal'
            if grid_data.get('start') == (r, c):
                cell_type = 'start'
            elif grid_data.get('end') == (r, c):
                cell_type = 'end'
            elif (r, c) in grid_data.get('obstacles', []):
                cell_type = 'obstacle'
            row.append({
                'num': r * n + c + 1,
                'type': cell_type,
                'row': r,
                'col': c
            })
        grid.append(row)
    return grid


def _next_state(state, action, n):
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    if nr < 0 or nr >= n or nc < 0 or nc >= n:
        return state
    if (nr, nc) in grid_data['obstacles']:
        return state
    return (nr, nc)


def _next_key(state, action, n):
    ns = _next_state(state, action, n)
    return f"{ns[0]},{ns[1]}"


def _policy_evaluation(n, policy, threshold=1e-4):
    end = grid_data['end']
    values = {f"{r},{c}": 0.0 for r in range(n) for c in range(n)}

    for _ in range(1000):
        delta = 0
        for r in range(n):
            for c in range(n):
                key = f"{r},{c}"
                if (r, c) == end or (r, c) in grid_data['obstacles']:
                    continue
                action = policy.get(key)
                if action is None:
                    continue
                new_val = STEP_REWARD + GAMMA * values[_next_key((r, c), action, n)]
                delta = max(delta, abs(new_val - values[key]))
                values[key] = new_val
        if delta < threshold:
            break
    return values


def _trace_path(n, policy):
    start = grid_data['start']
    end = grid_data['end']
    path = [f"{start[0]},{start[1]}"]
    current = start
    visited = {current}
    for _ in range(n * n):
        if current == end:
            break
        key = f"{current[0]},{current[1]}"
        action = policy.get(key)
        if action is None:
            break
        ns = _next_state(current, action, n)
        if ns in visited:
            break
        visited.add(ns)
        current = ns
        path.append(f"{current[0]},{current[1]}")
    return path


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
