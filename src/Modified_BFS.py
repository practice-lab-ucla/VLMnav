from collections import deque, defaultdict

def modified_bfs(edges, goal):
    """
    Find a path from the first node to the last node in the grid
    such that the minimum score along the path is maximized.

    Parameters:
        edges (list): List of tuples in the form 
                      (from_row, from_col, to_row, to_col, score)

    Returns:
        path (list): List of grid cells from start to end
        min_score (float): Minimum score along that path
    """
    # Handle trivial case: only one edge

    if len(edges) == 0:
        return [], None  


    if len(edges) == 1:
        r1, c1, r2, c2, score = edges[0]
        return [(r1, c1), (r2, c2)], score

    # Build graph
    graph = defaultdict(list)
    for r1, c1, r2, c2, score in edges:
        graph[(r1, c1)].append(((r2, c2), score))

    # Define start and goal
    start = edges[0][0], edges[0][1]
    # goal = edges[-1][2], edges[-1][3]

    ################################################# handle if the start = goal #######################################3
    if start == goal:
        return [start], 1.0

    # Modified BFS: track path and minimum score so far
    queue = deque([(start, [start], float('inf'))])
    visited = {}

    best_path = []
    best_min_score = -1

    while queue:
        node, path, min_score_so_far = queue.popleft()

        # Skip if already visited with better or equal min score
        if node in visited and visited[node] >= min_score_so_far:
            continue
        visited[node] = min_score_so_far

        # Reached goal
        if node == goal:
            if min_score_so_far > best_min_score:
                best_min_score = min_score_so_far
                best_path = path
            continue

        # Explore neighbors
        for neighbor, score in graph[node]:
            new_min_score = min(min_score_so_far, score)
            queue.append((neighbor, path + [neighbor], new_min_score))

    return best_path, best_min_score
