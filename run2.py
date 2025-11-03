import sys
from collections import deque
from functools import lru_cache
from typing import List, Tuple, Set, Dict, Optional


VIRUS_START_POSITION = 'a'
INFINITE_DISTANCE = 10**9


def build_graph(edges: Set[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Построение графа из списка рёбер
    
    Args:
        edges: множество рёбер
    Returns:
        граф в виде словаря смежности
    """
    g: Dict[str, Set[str]] = {}
    for u, v in edges:
        g.setdefault(u, set()).add(v)
        g.setdefault(v, set()).add(u)
    return g


def canonical_edge(a: str, b: str) -> Tuple[str, str]:
    """Канонизация ребра для хеширования"""
    return (a, b) if a <= b else (b, a)


def get_gates(nodes: Set[str]) -> List[str]:
    """Получение отсортированного списка шлюзов"""
    return sorted([n for n in nodes if n and n[0].isupper()])


def compute_virus_move(graph: Dict[str, Set[str]], virus: str) -> Optional[Tuple[str, str, Dict[str, int]]]:
    """Вычисление хода вируса
    
    Args:
        graph: граф сети
        virus: текущая позиция вируса
    Returns:
        tuple | None: (выбранный_шлюз, следующая_позиция, расстояния) или None, если нет доступных шлюзов
    """
    nodes = set(graph.keys())
    gates = get_gates(nodes)
    
    best_gate = None
    best_dist = None
    best_dist_map = None
    
    for gate in gates:
        dist: Dict[str, int] = {gate: 0}
        q = deque([gate])
        while q:
            cur = q.popleft()
            for nb in graph.get(cur, ()): 
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        
        if virus not in dist:
            continue
        
        d = dist[virus]
        if best_dist is None or d < best_dist or (d == best_dist and gate < best_gate):
            best_gate = gate
            best_dist = d
            best_dist_map = dist

    if best_gate is None:
        return None

    dvirus = best_dist_map[virus]
    if dvirus == 0:
        return (best_gate, best_gate, best_dist_map)

    candidates = []
    for nb in sorted(graph.get(virus, [])):
        if best_dist_map.get(nb, INFINITE_DISTANCE) == dvirus - 1:
            candidates.append(nb)
    
    if not candidates:
        return None
    
    next_node = candidates[0]
    return (best_gate, next_node, best_dist_map)


@lru_cache(maxsize=None)
def search(edges_frozen: frozenset, virus_pos: str) -> Optional[Tuple[Tuple[str, str], List[Tuple[str, str]]]]:
    """Рекурсивный поиск последовательности отключений с возвратом (backtracking)
    
    Args:
        edges_frozen: текущее состояние графа (множество рёбер)
        virus_pos: текущая позиция вируса
    Returns:
        tuple | None: ((gate, node), [остальные_отключения]) или None, если решения нет
    """
    graph = build_graph(set(edges_frozen))
    move = compute_virus_move(graph, virus_pos)
    
    if move is None:
        return (None, [])

    candidates: List[Tuple[str, str]] = []
    for (u, v) in sorted(edges_frozen):
        if u[0].isupper() and not v[0].isupper():
            candidates.append((u, v))
        elif v[0].isupper() and not u[0].isupper():
            candidates.append((v, u))
    
    seen = set()
    filtered = []
    for g, n in candidates:
        if (g, n) not in seen:
            seen.add((g, n))
            filtered.append((g, n))

    if not filtered:
        return None

    for gate, node in filtered:
        ce = canonical_edge(gate, node)
        if ce not in edges_frozen:
            continue
        
        new_edges = set(edges_frozen)
        new_edges.remove(ce)
        new_edges_f = frozenset(new_edges)

        new_graph = build_graph(set(new_edges_f))
        new_move = compute_virus_move(new_graph, virus_pos)
        
        if new_move is None:
            return ((gate, node), [])
        
        _, next_pos2, _ = new_move
        if next_pos2 and next_pos2[0].isupper():
            continue

        deeper = search(new_edges_f, next_pos2)
        if deeper is not None:
            first_pair, rest_list = deeper
            seq = [(gate, node)]
            if first_pair is not None:
                seq.append(first_pair)
            seq.extend(rest_list)
            return (seq[0], seq[1:]) if len(seq) > 1 else (seq[0], [])

    return None


def solve(edges: list[tuple[str, str]]) -> list[str]:
    """Решение задачи об изоляции вируса
    
    Поиск с возвратом с мемоизацией для нахождения 
    последовательности отключений коридоров, которая изолирует вирус
    
    Алгоритм:
      1. На каждом шаге рассматриваем все возможные отключения
      2. Отключения перебираем в лексикографическом порядке
      3. После каждого отключения симулируем детерминированный ход вируса
      4. Если вирус достигает шлюза - этот вариант неудачный, пробуем следующий
      5. Используем мемоизацию для избежания повторных вычислений
    
    Args:
        edges: список коридоров в формате (узел1, узел2)
    Returns:
        список отключаемых коридоров в формате "Шлюз-узел"
    """
    edge_set: Set[Tuple[str, str]] = set()
    for a, b in edges:
        edge_set.add(canonical_edge(a, b))

    sequence: List[Tuple[str, str]] = []
    cur_edges = frozenset(edge_set)
    cur_virus = VIRUS_START_POSITION
    
    while True:
        res = search(cur_edges, cur_virus)
        if res is None:
            break
        
        first_pair, _ = res
        if first_pair is None:
            break
        
        sequence.append(first_pair)
        ce = canonical_edge(first_pair[0], first_pair[1])
        cur_edges = frozenset(set(cur_edges) - {ce})

        graph = build_graph(set(cur_edges))
        move = compute_virus_move(graph, cur_virus)
        if move is None:
            break
        
        _, next_node, _ = move
        if next_node and next_node[0].isupper():
            break
        cur_virus = next_node

    return [f"{g}-{n}" for g, n in sequence]


def main():
    edges = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            node1, sep, node2 = line.partition('-')
            if sep:
                edges.append((node1, node2))

    result = solve(edges)
    for edge in result:
        print(edge)


if __name__ == "__main__":
    main()