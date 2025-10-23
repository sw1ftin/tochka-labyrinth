import sys
from heapq import heappush, heappop
from typing import Optional
from itertools import count


ENERGY_COST = {
    'A': 1,
    'B': 10,
    'C': 100,
    'D': 1000
}
TARGET_ROOM = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3
}
ROOM_POSITIONS = [2, 4, 6, 8]
HALLWAY_VALID_STOPS = [0, 1, 3, 5, 7, 9, 10]


def parse_input(lines: list[str]) -> tuple[list[Optional[str]], list[list[str]], int]:
    """Парсинг входных данных
    Args:
        lines (list[str]): строки входных данных
    Returns:
        tuple: кортеж из коридора, комнат и глубины комнаты
    """
    hallway = [None] * 11
    rooms = [[], [], [], []]
    
    room_depth = len(lines) - 3
    
    # Парсим комнаты (снизу вверх)
    for i in range(len(lines) - 2, 1, -1):
        line = lines[i]
        if len(line) >= 10:
            for room_idx in range(4):
                pos = 3 + room_idx * 2
                if pos < len(line) and line[pos] in 'ABCD':
                    rooms[room_idx].append(line[pos])
    
    return hallway, rooms, room_depth


def state_to_tuple(hallway: list[Optional[str]], rooms: list[list[str]]) -> tuple:
    """Преобразование состояния в хешируемый tuple для использования в dict в качестве ключа
    Args:
        hallway (list[Optional[str]]): коридор
        rooms (list[list[str]]): комнаты
    Returns:
        tuple: хешируемое представление состояния
    """
    return (tuple(hallway), tuple(tuple(room) for room in rooms))


def is_goal_state(rooms: list[list[str]], room_depth: int) -> bool:
    """Проверка достижения целевого состояния
    Args:
        rooms (list[list[str]]): комнаты
        room_depth (int): глубина комнаты
    Returns:
        bool: True, если достигнуто целевое состояние, иначе False
    """
    targets = TARGET_ROOM.keys()
    for i, target in enumerate(targets):
        if len(rooms[i]) != room_depth:
            return False
        if not all(obj == target for obj in rooms[i]):
            return False
    return True


def can_enter_room(room: list[str], object_type: str) -> bool:
    """Проверка, может ли объект войти в комнату
    Args:
        room (list[str]): Комната
        object_type (str): Тип объекта
    Returns:
        bool: True, если объект может войти, иначе False
    """
    return all(obj == object_type for obj in room)


def get_possible_moves(hallway: list[Optional[str]], rooms: list[list[str]], room_depth: int) -> list[tuple[int, list[Optional[str]], list[list[str]]]]:
    """Генерация всех возможных ходов из текущего состояния
    Args:
        hallway (list[Optional[str]]): коридор
        rooms (list[list[str]]): комнаты
        room_depth (int): глубина комнаты
    Returns:
        list[tuple[int, list[Optional[str]], list[list[str]]]]: список возможных ходов
    """
    moves = []
    
    # 1. Ходы из коридора в комнату
    for hall_pos in range(11):
        if hallway[hall_pos] is None:
            continue
        
        obj = hallway[hall_pos]
        target_room_idx = TARGET_ROOM[obj]
        target_pos = ROOM_POSITIONS[target_room_idx]
        
        if not can_enter_room(rooms[target_room_idx], obj):
            continue
        
        # Проверяем, свободен ли путь в коридоре
        start, end = min(hall_pos, target_pos), max(hall_pos, target_pos)
        if any(hallway[i] is not None for i in range(start, end + 1) if i != hall_pos):
            continue
        
        # Считаем стоимость хода
        steps_in_hallway = abs(hall_pos - target_pos)
        steps_in_room = room_depth - len(rooms[target_room_idx])
        total_steps = steps_in_hallway + steps_in_room
        cost = total_steps * ENERGY_COST[obj]
        
        # Создаем новое состояние
        new_hallway = hallway[:]
        new_hallway[hall_pos] = None
        new_rooms = [room[:] for room in rooms]
        new_rooms[target_room_idx] = new_rooms[target_room_idx] + [obj]
        
        moves.append((cost, new_hallway, new_rooms))
    
    # 2. Ходы из комнаты в коридор
    for room_idx in range(4):
        if len(rooms[room_idx]) == 0:
            continue
        
        # Проверяем, нужно ли вообще выходить из комнаты
        target_type = ['A', 'B', 'C', 'D'][room_idx]
        if all(obj == target_type for obj in rooms[room_idx]):
            continue
        
        # Берем верхний объект
        obj = rooms[room_idx][-1]
        room_pos = ROOM_POSITIONS[room_idx]
        
        # Пробуем переместить в каждую валидную позицию коридора
        for hall_pos in HALLWAY_VALID_STOPS:
            # Проверяем, свободен ли путь
            start, end = min(hall_pos, room_pos), max(hall_pos, room_pos)
            if any(hallway[i] is not None for i in range(start, end + 1)):
                continue
            
            # Считаем стоимость
            steps_in_room = room_depth - len(rooms[room_idx]) + 1
            steps_in_hallway = abs(hall_pos - room_pos)
            total_steps = steps_in_room + steps_in_hallway
            cost = total_steps * ENERGY_COST[obj]
            
            # Создаем новое состояние
            new_hallway = hallway[:]
            new_hallway[hall_pos] = obj
            new_rooms = [room[:] for room in rooms]
            new_rooms[room_idx] = new_rooms[room_idx][:-1]
            
            moves.append((cost, new_hallway, new_rooms))
    
    return moves


def heuristic(hallway: list[Optional[str]], rooms: list[list[str]]) -> int:
    """Эвристическая оценка оставшейся стоимости (A*)
    Args:
        hallway (list[Optional[str]]): Коридор
        rooms (list[list[str]]): Комнаты
    Returns:
        int: Оценка оставшейся стоимости
    """
    total = 0
    
    # Объекты в коридоре
    for pos, obj in enumerate(hallway):
        if obj is not None:
            target_room = TARGET_ROOM[obj]
            target_pos = ROOM_POSITIONS[target_room]
            distance = abs(pos - target_pos)
            total += distance * ENERGY_COST[obj]
    
    # Объекты в неправильных комнатах
    for room_idx, room in enumerate(rooms):
        target_type = ['A', 'B', 'C', 'D'][room_idx]
        for i, obj in enumerate(room):
            if obj != target_type:
                target_room = TARGET_ROOM[obj]
                distance = abs(ROOM_POSITIONS[room_idx] - ROOM_POSITIONS[target_room])
                steps_out = len(room) - i
                total += (steps_out + distance) * ENERGY_COST[obj]
    
    return total


def solve(lines: list[str]) -> int:
    """
    Решение задачи о сортировке в лабиринте

    Рассмотрим каждое состояние лабиринта как узел в графе, а возможные перемещения объектов как ребра с весами, равными затратам энергии на эти перемещения.
    Используем алгоритм A* для поиска пути с минимальной суммарной стоимостью от начального состояния до целевого состояния.
    Для этого используем Priority Queue для хранения состояний в порядке возрастания их оценки стоимости (текущие затраты + эвристическая оценка оставшейся стоимости).

    Args:
        lines: список строк, представляющих лабиринт

    Returns:
        минимальная энергия для достижения целевой конфигурации
    """
    hallway, rooms, room_depth = parse_input(lines)
    
    counter = count()
    initial_state = state_to_tuple(hallway, rooms)
    pq = [(heuristic(hallway, rooms), next(counter), 0, hallway, rooms)]
    visited = {initial_state: 0}
    
    while pq:
        _, _, cost, hallway, rooms = heappop(pq)
        
        # Проверяем целевое состояние
        if is_goal_state(rooms, room_depth):
            return cost
        
        current_state = state_to_tuple(hallway, rooms)
        
        # Если уже нашли лучший путь к этому состоянию, пропускаем
        if visited.get(current_state, float('inf')) < cost:
            continue
        
        # Генерируем возможные ходы
        for move_cost, new_hallway, new_rooms in get_possible_moves(hallway, rooms, room_depth):
            new_cost = cost + move_cost
            new_state = state_to_tuple(new_hallway, new_rooms)
            
            # Добавляем только если нашли лучший путь
            if new_cost < visited.get(new_state, float('inf')):
                visited[new_state] = new_cost
                priority = new_cost + heuristic(new_hallway, new_rooms)
                heappush(pq, (priority, next(counter), new_cost, new_hallway, new_rooms))
    
    return 0


def main():
    # Чтение входных данных
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(result)


if __name__ == "__main__":
    main()