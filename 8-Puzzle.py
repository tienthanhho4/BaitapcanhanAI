import pygame
import sys
import random
from collections import defaultdict, deque
import heapq
import timeit
import math
import os
import numpy as np
import pandas as pd
import time
import pickle

# EightPuzzle class definition
class EightPuzzle:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.current_state = initial_state  # Thêm current_state, ban đầu bằng initial_state

    def update_current_state(self, new_state):
        self.current_state = new_state

    def get_successors(self, state):
        # Logic để lấy các trạng thái kế tiếp
        pass

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None

    def get_neighbors(self, state):
        row, col = self.find_blank(state)
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        state_list = [list(row) for row in state]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in state_list]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                moves.append(tuple(map(tuple, new_state)))
        return moves

    def state_to_list(self, state):
        return [num for row in state for num in row]

    def list_to_state(self, lst):
        return tuple(tuple(lst[i * 3:(i + 1) * 3]) for i in range(3))

    def generate_random_state(self):
        numbers = list(range(9))
        random.shuffle(numbers)
        state = self.list_to_state(numbers)
        while not self.is_solvable(state):
            random.shuffle(numbers)
            state = self.list_to_state(numbers)
        return state

    def fitness(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] and state[i][j] != self.goal[i][j]:
                    value = state[i][j]
                    goal_i, goal_j = divmod(value - 1, 3)
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return -distance

    def crossover(self, parent1, parent2):
        p1_list = self.state_to_list(parent1)
        p2_list = self.state_to_list(parent2)
        crossover_point = random.randint(1, 7)
        child = p1_list[:crossover_point]
        seen = set(child)
        for num in p2_list:
            if num not in seen:
                child.append(num)
                seen.add(num)
        return self.list_to_state(child)

    def mutate(self, state, mutation_rate=0.05):
        state_list = self.state_to_list(state)
        if random.random() < mutation_rate:
            i, j = random.sample(range(9), 2)
            state_list[i], state_list[j] = state_list[j], state_list[i]
        return self.list_to_state(state_list)

    def reconstruct_path(self, final_state):
        path = [self.initial]
        current = self.initial
        while current != final_state:
            neighbors = self.get_neighbors(current)
            current = min(neighbors, key=lambda x: self.heuristic(x), default=final_state)
            path.append(current)
            if current == final_state:
                break
        return path

    def genetic_algorithm(self, population_size=50, max_generations=500):
        population = [self.generate_random_state() for _ in range(population_size)]
        explored_states = []
        best_fitness = float('-inf')
        no_improvement_count = 0
        max_no_improvement = 100

        for generation in range(max_generations):
            population = sorted(population, key=self.fitness, reverse=True)
            explored_states.extend(population[:5])
            best_state = population[0]
            current_fitness = self.fitness(best_state)

            if best_state == self.goal:
                path = self.reconstruct_path(best_state)
                return path, explored_states

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max_no_improvement:
                break

            new_population = population[:population_size // 2]
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(new_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population

        return None, explored_states

    def bfs(self):
        queue = deque([(self.initial, [])])
        visited = {self.initial}
        explored_states = []

        while queue:
            state, path = queue.popleft()
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [state]))
        return None, explored_states

    def dfs(self, depth_limit=1000):
        stack = [(self.initial, [])]
        visited = {self.initial}
        explored_states = []

        while stack:
            state, path = stack.pop()
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            if len(path) < depth_limit:
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [state]))
        return None, explored_states

    def ucs(self):
        pq = [(0, self.initial, [])]
        visited = {self.initial: 0}
        explored_states = []

        while pq:
            cost, state, path = heapq.heappop(pq)
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                new_cost = cost + 1
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor, path + [state]))
        return None, explored_states

    def ids(self, max_depth=100):
        for depth in range(max_depth + 1):
            stack = [(self.initial, [], 0)]
            visited = set()
            explored_states = []
            while stack:
                state, path, current_depth = stack.pop()
                if state not in visited:
                    visited.add(state)
                    explored_states.append(state)
                    if state == self.goal:
                        return path + [state], explored_states
                    if current_depth < depth:
                        for neighbor in self.get_neighbors(state):
                            if neighbor not in visited:
                                stack.append((neighbor, path + [state], current_depth + 1))
            if explored_states and explored_states[-1] == self.goal:
                return path + [state], explored_states
        return None, explored_states

    def greedy(self):
        pq = [(self.heuristic(self.initial), self.initial, [])]
        visited = {self.initial}
        explored_states = []

        while pq:
            _, state, path = heapq.heappop(pq)
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(pq, (self.heuristic(neighbor), neighbor, path + [state]))
        return None, explored_states

    def a_star(self, timeout=10.0):
        start_time = timeit.default_timer()
        pq = [(self.heuristic(self.initial), 0, self.initial, [])]
        visited = {}
        explored_states = []

        while pq:
            if timeit.default_timer() - start_time > timeout:
                print("A* timeout after", timeout, "seconds")
                return None, explored_states

            f, g, state, path = heapq.heappop(pq)
            if state not in visited or g < visited[state]:
                visited[state] = g
                explored_states.append(state)

                if state == self.goal:
                    return path + [state], explored_states

                for neighbor in self.get_neighbors(state):
                    new_g = g + 1
                    new_f = new_g + self.heuristic(neighbor)
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [state]))
        return None, explored_states

    def ida_star(self, timeout=10.0):
        start_time = timeit.default_timer()
        threshold = self.heuristic(self.initial)
        explored_states = []

        while True:
            if timeit.default_timer() - start_time > timeout:
                print("IDA* timeout after", timeout, "seconds")
                return None, explored_states

            result, new_threshold = self.ida_star_recursive(self.initial, [], 0, threshold, explored_states)
            if result:
                return result, explored_states
            if new_threshold == float('inf'):
                return None, explored_states
            threshold = new_threshold

    def ida_star_recursive(self, state, path, g, threshold, explored_states):
        f = g + self.heuristic(state)
        explored_states.append(state)

        if f > threshold:
            return None, f
        if state == self.goal:
            return path + [state], threshold

        min_threshold = float('inf')
        for neighbor in self.get_neighbors(state):
            if neighbor not in path:
                result, new_threshold = self.ida_star_recursive(neighbor, path + [state], g + 1, threshold, explored_states)
                if result:
                    return result, threshold
                min_threshold = min(min_threshold, new_threshold)

        return None, min_threshold

    def simple_hc(self):
        current = self.initial
        path = [current]
        explored_states = [current]
        while True:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            next_state = min(neighbors, key=self.heuristic)
            if self.heuristic(next_state) >= self.heuristic(current):
                break
            current = next_state
            path.append(current)
            explored_states.append(current)
            if current == self.goal:
                return path, explored_states
        return None, explored_states

    def steepest_hc(self):
        current = self.initial
        path = [current]
        explored_states = [current]
        while True:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            next_state = min(neighbors, key=self.heuristic)
            if self.heuristic(next_state) >= self.heuristic(current):
                break
            current = next_state
            path.append(current)
            explored_states.append(current)
            if current == self.goal:
                return path, explored_states
        return None, explored_states

    def random_hc(self, max_steps=1000):
        current = self.initial
        path = [current]
        explored_states = [current]
        for _ in range(max_steps):
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            next_state = random.choice(neighbors)
            if self.heuristic(next_state) < self.heuristic(current):
                current = next_state
                path.append(current)
                explored_states.append(current)
            if current == self.goal:
                return path, explored_states
        return None, explored_states

    def simulated_annealing(self, initial_temp=1000.0, cooling_rate=0.99, min_temp=0.01, max_no_improvement=2000):
        current = self.initial
        path = [current]
        explored_states = set()
        current_heuristic = self.heuristic(current)
        best_state = current
        best_heuristic = current_heuristic
        no_improvement_count = 0
        temperature = initial_temp

        while temperature > min_temp and no_improvement_count < max_no_improvement:
            explored_states.add(current)
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break

            neighbor_heuristic_pairs = [(neighbor, self.heuristic(neighbor)) for neighbor in neighbors]
            neighbor_heuristic_pairs.sort(key=lambda x: x[1])
            next_state, next_heuristic = neighbor_heuristic_pairs[0]

            if next_heuristic >= current_heuristic:
                delta = next_heuristic - current_heuristic
                acceptance_probability = math.exp(-delta / temperature)
                if random.uniform(0, 1) > acceptance_probability:
                    next_state, next_heuristic = random.choice(neighbor_heuristic_pairs)

            if next_heuristic < current_heuristic:
                no_improvement_count = 0
                if next_heuristic < best_heuristic:
                    best_state = next_state
                    best_heuristic = next_heuristic
            else:
                no_improvement_count += 1

            current = next_state
            current_heuristic = next_heuristic
            path.append(current)

            if current == self.goal:
                return path, list(explored_states)

            temperature *= cooling_rate

        if best_state == self.goal:
            return self.reconstruct_path(best_state), list(explored_states)
        return None, list(explored_states)

    def beam_search(self, beam_width=5):
        initial_state = self.initial
        if initial_state == self.goal:
            return [initial_state], []

        current_states = [initial_state]
        path = {initial_state: []}
        explored_states = set()

        while current_states:
            next_states = []
            for state in current_states:
                explored_states.add(state)
                neighbors = self.get_neighbors(state)
                for neighbor in neighbors:
                    if neighbor not in path:
                        path[neighbor] = path[state] + [state]
                        next_states.append(neighbor)

            evaluated = [(self.heuristic(state), state) for state in next_states]
            evaluated.sort(key=lambda x: x[0])

            current_states = [state for (_, state) in evaluated[:beam_width]]

            if self.goal in current_states:
                return path[self.goal] + [self.goal], list(explored_states)

        return None, list(explored_states)

    def and_or_search(self, max_steps=1000):
        queue = deque([(self.initial, [], {self.initial}, None)])
        visited = set()
        explored_states = []
        num_steps = 0

        while queue and num_steps < max_steps:
            state, path, and_group, action = queue.popleft()
            explored_states.append(state)
            num_steps += 1

            if all(s == self.goal for s in and_group):
                state_path = [p for p, a in path] + [state]
                return state_path, explored_states

            state_tuple = frozenset(and_group)
            if state_tuple in visited:
                continue
            visited.add(state_tuple)

            neighbors = self.get_neighbors(state)
            for action_idx, neighbor in enumerate(neighbors):
                and_states = {neighbor}
                if random.random() < 0.7:
                    i, j = self.find_blank(neighbor)
                    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
                    valid_directions = [(di, dj) for di, dj in directions if 0 <= i + di < 3 and 0 <= j + dj < 3]
                    for di, dj in valid_directions:
                        ni, nj = i + di, j + dj
                        state_list = list(map(list, neighbor))
                        state_list[i][j], state_list[ni][nj] = state_list[ni][nj], state_list[i][j]
                        uncertain_state = tuple(map(tuple, state_list))
                        and_states.add(uncertain_state)

                if and_states:
                    action_name = ["up", "down", "right", "left"][action_idx]
                    queue.append((neighbor, path + [(state, action_name)], and_states, action_name))

        return None, explored_states

    def generate_random_state(self, max_depth=5):
        current_state = self.initial
        for _ in range(random.randint(1, max_depth)):
            neighbors = self.get_neighbors(current_state)
            current_state = random.choice(neighbors)
        return current_state

    def bfs_for_belief(self, start_state, max_depth=10):
        queue = deque([(start_state, 0)])
        visited = {start_state}
        states = set()

        while queue and len(states) < 5:
            state, depth = queue.popleft()
            if depth < max_depth:
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        states.add(neighbor)
        return states

    def belief_state_search(self, initial_belief, max_steps=5000):
        initial_belief = set(initial_belief)
        explored = set()
        num_explored_states = 0
        belief_states_path = [list(initial_belief)]
        total_steps = 0

        for state in initial_belief:
            explored.add(state)
            num_explored_states += 1

        belief_queue = deque([(initial_belief, [])])
        visited = set()

        while belief_queue and num_explored_states < max_steps:
            belief_state, path = belief_queue.popleft()
            belief_state_tuple = frozenset(belief_state)

            if all(state == self.goal for state in belief_state):
                total_steps = 0
                for initial_state in initial_belief:
                    self.initial = initial_state
                    solution, _ = self.bfs()
                    if solution:
                        total_steps += len(solution) - 1
                    else:
                        return None, explored, 0
                belief_states_path.append([self.goal] * len(initial_belief))
                return belief_states_path, explored, total_steps

            if belief_state_tuple in visited:
                continue
            visited.add(belief_state_tuple)

            for action in range(4):
                new_belief = set()
                for state in belief_state:
                    neighbors = self.get_neighbors(state)
                    if action < len(neighbors):
                        next_state = neighbors[action]
                        new_belief.add(next_state)

                        if random.random() < 0.1:
                            i, j = None, None
                            for r in range(3):
                                for c in range(3):
                                    if next_state[r][c] == 0:
                                        i, j = r, c
                                        break

                            directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
                            valid_directions = [(di, dj) for di, dj in directions if
                                                0 <= i + di < 3 and 0 <= j + dj < 3]
                            if valid_directions:
                                di, dj = random.choice(valid_directions)
                                ni, nj = i + di, j + dj
                                state_list = [list(row) for row in next_state]
                                state_list[i][j], state_list[ni][nj] = state_list[ni][nj], state_list[i][j]
                                uncertain_state = tuple(tuple(row) for row in state_list)
                                new_belief.add(uncertain_state)
                    else:
                        new_belief.add(state)

                if new_belief:
                    new_belief = set(sorted(new_belief, key=self.heuristic)[:3])
                    for state in new_belief:
                        if state not in explored:
                            explored.add(state)
                            num_explored_states += 1
                    belief_queue.append((new_belief, path + [min(belief_state, key=self.heuristic)]))
                    belief_states_path.append(list(new_belief))

        return None, explored, 0

    def optimized_bfs_for_belief(self, start_state, max_depth=1):
        queue = deque([(start_state, 0)])
        visited = {start_state}
        states = [(self.heuristic(start_state), start_state)]

        while queue:
            state, depth = queue.popleft()
            if depth < max_depth:
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        states.append((self.heuristic(neighbor), neighbor))

        states.sort()
        return {state for _, state in states[:10]}

    def get_observation(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 1:
                    return (i, j)
        return None

    def find_states_with_one_at_00(self, start_state, max_states=3):
        queue = deque([(start_state, [])])
        visited = {start_state}
        states_with_one_at_00 = []

        while queue and len(states_with_one_at_00) < max_states:
            state, path = queue.popleft()
            if self.get_observation(state) == (0, 0):
                states_with_one_at_00.append(state)
                if len(states_with_one_at_00) >= max_states:
                    break
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [state]))

        while len(states_with_one_at_00) < max_states:
            numbers = list(range(9))
            random.shuffle(numbers)
            numbers[0] = 1
            remaining_numbers = [num for num in numbers[1:] if num != 1]
            if len(remaining_numbers) < 8:
                remaining_numbers.append(0)
            numbers = [1] + remaining_numbers[:8]
            state = self.list_to_state(numbers)
            if self.is_solvable(state) and state not in states_with_one_at_00:
                states_with_one_at_00.append(state)

        return states_with_one_at_00[:max_states]

    def partial_observable_search(self):
        initial_belief = self.find_states_with_one_at_00(self.initial, max_states=3)
        queue = deque([(set(initial_belief), [], 0)])
        visited = set()
        explored_states = []
        belief_states_path = [list(initial_belief)]
        max_steps = 1000

        while queue and len(queue) < max_steps:
            belief_state, path, steps = queue.popleft()
            belief_state_tuple = frozenset(belief_state)
            explored_states.extend(belief_state)

            if all(state == self.goal for state in belief_state):
                total_steps = steps
                belief_states_path.append([self.goal] * 3)
                return belief_states_path, explored_states, total_steps

            if belief_state_tuple in visited:
                continue
            visited.add(belief_state_tuple)

            for action in range(4):
                new_belief = set()
                for state in belief_state:
                    neighbors = self.get_neighbors(state)
                    if action < len(neighbors):
                        next_state = neighbors[action]
                        if self.get_observation(next_state) == (0, 0):
                            new_belief.add(next_state)

                        if random.random() < 0.1:
                            i, j = self.find_blank(next_state)
                            directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
                            valid_directions = [(di, dj) for di, dj in directions if
                                                0 <= i + di < 3 and 0 <= j + dj < 3]
                            if valid_directions:
                                di, dj = random.choice(valid_directions)
                                ni, nj = i + di, j + dj
                                state_list = [list(row) for row in next_state]
                                state_list[i][j], state_list[ni][nj] = state_list[ni][nj], state_list[i][j]
                                uncertain_state = tuple(tuple(row) for row in state_list)
                                if self.get_observation(uncertain_state) == (0, 0):
                                    new_belief.add(uncertain_state)

                if new_belief:
                    new_belief = set(sorted(new_belief, key=self.heuristic)[:3])
                    queue.append((new_belief, path + [min(belief_state, key=self.heuristic)], steps + 1))
                    belief_states_path.append(list(new_belief))

        return None, explored_states, 0

    def is_valid_assignment(self, state, pos, value):
        i, j = pos
        if i == 0 and j == 0 and value != 1:
            return False

        for r in range(3):
            for c in range(3):
                if (r, c) != pos and state[r][c] == value:
                    return False

        if j > 0 and state[i][j - 1] is not None and value != 0 and state[i][j - 1] != value - 1:
            return False
        if j < 2 and value != 0 and state[i][j + 1] is not None and state[i][j + 1] != value + 1:
            return False

        if i > 0 and state[i - 1][j] is not None and value != 0 and state[i - 1][j] != value - 3:
            return False
        if i < 2 and value != 0 and state[i + 1][j] is not None and state[i + 1][j] != value + 3:
            return False

        return True

    def is_solvable(self, state=None):
        if state is None:
            state = self.initial
        state_list = [num for row in state for num in row if num != 0]
        inversions = 0
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                if state_list[i] > state_list[j]:
                    inversions += 1
        return inversions % 2 == 0

    def q_learning_search(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.3, decay=0.995, max_steps=100):
        q_table = {}
        rewards_per_episode = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def find_blank(state):
            for i in range(3):
                for j in range(3):
                    if state[i][j] == 0:
                        return i, j

        def get_neighbors(state):
            i, j = find_blank(state)
            neighbors = []
            for idx, (di, dj) in enumerate(directions):
                ni, nj = i + di, j + dj
                if 0 <= ni < 3 and 0 <= nj < 3:
                    new_state = [list(row) for row in state]
                    new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                    neighbors.append((tuple(map(tuple, new_state)), idx))
            return neighbors

        def heuristic(state):
            dist = 0
            for i in range(3):
                for j in range(3):
                    val = state[i][j]
                    if val == 0:
                        continue
                    gi, gj = divmod(val - 1, 3)
                    dist += abs(i - gi) + abs(j - gj)
            return dist

        initial_state = self.initial
        goal_state = self.goal
        total_states_explored = 0

        for ep in range(episodes):
            state = initial_state
            total_reward = 0
            if state not in q_table:
                q_table[state] = np.zeros(4)
            for step in range(max_steps):
                if np.random.rand() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(q_table[state])

                neighbors = get_neighbors(state)
                next_state = None
                for n, a in neighbors:
                    if a == action:
                        next_state = n
                        break

                if not next_state:
                    reward = -10
                    next_state = state
                else:
                    reward = -heuristic(next_state)
                    if next_state == goal_state:
                        reward = 100

                if next_state not in q_table:
                    q_table[next_state] = np.zeros(4)

                q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
                total_reward += reward
                state = next_state
                total_states_explored += 1
                if state == goal_state:
                    break

            epsilon = max(0.01, epsilon * decay)
            rewards_per_episode.append(total_reward)

        with open("improved_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)

        path = [initial_state]
        state = initial_state
        visited = set([state])
        for _ in range(max_steps):
            if state == goal_state:
                return path, total_states_explored
            if state not in q_table:
                break
            action = np.argmax(q_table[state])
            dx, dy = directions[action]
            i, j = find_blank(state)
            ni, nj = i + dx, j + dy
            if not (0 <= ni < 3 and 0 <= nj < 3):
                break
            new_state = [list(row) for row in state]
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            next_state = tuple(tuple(row) for row in new_state)
            if next_state in visited:
                break
            path.append(next_state)
            visited.add(next_state)
            state = next_state

        if state == goal_state:
            return path, total_states_explored
        return None, total_states_explored

    def heuristic(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] and state[i][j] != self.goal[i][j]:
                    value = state[i][j]
                    goal_i, goal_j = divmod(value - 1, 3)
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

    def backtracking_search(self, depth_limit=9):
        visited = set()
        explored_states = []
        path = []

        def is_valid_assignment(state, pos, value):
            i, j = pos
            if i == 0 and j == 0 and value != 1:
                return False

            for r in range(3):
                for c in range(3):
                    if (r, c) != pos and state[r][c] == value:
                        return False

            if j > 0 and state[i][j - 1] is not None and value != 0 and state[i][j - 1] != value - 1:
                return False
            if j < 2 and state[i][j + 1] is not None and value != 0 and state[i][j + 1] != value + 1:
                return False

            if i > 0 and state[i - 1][j] is not None and value != 0 and state[i - 1][j] != value - 3:
                return False
            if i < 2 and state[i + 1][j] is not None and value != 0 and state[i + 1][j] != value + 3:
                return False

            return True

        def is_solvable(state):
            flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None and state[i][j] != 0]
            inversions = 0
            for i in range(len(flat)):
                for j in range(i + 1, len(flat)):
                    if flat[i] > flat[j]:
                        inversions += 1
            return inversions % 2 == 0

        def backtrack(state, assigned, pos_index):
            if pos_index == 9:
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple == self.goal and is_solvable(state):
                    path.append(state_tuple)
                    return path
                return None

            i, j = divmod(pos_index, 3)
            if i >= 3 or j >= 3:
                return None

            state_tuple = tuple(tuple(row if row is not None else (None, None, None)) for row in state)
            if state_tuple in visited:
                return None
            visited.add(state_tuple)
            explored_states.append(state_tuple)

            for value in range(9):
                if value not in assigned and is_valid_assignment(state, (i, j), value):
                    new_state = [row[:] for row in state]
                    new_state[i][j] = value
                    new_assigned = assigned | {value}

                    path.append(state_tuple)

                    result = backtrack(new_state, new_assigned, pos_index + 1)
                    if result is not None:
                        return result

                    path.pop()

            return None

        empty_state = [[None for _ in range(3)] for _ in range(3)]
        result = backtrack(empty_state, set(), 0)
        return result, explored_states

    def forward_checking_search(self, depth_limit=9):
        visited = set()
        explored_states = []
        path = []

        def get_domain(state, pos, assigned):
            domain = []
            for value in range(9):
                if value not in assigned and self.is_valid_assignment(state, pos, value):
                    domain.append(value)
            return domain

        def forward_check(state, pos, value, domains, assigned):
            i, j = pos
            new_domains = {k: v[:] for k, v in domains.items()}
            used_values = set(state[r][c] for r in range(3) for c in range(3) if state[r][c] is not None)

            related_positions = []
            if j > 0: related_positions.append((i, j - 1))
            if j < 2: related_positions.append((i, j + 1))
            if i > 0: related_positions.append((i - 1, j))
            if i < 2: related_positions.append((i + 1, j))

            for other_pos in related_positions:
                if other_pos not in assigned:
                    r, c = other_pos
                    new_domain = [val for val in new_domains[other_pos] if val not in used_values]
                    if (i, j) == (0, 0) and value == 1:
                        if other_pos == (0, 1):
                            new_domain = [2]
                        elif other_pos == (1, 0):
                            new_domain = [4]
                    elif value != 0:
                        if c > 0 and state[r][c - 1] is not None and state[r][c - 1] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[r][c - 1] == val - 1]
                        if c < 2 and state[r][c + 1] is not None and state[r][c + 1] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[r][c + 1] == val + 1]
                        if r > 0 and state[r - 1][c] is not None and state[r - 1][c] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[r - 1][c] == val - 3]
                        if r < 2 and state[r + 1][c] is not None and state[r + 1][c] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[r + 1][c] == val + 3]
                    new_domains[other_pos] = new_domain
                    if not new_domain:
                        return False, domains
            return True, new_domains

        def select_mrv_variable(positions, domains, state):
            min_domain_size = float('inf')
            selected_pos = None
            for pos in positions:
                domain_size = len(domains[pos])
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    selected_pos = pos
            return selected_pos

        def select_lcv_value(pos, domain, state, domains, assigned):
            value_scores = []
            for value in domain:
                temp_state = [row[:] for row in state]
                temp_state[pos[0]][pos[1]] = value
                _, new_domains = forward_check(temp_state, pos, value, domains, assigned)
                eliminated = sum(len(domains[p]) - len(new_domains[p]) for p in new_domains if p != pos)
                value_scores.append((eliminated, value))
            value_scores.sort()
            return [value for _, value in value_scores]

        def backtrack_with_fc(state, assigned, positions, domains):
            if len(assigned) == 9:
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple == self.goal and self.is_solvable(state):
                    path.append(state_tuple)
                    return path
                return None

            if len(assigned) >= 7:
                temp_state = [row[:] for row in state]
                temp_assigned = assigned.copy()
                temp_positions = [p for p in positions if p not in assigned]
                temp_domains = {k: v[:] for k, v in domains.items()}
                for p in temp_positions:
                    remaining_values = [v for v in range(9) if v not in temp_assigned.values()]
                    if not remaining_values:
                        return None
                    value = remaining_values[0]
                    temp_state[p[0]][p[1]] = value
                    temp_assigned[p] = value
                    temp_tuple = tuple(tuple(row) for row in temp_state)
                    path.append(temp_tuple)
                    success, temp_domains = forward_check(temp_state, p, value, temp_domains, temp_assigned)
                    if not success:
                        path.pop()
                        return None
                state_tuple = tuple(tuple(row) for row in temp_state)
                if state_tuple == self.goal and self.is_solvable(temp_state):
                    return path
                path.pop(len(temp_positions))
                return None

            pos = select_mrv_variable(positions, domains, state)
            if pos is None:
                return None

            domain = get_domain(state, pos, set(assigned.values()))
            sorted_values = select_lcv_value(pos, domain, state, domains, assigned)

            state_tuple = tuple(tuple(row if row is not None else (None, None, None)) for row in state)
            if state_tuple in visited:
                return None
            visited.add(state_tuple)
            explored_states.append(state_tuple)

            for value in sorted_values:
                new_state = [row[:] for row in state]
                new_state[pos[0]][pos[1]] = value
                new_assigned = assigned.copy()
                new_assigned[pos] = value
                new_positions = [p for p in positions if p != pos]
                path.append(state_tuple)

                success, new_domains = forward_check(new_state, pos, value, domains, new_assigned)
                if success:
                    result = backtrack_with_fc(new_state, new_assigned, new_positions, new_domains)
                    if result is not None:
                        return result
                path.pop()

            return None

        empty_state = [[None for _ in range(3)] for _ in range(3)]
        positions = [(i, j) for i in range(3) for j in range(3)]
        domains = {(i, j): list(range(9)) for i in range(3) for j in range(3)}
        assigned = {}
        result = backtrack_with_fc(empty_state, assigned, positions, domains)
        return result, explored_states

    def min_conflicts_search(self, max_iterations=1000, max_no_improvement=100, timeout=5.0):
        def count_conflicts(state):
            conflicts = 0
            value_counts = defaultdict(int)

            if state[0][0] != 1:
                conflicts += 1

            for i in range(3):
                for j in range(3):
                    val = state[i][j]
                    value_counts[val] += 1
                    if value_counts[val] > 1:
                        conflicts += value_counts[val] - 1

            for i in range(3):
                for j in range(2):
                    if state[i][j] != 0 and state[i][j + 1] != 0:
                        if state[i][j + 1] != state[i][j] + 1:
                            conflicts += 1

            for j in range(3):
                for i in range(2):
                    if state[i][j] != 0 and state[i + 1][j] != 0:
                        if state[i + 1][j] != state[i][j] + 3:
                            conflicts += 1

            if all(state[i][j] is not None for i in range(3) for j in range(3)):
                if not self.is_solvable(state):
                    conflicts += 1

            return conflicts

        def get_conflicting_positions(state):
            conflicts = []
            value_counts = defaultdict(int)
            conflict_positions = set()

            if state[0][0] != 1:
                conflict_positions.add((0, 0))

            for i in range(3):
                for j in range(3):
                    val = state[i][j]
                    value_counts[val] += 1
                    if value_counts[val] > 1:
                        conflict_positions.add((i, j))

            for i in range(3):
                for j in range(2):
                    if state[i][j] != 0 and state[i][j + 1] != 0:
                        if state[i][j + 1] != state[i][j] + 1:
                            conflict_positions.add((i, j))
                            conflict_positions.add((i, j + 1))

            for j in range(3):
                for i in range(2):
                    if state[i][j] != 0 and state[i + 1][j] != 0:
                        if state[i + 1][j] != state[i][j] + 3:
                            conflict_positions.add((i, j))
                            conflict_positions.add((i + 1, j))

            if all(state[i][j] is not None for i in range(3) for j in range(3)):
                if not self.is_solvable(state):
                    for i in range(3):
                        for j in range(3):
                            conflict_positions.add((i, j))

            return list(conflict_positions)

        def select_min_conflict_value(state, i, j, current_value, assigned_values):
            value_scores = []
            state_copy = [row[:] for row in state]

            for r in range(3):
                for c in range(3):
                    if (r, c) != (i, j):
                        state_copy = [row[:] for row in state]
                        state_copy[i][j], state_copy[r][c] = state_copy[r][c], state_copy[i][j]
                        conflicts = count_conflicts(state_copy)
                        value_scores.append((conflicts, state[r][c], (r, c)))

            for value in range(9):
                if value not in assigned_values - ({current_value} if current_value is not None else set()):
                    if (i, j) == (0, 0) and value != 1:
                        continue
                    state_copy = [row[:] for row in state]
                    state_copy[i][j] = value
                    conflicts = count_conflicts(state_copy)
                    value_scores.append((conflicts, value, None))

            if not value_scores:
                return None, None

            value_scores.sort()
            return value_scores[0][1], value_scores[0][2]

        def initialize_state():
            state = [[None for _ in range(3)] for _ in range(3)]
            numbers = list(range(9))
            random.shuffle(numbers)
            state[0][0] = 1
            numbers.remove(1)
            idx = 0
            for i in range(3):
                for j in range(3):
                    if (i, j) != (0, 0):
                        state[i][j] = numbers[idx]
                        idx += 1
            return state

        start_time = time.time()
        current_state = initialize_state()
        path = [tuple(tuple(row) for row in current_state)]
        num_explored_states = 1
        best_conflicts = float('inf')
        best_state = [row[:] for row in current_state]
        no_improvement_count = 0
        assigned_values = set(range(9))
        assigned_positions = {(i, j) for i in range(3) for j in range(3)}

        for iteration in range(max_iterations):
            if time.time() - start_time > timeout:
                print("Timeout reached")
                return None, num_explored_states

            current_state_tuple = tuple(tuple(row) for row in current_state)
            conflicts = count_conflicts(current_state)

            if current_state_tuple == self.goal and self.is_solvable(current_state):
                print(f"Solution found after {iteration} iterations")
                return path, num_explored_states

            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_state = [row[:] for row in current_state]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max_no_improvement:
                current_state = initialize_state()
                assigned_values = set(range(9))
                assigned_positions = {(i, j) for i in range(3) for j in range(3)}
                current_state_tuple = tuple(tuple(row) for row in current_state)
                path.append(current_state_tuple)
                num_explored_states += 1
                conflicts = count_conflicts(current_state)
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_state = [row[:] for row in current_state]
                no_improvement_count = 0
                continue

            conflicting_positions = get_conflicting_positions(current_state)
            if not conflicting_positions:
                if conflicts == 0 and self.is_solvable(current_state):
                    print(f"Solution found after {iteration} iterations")
                    return path, num_explored_states
                else:
                    current_state = initialize_state()
                    assigned_values = set(range(9))
                    assigned_positions = {(i, j) for i in range(3) for j in range(3)}
                    current_state_tuple = tuple(tuple(row) for row in current_state)
                    path.append(current_state_tuple)
                    num_explored_states += 1
                    continue

            i, j = random.choice(conflicting_positions)
            current_value = current_state[i][j]

            new_value, swap_pos = select_min_conflict_value(current_state, i, j, current_value, assigned_values)

            if new_value is None:
                continue

            current_state_list = [row[:] for row in current_state]
            if swap_pos:
                r, c = swap_pos
                current_state_list[i][j], current_state_list[r][c] = current_state_list[r][c], current_state_list[i][j]
            else:
                current_state_list[i][j] = new_value
                assigned_values.remove(current_value)
                assigned_values.add(new_value)

            current_state = current_state_list
            current_state_tuple = tuple(tuple(row) for row in current_state)
            path.append(current_state_tuple)
            num_explored_states += 1

        if tuple(tuple(row) for row in best_state) == self.goal and self.is_solvable(best_state):
            print("Returning best state as solution")
            return path, num_explored_states
        print("No solution found")
        return None, num_explored_states

def show_belief_screen(puzzle, screen, WIDTH, HEIGHT):
    # Khởi tạo màn hình tạm thời cho Belief Search
    temp_screen = pygame.Surface((WIDTH, HEIGHT))
    temp_screen.fill((200, 220, 255))  # Màu nền gradient nhẹ

    font = pygame.font.SysFont("Helvetica", 30, bold=True)
    text = font.render("Simulating Belief Search...", True, (255, 255, 255))
    temp_screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 20))

    # Vẽ lưới hiện tại để minh họa
    draw_grid(temp_screen, puzzle.current_state, WIDTH // 2 - 150, HEIGHT // 2 - 150, 100, font, (50, 50, 50))

    screen.blit(temp_screen, (0, 0))
    pygame.display.flip()

    # Giả lập thời gian xử lý (có thể thay bằng logic thực tế)
    pygame.time.wait(1000)  # Chờ 1 giây để mô phỏng
    start_time = pygame.time.get_ticks()

    # Logic giả lập Belief Search (có thể thay bằng thuật toán thực tế)
    explored_states = len(puzzle.get_successors(puzzle.current_state)) * 5  # Số trạng thái giả lập
    steps = random.randint(5, 15)  # Số bước giả lập
    runtime = pygame.time.get_ticks() - start_time

    # Kiểm tra sự kiện thoát
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "QUIT"

    return {
        "runtime": runtime,
        "steps": steps,
        "states_explored": explored_states
    }

def show_pos_screen(puzzle, screen, WIDTH, HEIGHT):
    # Khởi tạo màn hình tạm thời cho Partial Observable Search
    temp_screen = pygame.Surface((WIDTH, HEIGHT))
    temp_screen.fill((200, 220, 255))  # Màu nền gradient nhẹ

    font = pygame.font.SysFont("Helvetica", 30, bold=True)
    text = font.render("Simulating Partial Observable Search...", True, (255, 255, 255))
    temp_screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 20))

    # Vẽ lưới hiện tại để minh họa
    draw_grid(temp_screen, puzzle.current_state, WIDTH // 2 - 150, HEIGHT // 2 - 150, 100, font, (50, 50, 50))

    screen.blit(temp_screen, (0, 0))
    pygame.display.flip()

    # Giả lập thời gian xử lý (có thể thay bằng logic thực tế)
    pygame.time.wait(1000)  # Chờ 1 giây để mô phỏng
    start_time = pygame.time.get_ticks()

    # Logic giả lập Partial Observable Search (có thể thay bằng thuật toán thực tế)
    explored_states = len(puzzle.get_successors(puzzle.current_state)) * 7  # Số trạng thái giả lập
    steps = random.randint(5, 15)  # Số bước giả lập
    runtime = pygame.time.get_ticks() - start_time

    # Kiểm tra sự kiện thoát
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "QUIT"

    return {
        "runtime": runtime,
        "steps": steps,
        "states_explored": explored_states
    }

# Hàm hỗ trợ tạo gradient background
def create_gradient_background(width, height, color1, color2):
    surface = pygame.Surface((width, height))
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (0, y), (width, y))
    return surface

# Hàm vẽ nút với hiệu ứng hover và đổ bóng
def draw_button(screen, rect, text, font, base_color, hover_color, text_color, shadow_color, is_hovered):
    shadow_rect = rect.copy().inflate(8, 8)
    shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(shadow_surface, shadow_color + (100,), shadow_surface.get_rect(), border_radius=15)
    screen.blit(shadow_surface, shadow_rect)

    color = hover_color if is_hovered else base_color
    pygame.draw.rect(screen, color, rect, border_radius=15)
    pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=15)

    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

# Hàm vẽ lưới với hiệu ứng bo góc và đổ bóng
def draw_grid(screen, state, offset_x, offset_y, tile_size, number_font, shadow_color):
    for i in range(3):
        for j in range(3):
            rect = pygame.Rect(offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size)
            shadow_rect = rect.copy().inflate(6, 6)
            shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, shadow_color + (80,), shadow_surface.get_rect(), border_radius=12)
            screen.blit(shadow_surface, shadow_rect)

            if state[i][j] == 0:
                pygame.draw.rect(screen, (240, 240, 240), rect, border_radius=12)
            else:
                pygame.draw.rect(screen, (173, 216, 230), rect, border_radius=12)
            pygame.draw.rect(screen, (50, 50, 50), rect, 2, border_radius=12)

            if state[i][j] != 0:
                text = number_font.render(str(state[i][j]), True, (255, 140, 0))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

# Hàm vẽ tooltip với hiệu ứng mờ
def draw_tooltip(screen, text, font, pos, text_color=(255, 255, 255), bg_color=(50, 50, 50, 200)):
    lines = text.split(" ")
    wrapped_lines = []
    current_line = ""
    for word in lines:
        test_line = current_line + " " + word if current_line else word
        if font.size(test_line)[0] <= 400:
            current_line = test_line
        else:
            wrapped_lines.append(current_line)
            current_line = word
    wrapped_lines.append(current_line)

    line_height = font.size("A")[1] + 5
    tooltip_height = len(wrapped_lines) * line_height + 10
    tooltip_width = 420
    tooltip_surface = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)
    pygame.draw.rect(tooltip_surface, bg_color, tooltip_surface.get_rect(), border_radius=10)

    y = 5
    for line in wrapped_lines:
        text_surface = font.render(line, True, text_color)
        tooltip_surface.blit(text_surface, (10, y))
        y += line_height

    screen.blit(tooltip_surface, (pos[0], pos[1]))

# Hàm vẽ biểu đồ hiệu suất
def draw_performance_plotly(performance_history):
    algorithms = []
    avg_states_explored = []
    avg_runtimes = []

    for algo, runs in performance_history.items():
        if runs:
            algorithms.append(algo)
            states = [run["states_explored"] if isinstance(run["states_explored"], (int, float)) else len(run["states_explored"]) for run in runs]
            avg_states_explored.append(sum(states) / len(states))
            runtimes = [run["runtime"] for run in runs]
            avg_runtimes.append(sum(runtimes) / len(runtimes))

    if not algorithms:
        print("No performance data available.")
        return

    df = pd.DataFrame({
        'Algorithm': algorithms,
        'States Explored': avg_states_explored,
        'Runtime (ms)': avg_runtimes
    })

    print("\nPerformance Comparison:")
    print("-----------------------")
    for _, row in df.iterrows():
        print(f"Algorithm: {row['Algorithm']}")
        print(f"Average States Explored: {row['States Explored']:.2f}")
        print(f"Average Runtime: {row['Runtime (ms)']:.2f} ms")
        print("-----------------------")

# Hàm lưu thông tin thuật toán
def save_algorithm_info(algo_name, runtime, steps, states_explored, path_length):
    with open("algorithm_info.txt", "a") as f:
        f.write(f"\n=== Algorithm Performance Info ===\n")
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Run time: {runtime:.2f} ms\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"States Explored: {states_explored}\n")
        f.write(f"Path Length: {path_length}\n")
        f.write("-" * 30 + "\n")

# Hàm chọn trạng thái ban đầu với giao diện mới
def initial_state_selector(goal_state):
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle Initial State Selector")

    gradient_bg = create_gradient_background(WIDTH, HEIGHT, (135, 206, 250), (255, 182, 193))

    title_font = pygame.font.SysFont("Helvetica", 50, bold=True)
    label_font = pygame.font.SysFont("Helvetica", 35, bold=True)
    input_font = pygame.font.SysFont("Helvetica", 25, bold=True)
    number_font = pygame.font.SysFont("Helvetica", 40, bold=True)

    initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    tile_size = 100
    grid_offset_x = 50
    grid_offset_y = 150
    goal_offset_x = 500
    goal_offset_y = 150
    selected_cell = None
    input_active = False
    input_text = ""

    button_random_rect = pygame.Rect(150, 480, 150, 50)
    button_manual_rect = pygame.Rect(330, 480, 150, 50)
    button_confirm_rect = pygame.Rect(510, 480, 150, 50)
    hovered_button = None

    def is_valid_state(state):
        flat = [num for row in state for num in row]
        return sorted(flat) == list(range(9))

    puzzle_temp = EightPuzzle(initial_state, goal_state)

    running = True
    clock = pygame.time.Clock()
    while running:
        screen.blit(gradient_bg, (0, 0))

        title_text = title_font.render("Set Initial State", True, (255, 255, 255))
        title_shadow = title_font.render("Set Initial State", True, (50, 50, 50))
        screen.blit(title_shadow, (WIDTH // 2 - title_text.get_width() // 2 + 2, 52))
        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 50))

        label_start = label_font.render("Start", True, (255, 255, 255))
        screen.blit(label_start, (grid_offset_x + 50, grid_offset_y - 50))
        draw_grid(screen, initial_state, grid_offset_x, grid_offset_y, tile_size, number_font, (50, 50, 50))

        label_goal = label_font.render("Goal", True, (255, 255, 255))
        screen.blit(label_goal, (goal_offset_x + 50, goal_offset_y - 50))
        draw_grid(screen, goal_state, goal_offset_x, goal_offset_y, tile_size, number_font, (50, 50, 50))

        mouse_pos = pygame.mouse.get_pos()
        for button_rect, label in [(button_random_rect, "Random"), (button_manual_rect, "Manual"), (button_confirm_rect, "Confirm")]:
            is_hovered = button_rect.collidepoint(mouse_pos)
            if is_hovered:
                hovered_button = label
            draw_button(screen, button_rect, label, label_font, (100, 149, 237), (65, 105, 225), (255, 255, 255), (50, 50, 50), is_hovered)

        if input_active:
            input_box = pygame.Rect(150, 540, 500, 40)
            pygame.draw.rect(screen, (255, 255, 255), input_box, border_radius=10)
            pygame.draw.rect(screen, (50, 50, 50), input_box, 2, border_radius=10)
            text_surface = input_font.render(input_text, True, (50, 50, 50))
            screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for i in range(3):
                    for j in range(3):
                        rect = pygame.Rect(grid_offset_x + j * tile_size, grid_offset_y + i * tile_size, tile_size, tile_size)
                        if rect.collidepoint(mouse_pos):
                            selected_cell = (i, j)
                            break
                if button_random_rect.collidepoint(mouse_pos):
                    initial_state = list(map(list, puzzle_temp.generate_random_state()))
                    selected_cell = None
                    input_active = False
                    input_text = ""
                if button_manual_rect.collidepoint(mouse_pos):
                    input_active = True
                    selected_cell = None
                if button_confirm_rect.collidepoint(mouse_pos):
                    if input_active:
                        try:
                            numbers = [int(x) for x in input_text.split() if x.isdigit() and 0 <= int(x) <= 8]
                            if len(numbers) == 9:
                                new_state = [numbers[i:i + 3] for i in range(0, 9, 3)]
                                if is_valid_state(new_state) and puzzle_temp.is_solvable(new_state):
                                    initial_state = new_state
                                    input_active = False
                                    input_text = ""
                                    running = False
                                    return initial_state
                        except:
                            pass
                    else:
                        if is_valid_state(initial_state) and puzzle_temp.is_solvable(initial_state):
                            running = False
                            return initial_state

            elif event.type == pygame.KEYDOWN and selected_cell:
                i, j = selected_cell
                if event.unicode.isdigit() and 0 <= int(event.unicode) <= 8:
                    initial_state[i][j] = int(event.unicode)
                elif event.key == pygame.K_BACKSPACE:
                    initial_state[i][j] = 0
            elif event.type == pygame.KEYDOWN and input_active:
                if event.key == pygame.K_RETURN:
                    try:
                        numbers = [int(x) for x in input_text.split() if x.isdigit() and 0 <= int(x) <= 8]
                        if len(numbers) == 9:
                            new_state = [numbers[i:i + 3] for i in range(0, 9, 3)]
                            if is_valid_state(new_state) and puzzle_temp.is_solvable(new_state):
                                initial_state = new_state
                                input_active = False
                                input_text = ""
                    except:
                        pass
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isspace() or (event.unicode.isdigit() and len(input_text.split()) < 9):
                    input_text += event.unicode

    return initial_state

# Hàm chính của game với giao diện mới
def main_game(initial_state, goal_state):
    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle Solver")

    gradient_bg = create_gradient_background(WIDTH, HEIGHT, (200, 220, 255), (255, 200, 220))

    title_font = pygame.font.SysFont("Helvetica", 60, bold=True)
    label_font = pygame.font.SysFont("Helvetica", 30, bold=True)
    button_font = pygame.font.SysFont("Helvetica", 20, bold=True)
    info_font = pygame.font.SysFont("Helvetica", 25, bold=True)
    number_font = pygame.font.SysFont("Helvetica", 50, bold=True)
    tooltip_font = pygame.font.SysFont("Helvetica", 18, bold=False)

    tooltip_texts = {
        "BFS": "BFS: Duyệt theo chiều rộng, tìm đường đi ngắn nhất.",
        "DFS": "DFS: Duyệt theo chiều sâu, có thể kẹt nhánh.",
        "UCS": "UCS: Tìm đường đi tối ưu theo chi phí.",
        "IDS": "IDS: Kết hợp BFS và DFS theo độ sâu tăng dần.",
        "Greedy": "Greedy: Dựa trên heuristic, nhanh nhưng không tối ưu.",
        "A*": "A*: f(n) = g + h, heuristic dẫn đường tốt.",
        "IDA*": "IDA*: Phiên bản tối ưu hóa bộ nhớ của A*.",
        "SimpleHC": "Simple HC: Leo đồi đơn giản, dễ kẹt cực trị.",
        "SteepHC": "Steepest HC: Chọn nước đi cải thiện tốt nhất.",
        "RandomHC": "Random HC: Leo đồi chọn ngẫu nhiên.",
        "SA": "SA: Làm nguội mô phỏng, tránh kẹt cực trị.",
        "Beam": "Beam: Tìm kiếm theo chùm k trạng thái tốt nhất.",
        "Genetic": "Genetic: Tiến hóa qua lai ghép, đột biến.",
        "AND-OR": "AND-OR: Tìm chiến lược có điều kiện trong thế giới không chắc chắn.",
        "Belief": "Belief: Làm việc với nhiều trạng thái niềm tin.",
        "PartObs": "POS: Tìm kiếm trong môi trường quan sát một phần.",
        "Backtrack": "Backtrack: Gán giá trị từng bước, quay lui khi cần.",
        "Forward": "Forward Checking: Loại bỏ giá trị sai trước khi gán.",
        "MinConf": "Min-Conflicts: Sửa dần trạng thái để giảm xung đột.",
        "QLearn": "Q-Learning: Học chính sách từ thử sai."
    }

    # Vị trí và kích thước lưới
    small_tile_size = 80
    initial_grid_x, initial_grid_y = 50, 100  # Initial ở trên
    goal_grid_x, goal_grid_y = 50, 400       # Goal ở dưới, cách Initial 300 pixel
    algo_tile_size = 150
    algo_grid_x, algo_grid_y = 400, 150      # Lưới chính ở giữa

    # Thiết lập các nút thuật toán thành 2 cột
    button_width, button_height = 110, 40
    button_spacing_x, button_spacing_y = 120, 45
    start_x, start_y = 900, 100  # Bắt đầu từ góc phải trên
    buttons = [
        # Cột 1 (bên trái của 2 cột)
        ("BFS", pygame.Rect(start_x, start_y, button_width, button_height)),
        ("DFS", pygame.Rect(start_x, start_y + button_spacing_y, button_width, button_height)),
        ("UCS", pygame.Rect(start_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("IDS", pygame.Rect(start_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("Greedy", pygame.Rect(start_x, start_y + 4 * button_spacing_y, button_width, button_height)),
        ("A*", pygame.Rect(start_x, start_y + 5 * button_spacing_y, button_width, button_height)),
        ("IDA*", pygame.Rect(start_x, start_y + 6 * button_spacing_y, button_width, button_height)),
        ("SimpleHC", pygame.Rect(start_x, start_y + 7 * button_spacing_y, button_width, button_height)),
        ("SteepHC", pygame.Rect(start_x, start_y + 8 * button_spacing_y, button_width, button_height)),
        ("RandomHC", pygame.Rect(start_x, start_y + 9 * button_spacing_y, button_width, button_height)),
        # Cột 2 (bên phải của 2 cột)
        ("SA", pygame.Rect(start_x + button_spacing_x, start_y, button_width, button_height)),
        ("Beam", pygame.Rect(start_x + button_spacing_x, start_y + button_spacing_y, button_width, button_height)),
        ("Genetic", pygame.Rect(start_x + button_spacing_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("AND-OR", pygame.Rect(start_x + button_spacing_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("Belief", pygame.Rect(start_x + button_spacing_x, start_y + 4 * button_spacing_y, button_width, button_height)),
        ("PartObs", pygame.Rect(start_x + button_spacing_x, start_y + 5 * button_spacing_y, button_width, button_height)),
        ("Backtrack", pygame.Rect(start_x + button_spacing_x, start_y + 6 * button_spacing_y, button_width, button_height)),
        ("Forward", pygame.Rect(start_x + button_spacing_x, start_y + 7 * button_spacing_y, button_width, button_height)),
        ("MinConf", pygame.Rect(start_x + button_spacing_x, start_y + 8 * button_spacing_y, button_width, button_height)),
        ("QLearn", pygame.Rect(start_x + button_spacing_x, start_y + 9 * button_spacing_y, button_width, button_height)),
    ]

    # Nút điều khiển (hàng ngang ở dưới cùng)
    control_button_y = 700
    button_spacing_horizontal = 120
    back_button_rect = pygame.Rect(350, control_button_y, button_width, button_height)
    reset_button_rect = pygame.Rect(350 + button_spacing_horizontal, control_button_y, button_width, button_height)
    view_button_rect = pygame.Rect(350 + 2 * button_spacing_horizontal, control_button_y, button_width, button_height)
    show_info_button_rect = pygame.Rect(350 + 3 * button_spacing_horizontal, control_button_y, button_width, button_height)
    reset_chart_button_rect = pygame.Rect(350 + 4 * button_spacing_horizontal, control_button_y, button_width, button_height)

    puzzle = EightPuzzle(initial_state, goal_state)
    performance_history = {
        "BFS": [], "DFS": [], "UCS": [], "IDS": [], "Greedy": [], "A*": [], "IDA*": [],
        "SimpleHC": [], "SteepHC": [], "RandomHC": [], "SA": [], "Beam": [], "Genetic": [],
        "AND-OR": [], "Belief": [], "PartObs": [], "Backtrack": [], "Forward": [], "MinConf": [], "QLearn": []
    }

    solution = None
    solution_index = 0
    elapsed_time = 0
    steps = 0
    error_message = None
    error_timer = 0
    selected_button = None
    display_state = initial_state
    hovered_button = None
    current_tooltip = ""

    running = True
    clock = pygame.time.Clock()
    while running:
        screen.blit(gradient_bg, (0, 0))

        # Vẽ tiêu đề
        title_text = title_font.render("8-Puzzle Solver", True, (255, 255, 255))
        title_shadow = title_font.render("8-Puzzle Solver", True, (50, 50, 50))
        screen.blit(title_shadow, (WIDTH // 2 - title_text.get_width() // 2 + 2, 32))
        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 30))

        # Vẽ lưới Initial
        label_initial = label_font.render("Initial", True, (255, 255, 255))
        screen.blit(label_initial, (initial_grid_x + 50, initial_grid_y - 40))
        draw_grid(screen, initial_state, initial_grid_x, initial_grid_y, small_tile_size, number_font, (50, 50, 50))

        # Vẽ lưới Goal
        label_goal = label_font.render("Goal", True, (255, 255, 255))
        screen.blit(label_goal, (goal_grid_x + 50, goal_grid_y - 40))
        draw_grid(screen, goal_state, goal_grid_x, goal_grid_y, small_tile_size, number_font, (50, 50, 50))

        # Vẽ lưới chính (giải pháp hoặc trạng thái hiện tại)
        if solution:
            if solution_index < len(solution):
                current_state = solution[solution_index]
                if isinstance(current_state, list) and all(isinstance(s, tuple) for s in current_state):
                    for idx, sub_state in enumerate(current_state):
                        offset_x = algo_grid_x + (idx * algo_tile_size * 3)
                        draw_grid(screen, sub_state, offset_x, algo_grid_y, algo_tile_size, number_font, (50, 50, 50))
                else:
                    draw_grid(screen, current_state, algo_grid_x, algo_grid_y, algo_tile_size, number_font, (50, 50, 50))
                solution_index += 1
                pygame.time.wait(200)
            else:
                draw_grid(screen, solution[-1], algo_grid_x, algo_grid_y, algo_tile_size, number_font, (50, 50, 50))
        else:
            draw_grid(screen, display_state, algo_grid_x, algo_grid_y, algo_tile_size, number_font, (50, 50, 50))

        info_text = info_font.render(f"Time: {elapsed_time:.2f} ms", True, (255, 255, 255))
        screen.blit(info_text, (50, 650))
        steps_text = info_font.render(f"Steps: {steps}", True, (255, 255, 255))
        screen.blit(steps_text, (50, 690))

        if error_message and pygame.time.get_ticks() - error_timer < 1000:
            error_text = info_font.render("No Solution!", True, (255, 50, 50))
            screen.blit(error_text, (50, 680))

        # Vẽ các nút
        mouse_pos = pygame.mouse.get_pos()
        for label, rect in buttons:
            is_hovered = rect.collidepoint(mouse_pos)
            if is_hovered:
                hovered_button = label
                current_tooltip = tooltip_texts.get(label, "")
            draw_button(screen, rect, label, button_font, (100, 149, 237), (65, 105, 225), (255, 255, 255), (50, 50, 50), is_hovered)

        draw_button(screen, back_button_rect, "Back", button_font, (100, 149, 237), (65, 105, 225), (255, 255, 255), (50, 50, 50), back_button_rect.collidepoint(mouse_pos))
        draw_button(screen, reset_button_rect, "Reset", button_font, (100, 149, 237), (65, 105, 225), (255, 255, 255), (50, 50, 50), reset_button_rect.collidepoint(mouse_pos))
        draw_button(screen, view_button_rect, "View Stats", button_font, (100, 149, 237), (65, 105, 225), (255, 255, 255), (50, 50, 50), view_button_rect.collidepoint(mouse_pos))
        draw_button(screen, show_info_button_rect, "Show Info", button_font, (100, 149, 237), (65, 105, 225), (255, 255, 255), (50, 50, 50), show_info_button_rect.collidepoint(mouse_pos))
        draw_button(screen, reset_chart_button_rect, "Reset Chart", button_font, (100, 149, 237), (65, 105, 225), (255, 255, 255), (50, 50, 50), reset_chart_button_rect.collidepoint(mouse_pos))

        if current_tooltip:
            draw_tooltip(screen, current_tooltip, tooltip_font, (10, HEIGHT - 70))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                current_tooltip = ""
                for label, rect in buttons:
                    if rect.collidepoint(mouse_pos):
                        current_tooltip = tooltip_texts.get(label, "")
                        break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if show_info_button_rect.collidepoint(mouse_pos):
                    try:
                        os.system("notepad algorithm_info.txt")
                    except:
                        print("File algorithm_info.txt not found.")
                    continue

                if view_button_rect.collidepoint(mouse_pos):
                    draw_performance_plotly(performance_history)
                    continue

                if reset_button_rect.collidepoint(mouse_pos):
                    solution = None
                    solution_index = 0
                    elapsed_time = 0
                    steps = 0
                    error_message = None
                    display_state = initial_state
                    continue

                if back_button_rect.collidepoint(mouse_pos):
                    return "BACK"

                if reset_chart_button_rect.collidepoint(mouse_pos):
                    performance_history.clear()
                    for key in ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "IDA*", "SimpleHC", "SteepHC", "RandomHC", "SA", "Beam", "Genetic",
                                "AND-OR", "Belief", "PartObs", "Backtrack", "Forward", "MinConf", "QLearn"]:
                        performance_history[key] = []
                    continue

                for label, rect in buttons:
                    if rect.collidepoint(mouse_pos):
                        selected_button = label
                        solution = None
                        solution_index = 0
                        elapsed_time = 0
                        steps = 0
                        error_message = None
                        display_state = initial_state

                        try:
                            start_time = timeit.default_timer()
                            if label == "BFS":
                                solution, explored_states = puzzle.bfs()
                            elif label == "DFS":
                                solution, explored_states = puzzle.dfs()
                            elif label == "UCS":
                                solution, explored_states = puzzle.ucs()
                            elif label == "IDS":
                                solution, explored_states = puzzle.ids()
                            elif label == "Greedy":
                                solution, explored_states = puzzle.greedy()
                            elif label == "A*":
                                solution, explored_states = puzzle.a_star()
                            elif label == "IDA*":
                                solution, explored_states = puzzle.ida_star()
                            elif label == "SimpleHC":
                                solution, explored_states = puzzle.simple_hc()
                            elif label == "SteepHC":
                                solution, explored_states = puzzle.steepest_hc()
                            elif label == "RandomHC":
                                solution, explored_states = puzzle.random_hc()
                            elif label == "SA":
                                solution, explored_states = puzzle.simulated_annealing()
                            elif label == "Beam":
                                solution, explored_states = puzzle.beam_search()
                            elif label == "Genetic":
                                solution, explored_states = puzzle.genetic_algorithm()
                            elif label == "AND-OR":
                                solution, explored_states = puzzle.and_or_search()
                            elif label == "Belief":
                                # Hàm show_belief_screen cần được định nghĩa trong mã gốc
                                result = show_belief_screen(puzzle, screen, WIDTH, HEIGHT)
                                if isinstance(result, dict):
                                    performance_history["Belief"].append(result)
                                elif result == "QUIT":
                                    return "QUIT"
                                continue
                            elif label == "PartObs":
                                # Hàm show_pos_screen cần được định nghĩa trong mã gốc
                                result = show_pos_screen(puzzle, screen, WIDTH, HEIGHT)
                                if isinstance(result, dict):
                                    performance_history["PartObs"].append(result)
                                elif result == "QUIT":
                                    return "QUIT"
                                continue
                            elif label == "Backtrack":
                                solution, explored_states = puzzle.backtracking_search()
                            elif label == "Forward":
                                solution, explored_states = puzzle.forward_checking_search()
                            elif label == "MinConf":
                                solution, num_explored_states = puzzle.min_conflicts_search()
                                explored_states = num_explored_states
                            elif label == "QLearn":
                                solution, explored_states = puzzle.q_learning_search()

                            elapsed_time = (timeit.default_timer() - start_time) * 1000
                            steps = len(solution) - 1 if solution else 0
                            path_length = len(solution) - 1 if solution else 0
                            states_explored = len(explored_states) if isinstance(explored_states, (list, set, tuple)) else int(explored_states)

                            if not solution:
                                error_message = True
                                error_timer = pygame.time.get_ticks()
                            performance_history[label].append({
                                "runtime": elapsed_time,
                                "steps": steps,
                                "states_explored": states_explored if isinstance(explored_states, (list, set, tuple)) else int(explored_states),
                                "path": solution if solution else []
                            })
                            save_algorithm_info(label, elapsed_time, steps, states_explored, path_length)

                        except Exception as e:
                            print(f"Error in {label} algorithm: {e}")
                            error_message = True
                            error_timer = pygame.time.get_ticks()

    return None


# Đảm bảo rằng các hàm khác (như EightPuzzle, draw_performance_plotly, v.v.) vẫn được giữ nguyên từ mã gốc.
if __name__ == "__main__":
    pygame.init()
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    initial_state = initial_state_selector(goal_state)

    if initial_state is None:
        pygame.quit()
        sys.exit()

    while True:
        result = main_game(initial_state, goal_state)
        if result == "BACK":
            initial_state = initial_state_selector(goal_state)
            if initial_state is None:
                break
        else:
            break

    pygame.quit()