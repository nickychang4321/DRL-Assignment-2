import sys
import numpy as np
import random
import copy
import math
from loguru import logger

logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.MCTS = TD_MCTS(self)

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2
        logger.debug(f"test: {stones}")
        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        root = TD_MCTS_Node(copy.deepcopy(game))
        for _ in range(len(root.untried_actions)):
            self.MCTS.run_simulation(root, color)
        best_action, distribution = self.MCTS.best_action_distribution(root)
        move_str = f"{self.index_to_label(best_action[1])}{best_action[0]+1}"
        logger.debug(f"Best action: {best_action}, Distribution: {distribution}, Move: {move_str}")
        self.play_move(color, move_str)

        print(move_str, flush=True)
        return
    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                logger.debug(f"Invalid play command format: {command}")
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

class TD_MCTS_Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [(r, c) for r in range(self.state.size) for c in range(self.state.size) if self.state.board[r, c] == 0]
        # print("test7")

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

class TD_MCTS:
    def __init__(self, env, iterations=500, exploration_constant=1.41, rollout_depth=0, gamma=0.99):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state):
        # print("test5")
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        max_value = -float('inf')
        best_child = None
        for child in node.children.values():
            if child.visits == 0:
                uct_value = float('inf')  # Unvisited nodes are prioritized
            else:
                uct_value = child.total_reward / child.visits + self.c * np.sqrt(np.log(node.visits) / child.visits)
            if uct_value > max_value:
                max_value = uct_value
                best_child = child
        return best_child

    def get_turn(self, sim_env):
        # print("test4")
        # Determine the current turn based on the number of stones on the board.
        count = 0
        for r in range(sim_env.size):
            for c in range(sim_env.size):
                if sim_env.board[r][c] != 0:
                    count += 1
        count %= 4
        now_color = 0
        if count == 1 or count == 2:
            now_color = 2
        else:
            now_color = 1
        return now_color

    def evaluate(self, sim_env):
        # print("test3")
        direct = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = {1 : 0, 2 : 0}
        cost = {0 : 0, 1 : 1, 2 : 10, 3 : 1e2, 4 : 1e3, 5 : 1e4, 6 : 1e10}
        for r in range(sim_env.size):
            for c in range(sim_env.size):
                for dr, dc in direct:
                    if 5 * dr + r >= sim_env.size or 5 * dc + c >= sim_env.size or 6 * dr + r < 0 or 6 * dc + c < 0:
                        continue
                    cnt = [0, 0, 0]
                    nr, nc = r, c
                    for i in range(6):
                        cnt[sim_env.board[nr][nc]] += 1
                        nr += dr
                        nc += dc
                    if cnt[1] == 0:
                        score[2] += cost[cnt[2]]
                    elif cnt[2] == 0:
                        score[1] += cost[cnt[1]]
        return score

    def rollout(self, sim_env, depth, node):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.

        (r, c, color) = node.action
        next_color = 3 - color
        score = 0
        # print("test1")
        sim_env.board[r][c] = color
        score += self.evaluate(sim_env)[color]
        sim_env.board[r][c] = 0

        sim_env.board[r][c] = next_color
        score += self.evaluate(sim_env)[next_color]
        sim_env.board[r][c] = 0

        sim_env.board[r][c] = color

        return score


    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root, color):
        node = root
        # print("test2")
        sim_env = self.create_env_from_state(node.state.board)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            # Select the child with the highest UCT value.
            node = self.select_child(node)
            r, c, color = node.move
            sim_env.board[r][c] = color

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if not node.fully_expanded():
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)

            now_color = self.get_turn(sim_env)

            sim_env.board[action[0]][action[1]] = now_color
            move = (action[0], action[1], now_color)
            new_node = TD_MCTS_Node(copy.deepcopy(sim_env), parent=node, action=move)
            node.children[move] = new_node
            node = new_node

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth, node)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = {}
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.total_reward > best_visits:
                best_visits = child.total_reward
                best_action = action
        return best_action, distribution

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
