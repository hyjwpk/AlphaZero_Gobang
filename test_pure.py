import time
import copy
import multiprocessing
from mcts_pure import MCTSPlayer
from asyn_mcts_pure import MCTSPlayer as AsyncMCTSPlayer
from game import Board
from policy_value_net_pytorch import PolicyValueNet


def test_mcts_speed(player_class, board_class, n_playout=1000, repeat=3):
    board = board_class(width=9, height=9, n_in_row=5)
    board.init_board()
    player = player_class(n_playout=n_playout)

    times = []
    for i in range(repeat):
        board_copy = copy.deepcopy(board)
        start = time.perf_counter()
        _ = player.get_action(board_copy)
        print(f"action {i + 1} taken {_}")
        end = time.perf_counter()
        if i != 0:
            times.append(end - start)

    avg_time = sum(times) / (repeat - 1)
    print(
        f"{player_class.__name__}: avg_time = {avg_time:.4f}s, playout/s = {n_playout / avg_time:.1f}"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    print("=== MCTS 性能测试 ===")
    test_mcts_speed(MCTSPlayer, Board, n_playout=1000, repeat=3)
    test_mcts_speed(AsyncMCTSPlayer, Board, n_playout=1000, repeat=3)
