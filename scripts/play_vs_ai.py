#!/usr/bin/env python3
"""Play Yonokuni against an AI policy via the console, with optional logging & replay."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from yonokuni import (
    MCTSConfig,
    YonokuniEnv,
    YonokuniNet,
    YonokuniNetConfig,
    make_mcts_policy_from_model,
    RandomPolicy,
)
from yonokuni.core import decode_action
from yonokuni.core.state import Team, GameResult


def load_model(checkpoint_path: str, device: torch.device) -> YonokuniNet:
    model = YonokuniNet(YonokuniNetConfig())
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device=device)
    model.eval()
    return model


def select_ai_action(policy, state, legal_mask: np.ndarray, temperature: float) -> int:
    probs = policy.act(state, legal_mask)
    probs = probs * legal_mask
    if probs.sum() <= 0:
        probs = legal_mask.astype(np.float32)
    probs = probs / probs.sum()
    if temperature <= 1e-6:
        return int(np.argmax(probs))
    adjusted = probs ** (1.0 / temperature)
    adjusted = adjusted / adjusted.sum()
    return int(np.random.choice(len(adjusted), p=adjusted))


def format_board(env: YonokuniEnv) -> str:
    try:
        return env.render()
    except NotImplementedError:
        symbols = {0: ".", 1: "R", 2: "B", 3: "Y", 4: "G"}
        state = env._state
        rows = []
        for r in range(len(state.board)):
            rows.append("".join(symbols[int(cell)] for cell in state.board[r]))
        return "\n".join(rows)


def list_human_moves(env: YonokuniEnv, legal_mask: np.ndarray) -> List:
    indices = np.flatnonzero(legal_mask)
    moves = []
    for idx in indices:
        action = decode_action(int(idx))
        moves.append((idx, action))
    return moves


def prompt_human_move(env: YonokuniEnv, legal_mask: np.ndarray) -> int:
    moves = list_human_moves(env, legal_mask)
    move_indices = {entry[0] for entry in moves}
    print("合法手:")
    for idx, action in moves:
        fr, fc, tr, tc = action.from_row, action.from_col, action.to_row, action.to_col
        print(f"  {idx}: ({fr},{fc}) -> ({tr},{tc})")
    while True:
        raw = input("動かす手の index (q で終了): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("ゲームを終了します。")
            sys.exit(0)
        if not raw.isdigit():
            print("数字を入力してください。")
            continue
        idx = int(raw)
        if idx in move_indices:
            return idx
        print("不正な index です。もう一度。")


def save_log(log: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"ログを {path} に保存しました。")


def replay_logged_game(log_path: Path, *, verbose: bool = True) -> Dict[str, object]:
    data = json.loads(log_path.read_text())
    moves = data.get("moves", [])
    env = YonokuniEnv()
    env.reset()
    if verbose:
        print("ログリプレイを開始します。")
        print(format_board(env))
    for entry in moves:
        idx = entry["action_index"]
        action = decode_action(idx)
        env.step(idx)
        if verbose:
            actor = entry.get("actor", "unknown")
            team = entry.get("team", "?")
            print(
                f"{actor} (Team {team}) の手: ({action.from_row},{action.from_col}) -> ({action.to_row},{action.to_col})"
            )
            print(format_board(env))
    result = env._state.result
    summary = {
        "result": result.value if isinstance(result, GameResult) else str(result),
        "moves": len(moves),
        "board": env._state.board.tolist(),
    }
    if verbose:
        print("リプレイ終了。")
        print(f"結果: {summary['result']}")
    return summary


def play_interactive(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_records: List[Dict] = []

    if args.checkpoint:
        model = load_model(args.checkpoint, device)
        mcts_config = MCTSConfig(
            num_simulations=args.mcts_simulations,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_fraction=0.0,
        )
        policy_ai = make_mcts_policy_from_model(model, config=mcts_config, device=device)
        print(f"チェックポイント {args.checkpoint} を読み込みました。")
    else:
        policy_ai = RandomPolicy()
        print("チェックポイント未指定のため、AI はランダム政策でプレイします。")

    env = YonokuniEnv()
    env._state.max_ply = args.max_ply
    obs, info = env.reset()
    human_team = Team.A if args.human_team == "A" else Team.B

    terminated = False
    move_index = 0
    while not terminated:
        state_snapshot = env._state.copy()
        legal_mask = info["legal_action_mask"]
        team = state_snapshot.current_player.team

        print("\n現在の盤面:")
        print(format_board(env))
        print(f"手番: {state_snapshot.current_player.name} (Team {team.value})")

        if team == human_team:
            action_index = prompt_human_move(env, legal_mask)
            actor = "human"
        else:
            action_index = select_ai_action(policy_ai, state_snapshot, legal_mask, args.temperature)
            actor = "ai"
            action = decode_action(action_index)
            print(
                f"AI ({team.value}) の手: ({action.from_row},{action.from_col}) -> ({action.to_row},{action.to_col})"
            )

        action = decode_action(action_index)
        log_records.append(
            {
                "move_index": move_index,
                "actor": actor,
                "team": team.value,
                "action_index": int(action_index),
                "from": [action.from_row, action.from_col],
                "to": [action.to_row, action.to_col],
            }
        )

        obs, reward, terminated, truncated, info = env.step(action_index)
        move_index += 1
        if truncated:
            terminated = True

    print("\n最終盤面:")
    print(format_board(env))
    result = env._state.result
    if result == GameResult.TEAM_A_WIN:
        print("Team A の勝利！")
    elif result == GameResult.TEAM_B_WIN:
        print("Team B の勝利！")
    else:
        print("引き分け。")

    if args.log_file:
        metadata = {
            "human_team": args.human_team,
            "checkpoint": args.checkpoint,
            "mcts_simulations": args.mcts_simulations,
            "temperature": args.temperature,
            "result": result.value if isinstance(result, GameResult) else str(result),
        }
        log_data = {"metadata": metadata, "moves": log_records}
        save_log(log_data, Path(args.log_file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Yonokuni in the console against AI.")
    parser.add_argument("--checkpoint", help="Path to model checkpoint", default=None)
    parser.add_argument("--human-team", choices=["A", "B"], default="A")
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-ply", type=int, default=400)
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--replay-log", type=str, help="Replay a logged game and exit")
    parser.add_argument("--replay-quiet", action="store_true")
    args = parser.parse_args()

    if args.replay_log:
        replay_logged_game(Path(args.replay_log), verbose=not args.replay_quiet)
        return

    play_interactive(args)


if __name__ == "__main__":
    main()
