from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from yonokuni.core import (
    ACTION_VECTOR_SIZE,
    Action,
    GameResult,
    apply_action,
    decode_action,
    encode_action,
    enumerate_legal_actions,
    initialize_game_state,
)
from yonokuni.features import (
    AUX_VECTOR_SIZE,
    BOARD_CHANNELS,
    build_aux_vector,
    build_board_tensor,
)


class YonokuniEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        *,
        max_ply: int = 400,
        enforce_legal_actions: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._max_ply = max_ply
        self._enforce_legal = enforce_legal_actions
        self.render_mode = render_mode

        board_shape = (BOARD_CHANNELS, 8, 8)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0.0, high=1.0, shape=board_shape, dtype=np.float32),
                "aux": spaces.Box(low=0.0, high=1.0, shape=(AUX_VECTOR_SIZE,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(ACTION_VECTOR_SIZE)

        self._state = initialize_game_state(max_ply=max_ply)
        self._last_info: Dict[str, np.ndarray] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        max_ply = options.get("max_ply", self._max_ply) if options else self._max_ply
        self._state = initialize_game_state(max_ply=max_ply)
        observation = self._build_observation()
        info = self._build_info()
        self._last_info = info
        return observation, info

    def step(self, action_index: int):
        if not self.action_space.contains(action_index):
            raise ValueError(f"Action index {action_index} out of bounds.")

        legal_mask = self.legal_action_mask()
        if self._enforce_legal and not legal_mask[action_index]:
            raise ValueError("Illegal action provided and enforce_legal_actions=True.")

        action = decode_action(int(action_index))
        self._state = apply_action(self._state, action, in_place=False)

        observation = self._build_observation()
        info = self._build_info()
        self._last_info = info

        reward = self._compute_reward(self._state.result)
        terminated = self._state.result != GameResult.ONGOING
        truncated = False

        return observation, reward, terminated, truncated, info

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for action in enumerate_legal_actions(self._state):
            mask[encode_action(action)] = 1
        return mask

    def render(self):
        if self.render_mode != "ansi":
            raise NotImplementedError("Only 'ansi' render mode is supported.")
        return self._render_ascii()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> Dict[str, np.ndarray]:
        board = build_board_tensor(self._state)
        aux = build_aux_vector(self._state)
        return {"board": board, "aux": aux}

    def _build_info(self) -> Dict[str, np.ndarray]:
        return {"legal_action_mask": self.legal_action_mask()}

    def _compute_reward(self, result: GameResult) -> float:
        if result == GameResult.TEAM_A_WIN:
            return 1.0
        if result == GameResult.TEAM_B_WIN:
            return -1.0
        return 0.0

    def _render_ascii(self) -> str:
        symbols = {0: ".", 1: "R", 2: "B", 3: "Y", 4: "G"}
        rows = []
        for r in range(8):
            row = "".join(symbols[int(self._state.board[r, c])] for c in range(8))
            rows.append(row)
        return "\n".join(rows)
