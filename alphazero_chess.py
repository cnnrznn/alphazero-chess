"""
AlphaZero for Chess
A complete implementation of the AlphaZero algorithm for chess.
Requires: torch, numpy, python-chess
"""

import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

# Enable MPS fallback for unsupported operations on Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Neural network (reduced for CPU)
    num_res_blocks: int = 5  # Reduced from 19
    num_filters: int = 64    # Reduced from 256

    # MCTS (reduced for CPU)
    num_simulations: int = 100  # Reduced from 800
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Parallel MCTS
    virtual_loss: float = 1.0  # For parallel tree search
    batch_eval_size: int = 8   # Batch multiple positions for inference

    # Training
    batch_size: int = 64      # Reduced from 256
    learning_rate: float = 0.01  # Reduced from 0.2
    weight_decay: float = 1e-4
    momentum: float = 0.9
    num_epochs: int = 1
    num_workers: int = 4      # DataLoader workers

    # Self-play (reduced for CPU)
    num_actors: int = 100     # Reduced from 5000
    max_moves: int = 200      # Reduced from 512
    temperature_threshold: int = 15  # Reduced from 30
    num_parallel_games: int = mp.cpu_count() # Parallel self-play games

    # Replay buffer
    buffer_size: int = 50_000      # Reduced from 1_000_000
    min_buffer_size: int = 1_000   # Reduced from 10_000


def get_device() -> torch.device:
    """Detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_config_for_device(device: torch.device) -> Config:
    """Return optimized configuration based on the target device."""
    cfg = Config()

    if device.type == 'cuda':
        # NVIDIA GPU (e.g., 3090 with 24GB VRAM) - use larger settings
        cfg.num_res_blocks = 10
        cfg.num_filters = 128
        cfg.batch_size = 256
        cfg.num_simulations = 400
        cfg.num_workers = 4
        cfg.buffer_size = 200_000
        cfg.min_buffer_size = 5_000

    elif device.type == 'mps':
        # Apple Silicon (e.g., M2 Pro) - good GPU but shared memory
        cfg.num_res_blocks = 8
        cfg.num_filters = 96
        cfg.batch_size = 128
        cfg.num_simulations = 200
        cfg.num_workers = 0  # MPS doesn't benefit from DataLoader workers
        cfg.buffer_size = 100_000
        cfg.min_buffer_size = 2_000

    # CPU uses default reduced settings from Config class
    return cfg


# ============================================================================
# Board Encoding (19 planes of 8x8)
# ============================================================================

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode board state into neural network input.
    Returns shape (19, 8, 8):
    - 12 planes for piece positions (6 piece types x 2 colors)
    - 4 planes for castling rights
    - 1 plane for side to move
    - 1 plane for total move count
    - 1 plane for no-progress count (50-move rule)
    """
    planes = np.zeros((19, 8, 8), dtype=np.float32)

    # Piece planes (0-11) - optimized with numpy indexing
    piece_idx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            rank, file = divmod(sq, 8)
            planes[piece_idx[piece.symbol()], rank, file] = 1.0

    # Castling rights (12-15) - use fill instead of broadcasting
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12].fill(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13].fill(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14].fill(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15].fill(1.0)

    # Side to move (16)
    if board.turn == chess.WHITE:
        planes[16].fill(1.0)

    # Move counts (17-18)
    planes[17].fill(board.fullmove_number / 100.0)
    planes[18].fill(board.halfmove_clock / 100.0)

    return planes

# ============================================================================
# Move Encoding (4672 possible moves)
# ============================================================================

def _init_move_indices():
    """
    Create mapping from moves to policy indices.
    AlphaZero uses 73 planes of 8x8 = 4672 moves:
    - 56 planes for queen moves (7 squares x 8 directions)
    - 8 planes for knight moves
    - 9 planes for underpromotions (3 piece types x 3 directions)
    """
    move_to_idx = {}
    idx_to_move = {}
    idx = 0
    
    # Queen-like moves (including promotions to queen)
    directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    for from_sq in range(64):
        from_rank, from_file = divmod(from_sq, 8)
        for dir_idx, (dr, df) in enumerate(directions):
            for dist in range(1, 8):
                to_rank = from_rank + dr * dist
                to_file = from_file + df * dist
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    move = chess.Move(from_sq, to_sq)
                    move_to_idx[move.uci()] = idx
                    idx_to_move[idx] = move
                    # Queen promotion
                    if (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0):
                        if abs(df) <= 1 and dr != 0:
                            promo_move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                            move_to_idx[promo_move.uci()] = idx
                    idx += 1
    
    # Knight moves
    knight_deltas = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]
    for from_sq in range(64):
        from_rank, from_file = divmod(from_sq, 8)
        for dr, df in knight_deltas:
            to_rank = from_rank + dr
            to_file = from_file + df
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                move = chess.Move(from_sq, to_sq)
                move_to_idx[move.uci()] = idx
                idx_to_move[idx] = move
                idx += 1
    
    # Underpromotions (knight, bishop, rook)
    for from_file in range(8):
        for df in [-1, 0, 1]:
            to_file = from_file + df
            if 0 <= to_file < 8:
                for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                    # White promotion
                    from_sq = 6 * 8 + from_file
                    to_sq = 7 * 8 + to_file
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    move_to_idx[move.uci()] = idx
                    idx_to_move[idx] = move
                    idx += 1
                    # Black promotion
                    from_sq = 1 * 8 + from_file
                    to_sq = 0 * 8 + to_file
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    move_to_idx[move.uci()] = idx
                    idx_to_move[idx] = move
                    idx += 1
    
    return move_to_idx, idx_to_move, idx

MOVE_TO_IDX, IDX_TO_MOVE, NUM_MOVES = _init_move_indices()

def encode_move(move: chess.Move) -> int:
    return MOVE_TO_IDX.get(move.uci(), 0)

def decode_move(idx: int) -> chess.Move:
    return IDX_TO_MOVE.get(idx, chess.Move.null())

# ============================================================================
# Neural Network
# ============================================================================

class ResBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x

class AlphaZeroNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        nf = cfg.num_filters
        
        # Input convolution
        self.conv_input = nn.Conv2d(19, nf, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(nf)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(nf) for _ in range(cfg.num_res_blocks)
        ])
        
        # Policy head
        self.conv_policy = nn.Conv2d(nf, 32, 1, bias=False)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 64, NUM_MOVES)
        
        # Value head
        self.conv_value = nn.Conv2d(nf, 1, 1, bias=False)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(64, 256)
        self.fc_value2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Input block
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)
        
        # Value head
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))
        
        return p, v

# ============================================================================
# MCTS
# ============================================================================

class MCTSNode:
    __slots__ = ['parent', 'move', 'prior', 'children', 'visit_count',
                 'value_sum', 'board', 'virtual_loss']

    def __init__(self, parent: Optional['MCTSNode'], move: Optional[chess.Move],
                 prior: float, board: chess.Board):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.board = board
        self.children: dict[chess.Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0  # For parallel search

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

class MCTS:
    def __init__(self, cfg: Config, network: AlphaZeroNet, device: torch.device):
        self.cfg = cfg
        self.network = network
        self.device = device
        self.eval_cache = {}  # Cache for board evaluations

    def clear_cache(self):
        """Clear evaluation cache to free memory"""
        self.eval_cache.clear()

    def search(self, board: chess.Board, add_noise: bool = True) -> np.ndarray:
        root = MCTSNode(None, None, 0.0, board.copy())

        # Use inference mode for better CPU performance
        with torch.inference_mode():
            self._expand_single(root)

            # Add Dirichlet noise to root
            if add_noise and root.children:
                noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(root.children))
                for i, child in enumerate(root.children.values()):
                    child.prior = (1 - self.cfg.dirichlet_epsilon) * child.prior + \
                                  self.cfg.dirichlet_epsilon * noise[i]

            # Run simulations in batches
            num_batches = self.cfg.num_simulations // self.cfg.batch_eval_size
            for _ in range(num_batches):
                search_paths = []
                nodes_to_eval = []

                # Collect batch of nodes
                for _ in range(self.cfg.batch_eval_size):
                    node = root
                    search_path = [node]

                    # Select with virtual loss
                    while node.is_expanded() and not node.board.is_game_over():
                        node = self._select_child(node)
                        search_path.append(node)
                        # Add virtual loss to discourage other threads from selecting same path
                        node.virtual_loss += self.cfg.virtual_loss

                    search_paths.append(search_path)
                    if not node.board.is_game_over() and not node.is_expanded():
                        nodes_to_eval.append(node)

                # Batch evaluate all nodes
                if nodes_to_eval:
                    values = self._batch_expand_and_evaluate(nodes_to_eval)
                else:
                    values = []

                # Process results
                eval_idx = 0
                for search_path in search_paths:
                    node = search_path[-1]

                    # Remove virtual loss
                    for n in search_path[1:]:
                        n.virtual_loss -= self.cfg.virtual_loss

                    # Get value
                    if node.board.is_game_over():
                        result = node.board.result()
                        if result == "1-0":
                            value = 1.0 if node.board.turn == chess.BLACK else -1.0
                        elif result == "0-1":
                            value = 1.0 if node.board.turn == chess.WHITE else -1.0
                        else:
                            value = 0.0
                    else:
                        value = values[eval_idx] if eval_idx < len(values) else 0.0
                        eval_idx += 1

                    # Backpropagate
                    self._backpropagate(search_path, value)

        # Build policy from visit counts
        policy = np.zeros(NUM_MOVES, dtype=np.float32)
        for move, child in root.children.items():
            policy[encode_move(move)] = child.visit_count

        if policy.sum() > 0:
            policy /= policy.sum()

        return policy
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent = math.sqrt(node.visit_count)
        
        for child in node.children.values():
            # UCB formula
            q_value = -child.value()  # Negated for opponent's perspective
            u_value = self.cfg.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_single(self, node: MCTSNode):
        """Expand a single node (used for root)"""
        if node.board.is_game_over():
            return

        # Check cache first
        board_fen = node.board.fen()
        if board_fen in self.eval_cache:
            policy, _ = self.eval_cache[board_fen]
        else:
            # Get policy from network
            state = torch.from_numpy(encode_board(node.board)).unsqueeze(0).to(self.device)
            policy_logits, value = self.network(state)

            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            # Cache result
            self.eval_cache[board_fen] = (policy, value.item())
            # Limit cache size
            if len(self.eval_cache) > 10000:
                # Remove oldest entries (simple FIFO)
                self.eval_cache.pop(next(iter(self.eval_cache)))

        # Create children for legal moves
        legal_moves = list(node.board.legal_moves)
        priors = np.array([policy[encode_move(m)] for m in legal_moves])

        if priors.sum() > 0:
            priors /= priors.sum()
        else:
            priors = np.ones(len(legal_moves)) / len(legal_moves)

        for move, prior in zip(legal_moves, priors):
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(node, move, prior, child_board)

    def _batch_expand_and_evaluate(self, nodes: list[MCTSNode]) -> list[float]:
        """Batch expand and evaluate multiple nodes"""
        if not nodes:
            return []

        # Check cache and collect uncached nodes
        results = [None] * len(nodes)
        states_to_eval = []
        indices_to_eval = []

        for i, node in enumerate(nodes):
            if node.board.is_game_over():
                continue

            board_fen = node.board.fen()
            if board_fen in self.eval_cache:
                policy, value = self.eval_cache[board_fen]
                results[i] = (policy, value)
            else:
                states_to_eval.append(encode_board(node.board))
                indices_to_eval.append(i)

        # Batch evaluate uncached positions
        if states_to_eval:
            states_tensor = torch.from_numpy(np.array(states_to_eval)).to(self.device)
            policy_logits, values = self.network(states_tensor)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()

            for idx, node_idx in enumerate(indices_to_eval):
                node = nodes[node_idx]
                policy = policies[idx]
                value = float(values_np[idx])
                results[node_idx] = (policy, value)

                # Cache result
                board_fen = node.board.fen()
                self.eval_cache[board_fen] = (policy, value)
                if len(self.eval_cache) > 10000:
                    self.eval_cache.pop(next(iter(self.eval_cache)))

        # Expand all nodes and return values
        values = []
        for i, node in enumerate(nodes):
            if results[i] is None:
                values.append(0.0)
                continue

            policy, value = results[i]
            values.append(value)

            # Create children
            legal_moves = list(node.board.legal_moves)
            priors = np.array([policy[encode_move(m)] for m in legal_moves])

            if priors.sum() > 0:
                priors /= priors.sum()
            else:
                priors = np.ones(len(legal_moves)) / len(legal_moves)

            for move, prior in zip(legal_moves, priors):
                child_board = node.board.copy()
                child_board.push(move)
                node.children[move] = MCTSNode(node, move, prior, child_board)

        return values

    def _evaluate(self, board: chess.Board) -> float:
        # Check cache first
        board_fen = board.fen()
        if board_fen in self.eval_cache:
            _, value = self.eval_cache[board_fen]
            return value

        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.network(state)

        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value_item = value.item()
        # Cache result
        self.eval_cache[board_fen] = (policy, value_item)
        # Limit cache size
        if len(self.eval_cache) > 10000:
            self.eval_cache.pop(next(iter(self.eval_cache)))

        return value_item
    
    def _backpropagate(self, path: list[MCTSNode], value: float):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent

# ============================================================================
# Self-Play
# ============================================================================

@dataclass
class GameSample:
    state: np.ndarray
    policy: np.ndarray
    value: float

def _self_play_worker(network_state: dict, cfg: Config, seed: int) -> list[GameSample]:
    """Worker function for parallel self-play"""
    np.random.seed(seed)
    random.seed(seed)

    # Create network and MCTS in worker process
    device = torch.device('cpu')
    network = AlphaZeroNet(cfg).to(device)
    network.load_state_dict(network_state)
    network.eval()

    mcts = MCTS(cfg, network, device)
    return self_play_game(mcts, cfg)

def self_play_game(mcts: MCTS, cfg: Config) -> list[GameSample]:
    board = chess.Board()
    samples = []
    move_count = 0

    # Clear cache at start of game to free memory
    mcts.clear_cache()

    while not board.is_game_over() and move_count < cfg.max_moves:
        # Get MCTS policy
        policy = mcts.search(board, add_noise=True)
        
        # Store sample
        samples.append(GameSample(
            state=encode_board(board),
            policy=policy.copy(),
            value=0.0  # Will be filled later
        ))
        
        # Select move
        if move_count < cfg.temperature_threshold:
            # Sample proportionally
            probs = policy / policy.sum() if policy.sum() > 0 else np.ones(len(policy)) / len(policy)
            move_idx = np.random.choice(len(policy), p=probs)
        else:
            # Greedy
            move_idx = np.argmax(policy)
        
        # Find corresponding legal move
        legal_moves = list(board.legal_moves)
        move_probs = [(m, policy[encode_move(m)]) for m in legal_moves]
        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        if move_count < cfg.temperature_threshold:
            weights = [p for _, p in move_probs]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                move = random.choices([m for m, _ in move_probs], weights=weights)[0]
            else:
                move = random.choice(legal_moves)
        else:
            move = move_probs[0][0]
        
        board.push(move)
        move_count += 1
    
    # Determine game result
    result = board.result()
    if result == "1-0":
        final_value = 1.0
    elif result == "0-1":
        final_value = -1.0
    else:
        final_value = 0.0
    
    # Assign values (alternating perspective)
    for i, sample in enumerate(samples):
        sample.value = final_value if i % 2 == 0 else -final_value
    
    return samples

def parallel_self_play(network: AlphaZeroNet, cfg: Config, num_games: int) -> list[GameSample]:
    """Generate multiple self-play games in parallel"""
    # Move state dict to CPU for worker processes (works from any device: CUDA, MPS, CPU)
    network_state = {k: v.cpu() for k, v in network.state_dict().items()}
    all_samples = []

    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=cfg.num_parallel_games) as executor:
        futures = []
        for i in range(num_games):
            seed = random.randint(0, 2**31 - 1)
            future = executor.submit(_self_play_worker, network_state, cfg, seed)
            futures.append(future)

        # Collect results as they complete
        for i, future in enumerate(as_completed(futures)):
            try:
                samples = future.result()
                all_samples.extend(samples)
                print(f"  Game {i + 1}/{num_games} completed: {len(samples)} moves")
            except Exception as e:
                print(f"  Game {i + 1}/{num_games} failed: {e}")

    return all_samples

# ============================================================================
# Training
# ============================================================================

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, samples: list[GameSample]):
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        states = np.array([s.state for s in samples])
        policies = np.array([s.policy for s in samples])
        values = np.array([s.value for s in samples])
        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)

    def state_dict(self) -> dict:
        """Save replay buffer state"""
        return {
            'buffer': list(self.buffer),
            'max_size': self.max_size
        }

    def load_state_dict(self, state_dict: dict):
        """Load replay buffer state"""
        self.max_size = state_dict['max_size']
        self.buffer = deque(state_dict['buffer'], maxlen=self.max_size)

def train_step(network: AlphaZeroNet, optimizer: optim.Optimizer,
               states: np.ndarray, target_policies: np.ndarray,
               target_values: np.ndarray, device: torch.device) -> dict:
    network.train()

    states = torch.from_numpy(states).to(device)
    target_policies = torch.from_numpy(target_policies).to(device)
    target_values = torch.from_numpy(target_values).float().to(device)

    optimizer.zero_grad()

    policy_logits, values = network(states)
    values = values.squeeze(-1)

    # Policy loss (cross-entropy)
    policy_loss = -torch.sum(target_policies * F.log_softmax(policy_logits, dim=1)) / states.size(0)

    # Value loss (MSE)
    value_loss = F.mse_loss(values, target_values)

    # Total loss
    loss = policy_loss + value_loss

    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item()
    }

def train_epoch_with_dataloader(network: AlphaZeroNet, optimizer: optim.Optimizer,
                                replay_buffer: 'ReplayBuffer', cfg: Config,
                                device: torch.device,
                                scaler: Optional[torch.amp.GradScaler] = None) -> dict:
    """Train for one epoch using DataLoader with device-specific optimizations."""
    network.train()

    # Sample data
    states, policies, values = replay_buffer.sample(min(len(replay_buffer), cfg.batch_size * 20))

    # Create dataset and dataloader
    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(policies),
        torch.from_numpy(values).float()
    )

    # Device-specific DataLoader settings
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=(use_cuda and cfg.num_workers > 0)
    )

    # AMP is only supported on CUDA
    use_amp = use_cuda and scaler is not None

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for batch_states, batch_policies, batch_values in dataloader:
        batch_states = batch_states.to(device, non_blocking=use_cuda)
        batch_policies = batch_policies.to(device, non_blocking=use_cuda)
        batch_values = batch_values.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)

        # Use AMP autocast for CUDA
        with torch.amp.autocast('cuda', enabled=use_amp):
            policy_logits, pred_values = network(batch_states)
            pred_values = pred_values.squeeze(-1)

            # Policy loss (cross-entropy)
            policy_loss = -torch.sum(batch_policies * F.log_softmax(policy_logits, dim=1)) / batch_states.size(0)

            # Value loss (MSE)
            value_loss = F.mse_loss(pred_values, batch_values)

            # Total loss
            loss = policy_loss + value_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }

# ============================================================================
# Main Training Loop
# ============================================================================

def train_alphazero(cfg: Optional[Config] = None, resume_from: Optional[str] = None):
    """
    Train AlphaZero Chess model.

    Args:
        cfg: Configuration object with hyperparameters. If None, uses device-optimized defaults.
        resume_from: Path to checkpoint file to resume training from.
                    Supports device switching (GPU->CPU, CPU->GPU, MPS->CPU, etc.).

    The function automatically detects available device (CUDA/MPS/CPU) and handles
    device migration when resuming from checkpoints trained on different devices.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Use device-optimized config if none provided
    if cfg is None:
        cfg = get_config_for_device(device)
        print(f"Using device-optimized config for {device.type}")

    # Initialize AMP scaler for CUDA (provides ~2x speedup with mixed precision)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("Automatic Mixed Precision (AMP) enabled")

    # Initialize network
    network = AlphaZeroNet(cfg).to(device)
    optimizer = optim.SGD(
        network.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

    # Initialize MCTS and replay buffer
    mcts = MCTS(cfg, network, device)
    replay_buffer = ReplayBuffer(cfg.buffer_size)

    iteration = 0

    # Resume from checkpoint if specified
    if resume_from:
        print(f"Loading checkpoint from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state and move to correct device
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Fix optimizer state device (important for GPU<->CPU switches)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        iteration = checkpoint['iteration']

        # Load replay buffer if available
        if 'replay_buffer' in checkpoint:
            replay_buffer.load_state_dict(checkpoint['replay_buffer'])
            print(f"Loaded replay buffer with {len(replay_buffer)} samples")

        print(f"Resumed from iteration {iteration} on {device}")
    while True:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")

        # Self-play phase - use parallel execution
        # Parallel self-play works for CPU and MPS (uses CPU workers with model copy)
        use_parallel = device.type in ('cpu', 'mps') and cfg.num_parallel_games > 1
        print(f"Self-play ({'parallel' if use_parallel else 'sequential'} on {cfg.num_parallel_games} workers)...")
        network.eval()

        # More games per iteration on GPU due to faster inference
        games_per_iteration = 3 if device.type == 'cpu' else (5 if device.type == 'mps' else 10)

        if use_parallel:
            samples = parallel_self_play(network, cfg, games_per_iteration)
            replay_buffer.add(samples)
            print(f"  Total samples collected: {len(samples)}")
        else:
            # Sequential self-play (CUDA uses GPU directly)
            for game_num in range(games_per_iteration):
                samples = self_play_game(mcts, cfg)
                replay_buffer.add(samples)
                print(f"  Game {game_num + 1}: {len(samples)} moves")

        # Training phase - use DataLoader for efficient batching
        if len(replay_buffer) >= cfg.min_buffer_size:
            print("Training...")
            for epoch in range(cfg.num_epochs):
                metrics = train_epoch_with_dataloader(network, optimizer, replay_buffer, cfg, device, scaler)
                print(f"  Epoch {epoch + 1}: loss={metrics['loss']:.4f} "
                      f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f})")
        else:
            print(f"Buffer size: {len(replay_buffer)}/{cfg.min_buffer_size}")

        # Save checkpoint periodically
        if iteration % 10 == 0:
            checkpoint_path = f'chess_{iteration}.pt'
            torch.save({
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'replay_buffer': replay_buffer.state_dict(),
                'config': cfg,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            print(f"  - Iteration: {iteration}")
            print(f"  - Replay buffer size: {len(replay_buffer)}")

# ============================================================================
# Play Against the Model
# ============================================================================

def play_against_ai(model_path: Optional[str] = None, cfg: Config = Config()):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CPU-specific optimizations
    if device.type == 'cpu':
        torch.set_num_threads(4)
        print("CPU mode: Using reduced simulations for faster play")

    network = AlphaZeroNet(cfg).to(device)
    if model_path:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    mcts = MCTS(cfg, network, device)
    board = chess.Board()
    
    print("You are White. Enter moves in UCI format (e.g., e2e4)")
    print(board)
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Human move
            while True:
                move_str = input("Your move: ").strip()
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        break
                    print("Illegal move!")
                except:
                    print("Invalid format!")
        else:
            # AI move
            print("AI thinking...")
            policy = mcts.search(board, add_noise=False)
            
            # Select best move
            legal_moves = list(board.legal_moves)
            move_probs = [(m, policy[encode_move(m)]) for m in legal_moves]
            move_probs.sort(key=lambda x: x[1], reverse=True)
            move = move_probs[0][0]
            print(f"AI plays: {move.uci()}")
        
        board.push(move)
        print(board)
        print()
    
    print(f"Game over: {board.result()}")

if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    mp.set_start_method('spawn', force=True)

    import sys
    import argparse

    parser = argparse.ArgumentParser(description='AlphaZero Chess Training and Playing')
    parser.add_argument('mode', choices=['train', 'play'], nargs='?', default='train',
                        help='Mode: train or play (default: train)')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Resume training from checkpoint (e.g., chess_10.pt)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Model path for playing (e.g., chess_100.pt)')

    # Handle legacy command-line format for backward compatibility
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
        play_against_ai(model_path)
    else:
        args = parser.parse_args()

        if args.mode == 'play':
            play_against_ai(args.model)
        else:  # train
            train_alphazero(resume_from=args.resume)