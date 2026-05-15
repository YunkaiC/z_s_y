"""
争上游卡牌游戏 - AI出牌决策模块

AI策略拆分为独立函数，便于后续优化：
1. 主动出牌策略 (ai_play_free)
2. 被动跟牌策略 (ai_play_follow)
3. 各种辅助决策函数
4. Hard AI: 启发式评分 + Monte Carlo Simulation（随机推测隐藏手牌后的蒙特卡洛模拟搜索）

设计原则：
- 被动跟牌：优先选最小能压住的牌，尽量不拆炸弹
- 主动出牌：优先出组合牌减少手牌，避免无意义拆散大组合
- 手牌少时提高进攻性

难度级别：
- easy: 30%随机出牌，极少使用炸弹，无终局优化
- normal: 标准策略，合理使用炸弹
- hard: 启发式评分 + 轻量级 Determinization Monte Carlo Simulation
  - 枚举候选出牌 → 启发式评分筛选 → 随机分配隐藏手牌 → 多次模拟 → 选最优
  - 不引入深度学习或模型训练
  - 模拟过程复用现有游戏规则和基础AI行为
  - 当信息不足时回退到启发式规则
"""
import random
import time
from collections import Counter
from card import Card
from game_logic import (
    Pattern, HandType, recognize_pattern, compare_hands,
    find_all_valid_plays, _count_ranks, find_all_bombs,
    _get_bomb_ranks, _effective_bomb_tier, BOMB_CONFIG
)
from constants import Rank, Suit


# 难度对应的进攻性系数
DIFFICULTY_AGGRESSIVENESS = {
    'easy': 0.3,
    'normal': 0.5,
    'hard': 0.8,
}


# ==================== Monte Carlo 配置参数 ====================
MC_CONFIG = {
    # 每个候选出牌的模拟次数（越大越准确但越慢）
    'num_simulations': 20,
    # 启发式筛选后最多模拟的候选数
    'max_candidates': 5,
    # 每次模拟最大步数（防止无限循环）
    'max_sim_steps': 40,
    # 手牌少于此数时跳过MC，直接用启发式（手牌太少时MC无意义）
    'min_hand_size_for_mc': 4,
    # MC总超时毫秒数（超过此时间强制返回当前最佳结果）
    'timeout_ms': 3000,
    # 跟牌时是否将"不出"作为候选
    'include_pass_as_candidate': True,
}


class AIEngine:
    """
    AI出牌决策引擎

    所有决策函数均为独立函数，便于单独测试和优化

    支持三种难度：
    - easy: 30%随机出牌，极少使用炸弹，无终局优化
    - normal: 标准策略，合理使用炸弹和终局优化
    - hard: 启发式评分 + 轻量级 Monte Carlo Simulation
      - 核心流程：枚举候选 → 启发式评分筛选 → 随机分配隐藏手牌 →
        对每个候选多次模拟后续对局 → 依据胜率/名次/清牌速度选择最终出牌
      - 当 game_state 信息不足时，自动回退到启发式规则
    """

    def __init__(self, aggressiveness: float = None, difficulty: str = 'normal'):
        """
        Args:
            aggressiveness: 进攻性系数 0.0-1.0，值越大越激进（已弃用，用difficulty代替）
            difficulty: 难度级别 'easy'/'normal'/'hard'
        """
        self.difficulty = difficulty
        if aggressiveness is not None:
            self.aggressiveness = aggressiveness
        else:
            self.aggressiveness = DIFFICULTY_AGGRESSIVENESS.get(difficulty, 0.5)

    def decide(self, cards: list[Card], last_play: Pattern | None,
               is_free_turn: bool, game_state: dict = None) -> tuple[list[Card], bool]:
        """
        AI决策入口

        Args:
            cards: 当前手牌
            last_play: 上一次出牌的牌型
            is_free_turn: 是否为自由出牌轮
            game_state: 游戏状态信息（用于更高级的决策）

        Returns:
            (出牌列表, 是否选择不要)
            如果选择不要，返回 ([], True)
        """
        if not cards:
            return [], not is_free_turn

        # Easy模式：30%概率随机出牌
        if self.difficulty == 'easy' and random.random() < 0.3:
            play = self._random_valid_play(cards, last_play, is_free_turn)
            if play is not None:
                return play, False
            # 随机出牌失败，回退到正常逻辑

        # 终局优化：如果手牌<=3张，检查是否能一次性出完
        # Easy模式跳过终局优化
        if self.difficulty != 'easy' and len(cards) <= 3:
            all_cards_play = self._can_play_all_remaining(cards, last_play, is_free_turn)
            if all_cards_play is not None:
                return all_cards_play, False

        # Hard模式：尝试 Monte Carlo 模拟决策
        if self.difficulty == 'hard':
            mc_result = self._mc_decide(cards, last_play, is_free_turn, game_state)
            if mc_result is not None:
                return mc_result

        # 默认逻辑（easy/normal/hard回退）
        if is_free_turn or last_play is None:
            play_cards = self.ai_play_free(cards, game_state)
            return play_cards, False
        else:
            return self.ai_play_follow(cards, last_play, game_state)

    # ==================== 主动出牌策略 ====================

    def ai_play_free(self, cards: list[Card], game_state: dict = None) -> list[Card]:
        """
        主动出牌策略（自由出牌）

        策略优先级：
        1. 如果手牌只剩1张或1对，直接出
        2. 终局检查：手牌<=6时检查是否能一次出完
        3. 优先出顺子、连对、飞机等组合牌（减少手牌数量最快）
        4. 出三带一/三带二（比纯三张更有效率）
        5. 出单张或对子（从小到大，避开炸弹点数）
        6. 保留炸弹作为后手

        Args:
            cards: 手牌
            game_state: 游戏状态

        Returns:
            要出的牌列表
        """
        if not cards:
            return []

        remaining = len(cards)

        # 如果只剩1张，直接出
        if remaining == 1:
            return cards[:]

        # 如果只剩2张且是合法牌型，直接出
        if remaining == 2:
            pattern = recognize_pattern(cards)
            if pattern.hand_type != HandType.INVALID:
                return cards[:]
            # 否则出最小的单张
            return [min(cards, key=lambda c: c.sort_key)]

        # 终局优化：检查是否能一次性出完
        if self.difficulty != 'easy' and remaining <= 6:
            all_play = self._can_play_all_remaining(cards, None, True)
            if all_play is not None:
                return all_play

        # 获取进攻性调整（手牌越少越激进）
        aggr = self._adjusted_aggressiveness(remaining)

        # 分析手牌结构
        rank_count = _count_ranks(cards)
        bomb_ranks = _get_bomb_ranks(rank_count)
        all_plays = find_all_valid_plays(cards, None)

        # 分类可用牌型
        combos = []      # 组合牌（顺子、连对、飞机等）
        triples = []     # 三张
        triple_ones = []  # 三带一
        triple_twos = []  # 三带二
        pairs = []       # 对子
        singles = []     # 单张
        bombs = []       # 炸弹

        for play in all_plays:
            if play.hand_type in (HandType.STRAIGHT, HandType.STRAIGHT_PAIR,
                                  HandType.TWO_STRAIGHT_PAIR,
                                  HandType.STRAIGHT_TRIPLE, HandType.PLANE_ONE,
                                  HandType.PLANE_TWO):
                combos.append(play)
            elif play.hand_type == HandType.TRIPLE:
                triples.append(play)
            elif play.hand_type == HandType.TRIPLE_ONE:
                triple_ones.append(play)
            elif play.hand_type == HandType.TRIPLE_TWO:
                triple_twos.append(play)
            elif play.hand_type == HandType.PAIR:
                pairs.append(play)
            elif play.hand_type == HandType.SINGLE:
                singles.append(play)
            elif play.hand_type in (HandType.BOMB, HandType.ROCKET):
                bombs.append(play)

        # Hard模式：更早使用炸弹
        bomb_threshold_free = 8 if self.difficulty == 'hard' else 5

        # 策略1：如果手牌较少且有炸弹，考虑使用炸弹快速清场
        if remaining <= bomb_threshold_free and bombs:
            non_bomb_cards = self._cards_without_bombs(cards, bombs)
            if not non_bomb_cards:
                # 只有炸弹了，出炸弹
                return self._select_smallest_bomb(bombs)
            # 如果炸弹出完后剩余牌能一次出完，先出炸弹
            bomb_cards = self._select_smallest_bomb(bombs)
            if bomb_cards:
                remaining_after_bomb = self._remove_cards(cards, bomb_cards)
                if not remaining_after_bomb or self._can_play_all_remaining(
                        remaining_after_bomb, None, True):
                    return bomb_cards

        # 策略1b：手牌较少且激进，考虑炸弹清场
        if remaining <= 8 and aggr >= 0.7 and bombs:
            non_bomb_cards = self._cards_without_bombs(cards, bombs)
            if not non_bomb_cards:
                return self._select_smallest_bomb(bombs)

        # 策略2：优先出组合牌（顺子、连对等）
        if combos:
            best_combo = self._select_best_combo(combos, aggr, rank_count)
            if best_combo:
                return best_combo.cards

        # 策略3：出三带一/三带二（比纯三张更高效）
        if triple_twos:
            triple_twos.sort(key=lambda p: p.main_rank)
            return triple_twos[0].cards

        if triple_ones:
            triple_ones.sort(key=lambda p: p.main_rank)
            return triple_ones[0].cards

        # 策略4：出三张（如果不需要带牌）
        if triples:
            triples.sort(key=lambda p: p.main_rank)
            return triples[0].cards

        # 策略5：出对子（从小到大，避开炸弹点数）
        if pairs:
            pairs.sort(key=lambda p: p.main_rank)
            return pairs[0].cards

        # 策略6：出单张（从小到大，避开炸弹点数）
        if singles:
            singles.sort(key=lambda p: p.main_rank)
            return singles[0].cards

        # 策略7：出炸弹
        if bombs:
            return self._select_smallest_bomb(bombs)

        # 兜底：出最小的牌
        cards_sorted = sorted(cards, key=lambda c: c.sort_key)
        return [cards_sorted[0]]

    # ==================== 被动跟牌策略 ====================

    def ai_play_follow(self, cards: list[Card], last_play: Pattern,
                       game_state: dict = None) -> tuple[list[Card], bool]:
        """
        被动跟牌策略

        策略：
        1. 优先选择能压住的最小合法牌
        2. 尽量不拆炸弹
        3. 手牌少时提高进攻性
        4. 没有合适牌时选择不要

        Args:
            cards: 手牌
            last_play: 上一次出牌的牌型
            game_state: 游戏状态

        Returns:
            (出牌列表, 是否选择不要)
        """
        remaining = len(cards)
        aggr = self._adjusted_aggressiveness(remaining)

        # 找到所有能压住上家的合法出牌
        valid_plays = find_all_valid_plays(cards, last_play)

        if not valid_plays:
            # 无法压住，选择不要
            return [], True

        # 分类：普通牌型 vs 炸弹
        normal_plays = [p for p in valid_plays
                        if p.hand_type not in (HandType.BOMB, HandType.ROCKET)]
        bomb_plays = [p for p in valid_plays
                      if p.hand_type in (HandType.BOMB, HandType.ROCKET)]

        # 如果有普通牌型可以压
        if normal_plays:
            # 选择最小的能压住的牌（按主牌点数排序，点数相同按消耗牌数多优先）
            normal_plays.sort(key=lambda p: (p.main_rank, -len(p.cards)))

            # 尽量不拆炸弹
            non_bomb_breaking = self._filter_non_bomb_breaking(normal_plays, cards)

            chosen_cards = None
            if non_bomb_breaking:
                chosen_cards = non_bomb_breaking[0].cards
            elif aggr > 0.6:
                chosen_cards = normal_plays[0].cards
            elif game_state and self._should_use_bomb(game_state, remaining):
                if bomb_plays:
                    chosen_cards = self._select_smallest_bomb(bomb_plays)
                else:
                    chosen_cards = normal_plays[0].cards
            else:
                chosen_cards = normal_plays[0].cards

            # 安全验证
            if chosen_cards:
                validated = self._validate_play(chosen_cards, last_play)
                if validated:
                    return chosen_cards, False
                for play in normal_plays:
                    if self._validate_play(play.cards, last_play):
                        return play.cards, False
                if bomb_plays and self._should_use_bomb(game_state, remaining):
                    for bplay in bomb_plays:
                        if self._validate_play(bplay.cards, last_play):
                            return bplay.cards, False
                return [], True
            return [], True

        # 只有炸弹可以压
        if bomb_plays:
            use_bomb = False

            if self.difficulty == 'hard':
                use_bomb = True
            else:
                opponent_min = 999
                if game_state:
                    opponent_min = game_state.get('opponent_min_cards', 999)
                if opponent_min <= 2:
                    use_bomb = True
                if remaining <= 5:
                    use_bomb = True
                if aggr > 0.5 or self._should_use_bomb(game_state, remaining):
                    use_bomb = True
                if self.difficulty == 'easy' and use_bomb:
                    use_bomb = random.random() < 0.1

            if use_bomb:
                chosen_cards = self._select_smallest_winning_bomb(bomb_plays, last_play)
                if chosen_cards:
                    validated = self._validate_play(chosen_cards, last_play)
                    if validated:
                        return chosen_cards, False
                for bplay in bomb_plays:
                    if self._validate_play(bplay.cards, last_play):
                        return bplay.cards, False
                return [], True

            return [], True

        return [], True

    # ==================== 辅助决策函数 ====================

    def _random_valid_play(self, cards: list[Card], last_play: Pattern | None,
                           is_free_turn: bool) -> list[Card] | None:
        """随机选择一个合法出牌（Easy模式用）"""
        target = last_play if not is_free_turn else None
        valid_plays = find_all_valid_plays(cards, target)
        if not valid_plays:
            return None
        non_bomb_plays = [p for p in valid_plays
                          if p.hand_type not in (HandType.BOMB, HandType.ROCKET)]
        if non_bomb_plays:
            return random.choice(non_bomb_plays).cards
        if random.random() < 0.05:
            return random.choice(valid_plays).cards
        return None

    def _validate_play(self, cards: list[Card], last_play: Pattern) -> bool:
        """验证出牌是否合法且能压过上家"""
        if not cards:
            return False
        pattern = recognize_pattern(cards)
        if pattern.hand_type == HandType.INVALID:
            return False
        if last_play is not None:
            result = compare_hands(pattern, last_play)
            if result <= 0:
                return False
        return True

    def _adjusted_aggressiveness(self, remaining_cards: int) -> float:
        """根据剩余手牌数量调整进攻性"""
        if remaining_cards <= 3:
            return min(1.0, self.aggressiveness + 0.4)
        elif remaining_cards <= 6:
            return min(1.0, self.aggressiveness + 0.2)
        elif remaining_cards <= 10:
            return self.aggressiveness
        else:
            return max(0.0, self.aggressiveness - 0.1)

    def _select_best_combo(self, combos: list[Pattern],
                           aggr: float, rank_count: Counter = None) -> Pattern | None:
        """选择最佳组合牌出牌"""
        if not combos:
            return None
        combos.sort(key=lambda p: -len(p.cards))
        if aggr > 0.6:
            return combos[0]
        else:
            return combos[-1]

    def _select_smallest_bomb(self, bombs: list[Pattern]) -> list[Card]:
        """选择最小的炸弹"""
        if not bombs:
            return []
        normal_bombs = [b for b in bombs if b.hand_type == HandType.BOMB]
        rockets = [b for b in bombs if b.hand_type == HandType.ROCKET]
        if normal_bombs:
            normal_bombs.sort(key=lambda b: (b.bomb_size, b.main_rank))
            return normal_bombs[0].cards
        if rockets:
            single_rockets = [r for r in rockets if not r.is_double_rocket]
            double_rockets = [r for r in rockets if r.is_double_rocket]
            if single_rockets:
                return single_rockets[0].cards
            return double_rockets[0].cards
        return bombs[0].cards

    def _select_smallest_winning_bomb(self, bombs: list[Pattern],
                                       last_play: Pattern) -> list[Card]:
        """选择最小的能赢的炸弹"""
        if not bombs:
            return []
        winning_bombs = []
        for b in bombs:
            if compare_hands(b, last_play) > 0:
                winning_bombs.append(b)
        if not winning_bombs:
            return []
        normal_winning = [b for b in winning_bombs if b.hand_type == HandType.BOMB]
        rocket_winning = [b for b in winning_bombs if b.hand_type == HandType.ROCKET]
        if normal_winning:
            normal_winning.sort(key=lambda b: (b.bomb_size, b.main_rank))
            return normal_winning[0].cards
        if rocket_winning:
            single_rockets = [r for r in rocket_winning if not r.is_double_rocket]
            double_rockets = [r for r in rocket_winning if r.is_double_rocket]
            if single_rockets:
                return single_rockets[0].cards
            return double_rockets[0].cards
        return []

    def _filter_non_bomb_breaking(self, plays: list[Pattern],
                                  cards: list[Card]) -> list[Pattern]:
        """过滤掉会拆炸弹的出牌"""
        rank_count = _count_ranks(cards)
        bomb_ranks = _get_bomb_ranks(rank_count)
        if not bomb_ranks:
            return plays
        safe_plays = []
        for play in plays:
            if play.hand_type in (HandType.BOMB, HandType.ROCKET):
                safe_plays.append(play)
                continue
            play_rank_count = _count_ranks(play.cards)
            breaks_bomb = False
            for rank in play_rank_count:
                if rank in bomb_ranks:
                    breaks_bomb = True
                    break
            if not breaks_bomb:
                safe_plays.append(play)
        return safe_plays

    def _should_use_bomb(self, game_state: dict | None,
                         my_remaining: int) -> bool:
        """判断是否应该使用炸弹"""
        if game_state is None:
            return my_remaining <= 5
        opponent_cards = game_state.get('opponent_min_cards', 999)
        if opponent_cards <= 2:
            return True
        if opponent_cards <= 3:
            return True
        if my_remaining <= 4:
            return True
        return False

    def _can_play_all_remaining(self, cards: list[Card],
                                 last_play: Pattern | None = None,
                                 is_free_turn: bool = True) -> list[Card] | None:
        """检查是否可以一次性出完所有手牌"""
        if not cards:
            return None
        pattern = recognize_pattern(cards)
        if pattern.hand_type == HandType.INVALID:
            return None
        if last_play is not None and not is_free_turn:
            if compare_hands(pattern, last_play) <= 0:
                return None
        return cards[:]

    def _cards_without_bombs(self, cards: list[Card],
                              bombs: list[Pattern]) -> list[Card]:
        """获取不包含炸弹牌的手牌"""
        bomb_keys = set()
        for bomb in bombs:
            for c in bomb.cards:
                bomb_keys.add((c.suit, c.rank, c.deck_id))
        return [c for c in cards if (c.suit, c.rank, c.deck_id) not in bomb_keys]

    def _remove_cards(self, cards: list[Card], to_remove: list[Card]) -> list[Card]:
        """从手牌中移除指定牌"""
        remove_keys = set((c.suit, c.rank, c.deck_id) for c in to_remove)
        return [c for c in cards if (c.suit, c.rank, c.deck_id) not in remove_keys]

    # ====================================================================
    # Hard AI: 启发式评分 + 轻量级 Determinization Monte Carlo Simulation
    # ====================================================================

    def _mc_decide(self, cards: list[Card], last_play: Pattern | None,
                   is_free_turn: bool, game_state: dict | None) -> tuple[list[Card], bool] | None:
        """
        Monte Carlo 模拟决策入口（仅 hard 难度使用）

        核心流程：
        1. 枚举当前所有合法候选出牌
        2. 使用启发式规则进行初步评分和筛选
        3. 随机分配未知手牌以构造多种可能局面
        4. 对每个候选出牌进行多次后续对局模拟
        5. 依据胜率、平均名次、清牌速度选择最终出牌

        Args:
            cards: 当前手牌
            last_play: 上一次出牌的牌型
            is_free_turn: 是否自由出牌
            game_state: 游戏状态信息

        Returns:
            (出牌列表, 是否不要) 或 None（回退到默认逻辑）
        """
        # 手牌太少时MC无意义，直接用启发式
        if len(cards) < MC_CONFIG['min_hand_size_for_mc']:
            return None

        # 1. 枚举候选出牌
        target = None if is_free_turn else last_play
        candidates = find_all_valid_plays(cards, target)

        if not candidates:
            if is_free_turn:
                # 自由出牌必须出牌，兜底
                sorted_cards = sorted(cards, key=lambda c: c.sort_key)
                return [sorted_cards[0]], False
            return [], True

        # 2. 启发式评分和筛选
        scored_candidates = self._score_and_filter_candidates(
            candidates, cards, last_play, is_free_turn, game_state
        )

        # 如果只有一个候选，直接返回
        if len(scored_candidates) == 1:
            best = scored_candidates[0]
            if best is None:
                # "不出"候选
                return [], True
            return best.cards, False

        # 3. 检查是否有足够信息进行MC模拟
        if not self._can_run_mc(game_state, cards):
            # 回退到启发式：返回评分最高的候选
            best = scored_candidates[0]
            if best is None:
                return [], True
            return best.cards, False

        # 4. 运行 Monte Carlo 模拟
        mc_result = self._run_mc_simulations(
            scored_candidates, cards, last_play, is_free_turn, game_state
        )

        if mc_result is not None:
            return mc_result

        # MC失败，回退到启发式
        best = scored_candidates[0]
        if best is None:
            return [], True
        return best.cards, False

    # -------------------- 候选牌评分 --------------------

    def _heuristic_score(self, candidate: Pattern | None, my_cards: list[Card],
                         last_play: Pattern | None, is_free_turn: bool,
                         game_state: dict | None) -> float:
        """
        启发式评分函数

        评估因素：
        1. 出牌张数（越多越好，减少手牌）
        2. 是否拆对子/三张/顺子/连对等良好结构
        3. 是否拆炸弹
        4. 出牌后孤张数量
        5. 自身剩余牌数
        6. 对手剩余牌数
        7. 主动出牌 vs 跟牌
        8. 对手是否即将走完
        9. 自己是否接近走完
        10. 炸弹是否被无意义浪费

        Args:
            candidate: 候选出牌（None表示"不出"）
            my_cards: 当前手牌
            last_play: 上一次出牌
            is_free_turn: 是否自由出牌
            game_state: 游戏状态

        Returns:
            评分（越高越好）
        """
        # "不出"候选的评分
        if candidate is None:
            score = 0.0
            # 跟牌时不出通常是保守选择
            if not is_free_turn and last_play is not None:
                score += 1.0  # 基础分：不出是安全的
                # 但如果对手快走完了，不出是危险的
                if game_state:
                    opp_min = game_state.get('opponent_min_cards', 999)
                    if opp_min <= 2:
                        score -= 15.0  # 对手快赢了，不出很危险
                    elif opp_min <= 4:
                        score -= 5.0
                # 如果手牌很多，不出可以等更好的机会
                if len(my_cards) > 10:
                    score += 2.0
            return score

        score = 0.0
        remaining_after = self._remove_cards(my_cards, candidate.cards)
        remaining_count = len(remaining_after)

        # ---- 因素1: 出牌张数（越多越好）----
        score += len(candidate.cards) * 1.0

        # ---- 因素2: 能否一次出完（极大加分）----
        if remaining_count == 0:
            score += 200.0  # 直接赢
        elif remaining_count <= 3:
            all_play = self._can_play_all_remaining(remaining_after, None, True)
            if all_play is not None:
                score += 80.0  # 下一轮就能赢

        # ---- 因素3: 组合牌加分（顺子/连对/飞机等减少手牌效率高）----
        if candidate.hand_type in (HandType.STRAIGHT, HandType.STRAIGHT_PAIR,
                                   HandType.TWO_STRAIGHT_PAIR,
                                   HandType.STRAIGHT_TRIPLE,
                                   HandType.PLANE_ONE, HandType.PLANE_TWO):
            score += len(candidate.cards) * 0.8  # 组合牌额外加分

        # ---- 因素4: 结构破坏惩罚 ----
        my_rc = _count_ranks(my_cards)
        play_rc = _count_ranks(candidate.cards)
        for rank, play_count in play_rc.items():
            hand_count = my_rc.get(rank, 0)
            if hand_count >= 4 and play_count < hand_count:
                # 从4+张中取部分，可能涉及拆炸弹（但find_all_valid_plays已过滤）
                # 这里作为额外惩罚
                if candidate.hand_type not in (HandType.BOMB, HandType.ROCKET):
                    score -= 3.0  # 严重惩罚
            elif hand_count == 3 and play_count == 1:
                # 从三张中取一张，破坏三张结构
                score -= 1.5
            elif hand_count == 2 and play_count == 1:
                # 从对子中取一张，破坏对子
                score -= 0.8
            elif hand_count == 3 and play_count == 2:
                # 从三张中取两张，破坏三张
                score -= 1.0

        # ---- 因素5: 出牌后孤张数量惩罚 ----
        if remaining_count > 0:
            after_rc = _count_ranks(remaining_after)
            after_bomb_ranks = _get_bomb_ranks(after_rc)
            singles = sum(1 for r, c in after_rc.items()
                         if c == 1 and r not in after_bomb_ranks)
            score -= singles * 0.4

        # ---- 因素6: 炸弹使用合理性 ----
        if candidate.hand_type in (HandType.BOMB, HandType.ROCKET):
            opp_min = 999
            if game_state:
                opp_min = game_state.get('opponent_min_cards', 999)

            if remaining_count == 0:
                score += 50.0  # 炸弹清场获胜
            elif opp_min <= 2:
                score += 25.0  # 阻止对手获胜
            elif opp_min <= 4:
                score += 10.0  # 对手快走完
            elif remaining_count <= 4:
                score += 8.0   # 自己快走完
            else:
                score -= 12.0  # 无意义浪费炸弹

        # ---- 因素7: 主动出牌策略 ----
        if is_free_turn or last_play is None:
            # 优先出小牌保留大牌
            score += (Rank.BIG_JOKER - candidate.main_rank) * 0.15
            # 三带一/三带二比纯三张好
            if candidate.hand_type in (HandType.TRIPLE_ONE, HandType.TRIPLE_TWO):
                score += 1.5
            # 纯三张不如带牌
            if candidate.hand_type == HandType.TRIPLE:
                score -= 0.5

        # ---- 因素8: 跟牌策略 ----
        else:
            # 用尽量小的牌压住
            score += (Rank.BIG_JOKER - candidate.main_rank) * 0.25
            # 对手快走完时，任何能出的牌都好
            opp_min = 999
            if game_state:
                opp_min = game_state.get('opponent_min_cards', 999)
            if opp_min <= 3:
                score += 8.0  # 紧急压制
            elif opp_min <= 5:
                score += 3.0

        # ---- 因素9: 自身紧迫度 ----
        if remaining_count <= 5:
            score += (5 - remaining_count) * 2.0  # 手牌越少越要积极
        if remaining_count <= 3:
            score += 5.0

        # ---- 因素10: 避免出大牌压小牌 ----
        if last_play is not None and not is_free_turn:
            rank_diff = candidate.main_rank - last_play.main_rank
            if rank_diff > 5 and candidate.hand_type not in (HandType.BOMB, HandType.ROCKET):
                score -= rank_diff * 0.3  # 用太大的牌压小牌不划算

        return score

    def _score_and_filter_candidates(self, candidates: list[Pattern],
                                     my_cards: list[Card],
                                     last_play: Pattern | None,
                                     is_free_turn: bool,
                                     game_state: dict | None) -> list[Pattern | None]:
        """
        启发式评分并筛选候选出牌

        返回按评分从高到低排列的候选列表（最多 max_candidates 个），
        跟牌时可能包含 None（表示"不出"）。

        Args:
            candidates: 所有合法候选出牌
            my_cards: 当前手牌
            last_play: 上一次出牌
            is_free_turn: 是否自由出牌
            game_state: 游戏状态

        Returns:
            排序后的候选列表（Pattern 或 None）
        """
        # 评分
        scored = []
        for cand in candidates:
            score = self._heuristic_score(cand, my_cards, last_play, is_free_turn, game_state)
            scored.append((cand, score))

        # 跟牌时添加"不出"候选
        if not is_free_turn and MC_CONFIG.get('include_pass_as_candidate', True):
            pass_score = self._heuristic_score(None, my_cards, last_play, is_free_turn, game_state)
            scored.append((None, pass_score))

        # 按评分降序排列
        scored.sort(key=lambda x: -x[1])

        # 保留 top N
        max_cands = MC_CONFIG['max_candidates']
        filtered = [cand for cand, _ in scored[:max_cands]]

        return filtered

    # -------------------- MC 可行性检查 --------------------

    def _can_run_mc(self, game_state: dict | None, cards: list[Card]) -> bool:
        """
        检查是否有足够信息运行 Monte Carlo 模拟

        必须信息：
        - game_state 不为 None
        - num_decks（牌副数）
        - opponent_card_counts（对手手牌数）

        Args:
            game_state: 游戏状态
            cards: 当前手牌

        Returns:
            是否可以运行MC
        """
        if game_state is None:
            return False
        if 'num_decks' not in game_state:
            return False
        if 'opponent_card_counts' not in game_state:
            return False
        opp_counts = game_state.get('opponent_card_counts', [])
        if not opp_counts:
            return False
        return True

    # -------------------- 隐藏手牌随机分配 --------------------

    def _allocate_hidden_hands(self, my_cards: list[Card],
                                game_state: dict) -> list[list[Card]]:
        """
        随机分配隐藏手牌（Determinization）

        基于已知信息（自己的手牌、已出牌、总牌池），随机分配未知手牌给对手。
        如果有已出牌信息，排除已出的牌；否则只排除自己的手牌。

        Args:
            my_cards: 自己的手牌
            game_state: 游戏状态，包含：
                - num_decks: 牌副数
                - opponent_card_counts: 对手手牌数列表 [(idx, count), ...]
                - played_cards: 已出牌列表（可选，提高准确性）

        Returns:
            对手手牌列表，每个元素是一个Card列表
        """
        num_decks = game_state.get('num_decks', 1)
        opp_counts = game_state.get('opponent_card_counts', [])
        played_cards_info = game_state.get('played_cards', None)

        # 构建完整牌池
        pool = []
        for deck_id in range(num_decks):
            for suit in [Suit.SPADE, Suit.HEART, Suit.DIAMOND, Suit.CLUB]:
                for rank in range(Rank.THREE, Rank.TWO + 1):
                    pool.append(Card(suit, Rank(rank), deck_id))
            pool.append(Card(Suit.JOKER, Rank.SMALL_JOKER, deck_id))
            pool.append(Card(Suit.JOKER, Rank.BIG_JOKER, deck_id))

        # 排除自己的手牌
        my_keys = set((c.suit, c.rank, c.deck_id) for c in my_cards)
        pool = [c for c in pool if (c.suit, c.rank, c.deck_id) not in my_keys]

        # 排除已出牌（如果有）
        if played_cards_info:
            played_keys = set((c.suit, c.rank, c.deck_id) for c in played_cards_info)
            pool = [c for c in pool if (c.suit, c.rank, c.deck_id) not in played_keys]

        # 洗牌
        random.shuffle(pool)

        # 分配给对手
        total_needed = sum(count for _, count in opp_counts)
        if len(pool) < total_needed:
            # 牌不够（可能因为缺少已出牌信息），调整分配
            # 按比例缩减
            if len(pool) == 0:
                return [[] for _ in opp_counts]
            scale = len(pool) / max(1, total_needed)
            adjusted_counts = [(idx, max(0, int(count * scale))) for idx, count in opp_counts]
        else:
            adjusted_counts = opp_counts

        hands = []
        idx = 0
        for opp_idx, count in adjusted_counts:
            hand = pool[idx:idx + count]
            hand.sort(key=lambda c: c.sort_key)
            hands.append(hand)
            idx += count

        return hands

    # -------------------- MC 模拟执行 --------------------

    def _run_mc_simulations(self, scored_candidates: list[Pattern | None],
                            my_cards: list[Card], last_play: Pattern | None,
                            is_free_turn: bool,
                            game_state: dict) -> tuple[list[Card], bool] | None:
        """
        对每个候选出牌运行 Monte Carlo 模拟

        对每个候选：
        1. 应用该候选出牌
        2. 随机分配隐藏手牌
        3. 进行多次完整对局模拟
        4. 收集模拟结果

        Args:
            scored_candidates: 经启发式筛选后的候选列表
            my_cards: 当前手牌
            last_play: 上一次出牌
            is_free_turn: 是否自由出牌
            game_state: 游戏状态

        Returns:
            (出牌列表, 是否不要) 或 None
        """
        start_time = time.time()
        timeout_sec = MC_CONFIG['timeout_ms'] / 1000.0
        num_sims = MC_CONFIG['num_simulations']
        num_players = 1 + len(game_state.get('opponent_card_counts', []))
        # 在模拟中，我的手牌始终在索引0，对手在1, 2, ...
        my_sim_idx = 0

        # 候选模拟结果：candidate -> list of scores
        candidate_scores = {}

        for cand in scored_candidates:
            candidate_scores[id(cand) if cand is not None else 0] = []

        for sim_idx in range(num_sims):
            # 检查超时
            if time.time() - start_time > timeout_sec:
                break

            # 随机分配隐藏手牌
            opp_hands = self._allocate_hidden_hands(my_cards, game_state)

            for cand in scored_candidates:
                # 再次检查超时
                if time.time() - start_time > timeout_sec:
                    break

                cand_key = id(cand) if cand is not None else 0

                if cand is None:
                    # "不出"候选：从下一个玩家开始模拟
                    sim_hands = [my_cards[:]] + [h[:] for h in opp_hands]
                    # 不出后，当前玩家变为下一个（在模拟中，我从索引0开始）
                    current_idx = 1 % num_players
                    # 跳过已完成的玩家
                    sim_active = [len(h) > 0 for h in sim_hands]
                    while current_idx != my_sim_idx and not sim_active[current_idx]:
                        current_idx = (current_idx + 1) % num_players
                    # 如果回到自己，说明其他人都不在了
                    if current_idx == my_sim_idx:
                        current_idx = (current_idx + 1) % num_players

                    sim_lp = last_play
                    sim_pass_count = 1  # 我们已经"不出"了一次
                    sim_is_free = False
                else:
                    # 出牌候选：从手牌中移除出的牌
                    my_remaining = self._remove_cards(my_cards, cand.cards)
                    sim_hands = [my_remaining] + [h[:] for h in opp_hands]

                    # 出牌后下一个玩家
                    current_idx = 1 % num_players
                    sim_active = [len(h) > 0 for h in sim_hands]
                    while current_idx != my_sim_idx and not sim_active[current_idx]:
                        current_idx = (current_idx + 1) % num_players
                    if current_idx == my_sim_idx:
                        current_idx = (current_idx + 1) % num_players

                    sim_lp = cand  # 上一次出牌变为候选出牌
                    sim_pass_count = 0
                    sim_is_free = False

                # 运行一次模拟
                sim_score = self._simulate_one_game(
                    sim_hands, my_sim_idx, current_idx,
                    sim_lp, sim_pass_count, sim_is_free, num_players
                )

                candidate_scores[cand_key].append(sim_score)

            # 如果已超时，跳出外层循环
            if time.time() - start_time > timeout_sec:
                break

        # 计算每个候选的平均得分
        avg_scores = {}
        for cand in scored_candidates:
            cand_key = id(cand) if cand is not None else 0
            scores = candidate_scores[cand_key]
            if scores:
                avg_scores[cand_key] = sum(scores) / len(scores)
            else:
                avg_scores[cand_key] = 0.0

        # 选择平均得分最高的候选
        best_cand = None
        best_avg = -float('inf')
        for cand in scored_candidates:
            cand_key = id(cand) if cand is not None else 0
            avg = avg_scores.get(cand_key, 0.0)
            if avg > best_avg:
                best_avg = avg
                best_cand = cand

        if best_cand is None:
            return [], True
        return best_cand.cards, False

    # -------------------- 单次对局模拟 --------------------

    def _simulate_one_game(self, all_hands: list[list[Card]],
                           my_idx: int, current_idx: int,
                           last_play: Pattern | None,
                           pass_count: int, is_free_turn: bool,
                           num_players: int) -> float:
        """
        模拟一局对局的后续发展

        使用简化策略（快速出牌选择）进行rollout，
        直到游戏结束或达到最大步数限制。

        Args:
            all_hands: 所有玩家的手牌（会被修改）
            my_idx: 自己的玩家索引
            current_idx: 当前出牌玩家索引
            last_play: 上一次出牌
            pass_count: 连续不出次数
            is_free_turn: 是否自由出牌
            num_players: 玩家总数

        Returns:
            模拟评分（0-100，越高越好）
        """
        # 深拷贝手牌，避免修改原始数据
        hands = [h[:] for h in all_hands]
        finished = [len(h) == 0 for h in hands]
        finish_order = []
        for i, f in enumerate(finished):
            if f:
                finish_order.append(i)

        lp = last_play
        lp_player = -1
        if last_play is not None:
            # 出牌者：在模拟中，如果是从"我出牌后"开始，出牌者就是my_idx
            # 安全检查：确保my_idx在有效范围内
            if 0 <= my_idx < len(finished):
                lp_player = my_idx

        cur = current_idx % num_players  # 安全取模
        p_count = pass_count
        free = is_free_turn
        max_steps = MC_CONFIG['max_sim_steps']

        for step in range(max_steps):
            # 检查游戏是否结束
            active_count = sum(1 for f in finished if not f)
            if active_count <= 1:
                break

            # 跳过已完成的玩家
            attempts = 0
            while finished[cur] and attempts < num_players:
                cur = (cur + 1) % num_players
                attempts += 1

            if attempts >= num_players:
                break

            hand = hands[cur]
            if not hand:
                cur = (cur + 1) % num_players
                continue

            # 检查自由出牌条件
            if lp is not None and 0 <= lp_player < len(finished):
                active_count = sum(1 for f in finished if not f)
                lp_active = not finished[lp_player]
                threshold = active_count - 1 if lp_active else active_count
                if p_count >= threshold:
                    free = True
                    lp = None
                    lp_player = -1
                    p_count = 0

            # 快速出牌决策
            play_cards, play_pattern, should_pass = self._fast_rollout_play(
                hand, lp, free
            )

            if should_pass:
                # 不出
                p_count += 1
            else:
                # 出牌
                play_keys = set((c.suit, c.rank, c.deck_id) for c in play_cards)
                hands[cur] = [c for c in hands[cur]
                              if (c.suit, c.rank, c.deck_id) not in play_keys]
                lp = play_pattern
                lp_player = cur
                p_count = 0
                free = False

                # 检查是否出完
                if not hands[cur] and not finished[cur]:
                    finished[cur] = True
                    finish_order.append(cur)

            # 下一个玩家
            cur = (cur + 1) % num_players

        # 评估结果
        return self._eval_sim_result(hands, finished, finish_order, my_idx)

    # -------------------- 快速出牌策略（rollout用）--------------------

    def _fast_rollout_play(self, cards: list[Card], last_play: Pattern | None,
                           is_free_turn: bool) -> tuple[list[Card], Pattern | None, bool]:
        """
        快速出牌策略，用于Monte Carlo模拟rollout

        比完整的 find_all_valid_plays + AI决策快得多，
        只寻找一个合理的出牌，不需要枚举所有可能。

        Args:
            cards: 当前手牌
            last_play: 上一次出牌
            is_free_turn: 是否自由出牌

        Returns:
            (出牌列表, 出牌Pattern, 是否不出)
        """
        if not cards:
            return [], None, True

        rank_count = _count_ranks(cards)
        bomb_ranks = _get_bomb_ranks(rank_count)

        if is_free_turn or last_play is None:
            # ---- 自由出牌：寻找最简单的出牌 ----
            # 优先顺序：单张 < 对子 < 三张 < 三带一 < 炸弹
            non_bomb_ranks = sorted([r for r in rank_count if r not in bomb_ranks])

            # 尝试出最小的单张
            for rank in non_bomb_ranks:
                if rank_count[rank] == 1:
                    c = [card for card in cards if card.rank == rank][0]
                    pattern = Pattern(HandType.SINGLE, rank, cards=[c])
                    return [c], pattern, False

            # 尝试出最小的对子
            for rank in non_bomb_ranks:
                if rank_count[rank] == 2:
                    cs = [card for card in cards if card.rank == rank][:2]
                    pattern = Pattern(HandType.PAIR, rank, cards=cs)
                    return cs, pattern, False

            # 尝试出三带一
            for rank in non_bomb_ranks:
                if rank_count[rank] == 3:
                    triple_cs = [card for card in cards if card.rank == rank][:3]
                    # 找一个最小的带牌
                    for kicker_rank in non_bomb_ranks:
                        if kicker_rank != rank:
                            kicker_c = [card for card in cards if card.rank == kicker_rank][0]
                            all_cs = triple_cs + [kicker_c]
                            pattern = Pattern(HandType.TRIPLE_ONE, rank, cards=all_cs)
                            return all_cs, pattern, False
                    # 没有带牌，出纯三张
                    pattern = Pattern(HandType.TRIPLE, rank, cards=triple_cs)
                    return triple_cs, pattern, False

            # 尝试出对子（从有多张的rank中取2张）
            for rank in non_bomb_ranks:
                if rank_count[rank] >= 2:
                    cs = [card for card in cards if card.rank == rank][:2]
                    pattern = Pattern(HandType.PAIR, rank, cards=cs)
                    return cs, pattern, False

            # 只剩炸弹，出最小的
            for rank in sorted(rank_count.keys()):
                if rank in bomb_ranks:
                    cs = [card for card in cards if card.rank == rank][:
                         BOMB_CONFIG['min_bomb_size']]
                    pattern = Pattern(HandType.BOMB, rank, cards=cs,
                                     bomb_size=len(cs))
                    return cs, pattern, False

            # 兜底：出最小单张（包括炸弹点数的）
            sorted_cards = sorted(cards, key=lambda c: c.sort_key)
            c = sorted_cards[0]
            pattern = Pattern(HandType.SINGLE, c.rank, cards=[c])
            return [c], pattern, False

        else:
            # ---- 跟牌：寻找最小能压住的牌 ----
            target_type = last_play.hand_type
            target_rank = last_play.main_rank

            if target_type == HandType.SINGLE:
                for rank in sorted(rank_count.keys()):
                    if rank in bomb_ranks:
                        continue
                    if rank > target_rank:
                        c = [card for card in cards if card.rank == rank][0]
                        pattern = Pattern(HandType.SINGLE, rank, cards=[c])
                        return [c], pattern, False

            elif target_type == HandType.PAIR:
                for rank in sorted(rank_count.keys()):
                    if rank in bomb_ranks:
                        continue
                    if rank > target_rank and rank_count[rank] >= 2:
                        cs = [card for card in cards if card.rank == rank][:2]
                        pattern = Pattern(HandType.PAIR, rank, cards=cs)
                        return cs, pattern, False

            elif target_type == HandType.TRIPLE:
                for rank in sorted(rank_count.keys()):
                    if rank in bomb_ranks:
                        continue
                    if rank > target_rank and rank_count[rank] >= 3:
                        cs = [card for card in cards if card.rank == rank][:3]
                        pattern = Pattern(HandType.TRIPLE, rank, cards=cs)
                        return cs, pattern, False

            elif target_type == HandType.TRIPLE_ONE:
                for rank in sorted(rank_count.keys()):
                    if rank in bomb_ranks:
                        continue
                    if rank > target_rank and rank_count[rank] >= 3:
                        triple_cs = [card for card in cards if card.rank == rank][:3]
                        for kicker_rank in sorted(rank_count.keys()):
                            if kicker_rank != rank and kicker_rank not in bomb_ranks:
                                kicker_c = [card for card in cards
                                           if card.rank == kicker_rank][0]
                                all_cs = triple_cs + [kicker_c]
                                pattern = Pattern(HandType.TRIPLE_ONE, rank,
                                                 cards=all_cs)
                                return all_cs, pattern, False

            elif target_type == HandType.TRIPLE_TWO:
                for rank in sorted(rank_count.keys()):
                    if rank in bomb_ranks:
                        continue
                    if rank > target_rank and rank_count[rank] >= 3:
                        triple_cs = [card for card in cards if card.rank == rank][:3]
                        for pair_rank in sorted(rank_count.keys()):
                            if pair_rank != rank and pair_rank not in bomb_ranks:
                                if rank_count[pair_rank] >= 2:
                                    pair_cs = [card for card in cards
                                              if card.rank == pair_rank][:2]
                                    all_cs = triple_cs + pair_cs
                                    pattern = Pattern(HandType.TRIPLE_TWO, rank,
                                                     cards=all_cs)
                                    return all_cs, pattern, False

            elif target_type == HandType.STRAIGHT:
                target_len = last_play.length
                # 简化：只尝试找同长度的顺子
                single_ranks = sorted([r for r in rank_count
                                      if r < Rank.TWO and r not in bomb_ranks])
                for start_idx in range(len(single_ranks) - target_len + 1):
                    seq = single_ranks[start_idx:start_idx + target_len]
                    if len(seq) == target_len and seq[-1] > target_rank:
                        is_consecutive = all(
                            seq[i + 1] == seq[i] + 1 for i in range(len(seq) - 1)
                        )
                        if is_consecutive:
                            cs = []
                            for r in seq:
                                cs.append([card for card in cards if card.rank == r][0])
                            pattern = Pattern(HandType.STRAIGHT, seq[-1],
                                             length=target_len, cards=cs)
                            return cs, pattern, False

            elif target_type in (HandType.STRAIGHT_PAIR, HandType.TWO_STRAIGHT_PAIR):
                target_len = last_play.length
                pair_ranks = sorted([r for r, c in rank_count.items()
                                    if c >= 2 and r < Rank.TWO and r not in bomb_ranks])
                for start_idx in range(len(pair_ranks) - target_len + 1):
                    seq = pair_ranks[start_idx:start_idx + target_len]
                    if len(seq) == target_len and seq[-1] > target_rank:
                        is_consecutive = all(
                            seq[i + 1] == seq[i] + 1 for i in range(len(seq) - 1)
                        )
                        if is_consecutive:
                            cs = []
                            for r in seq:
                                cs.extend([card for card in cards
                                          if card.rank == r][:2])
                            ht = (HandType.TWO_STRAIGHT_PAIR
                                  if target_len == 2 and target_type == HandType.TWO_STRAIGHT_PAIR
                                  else HandType.STRAIGHT_PAIR)
                            pattern = Pattern(ht, seq[-1],
                                             length=target_len, cards=cs)
                            return cs, pattern, False

            elif target_type in (HandType.BOMB, HandType.ROCKET):
                # 跟炸弹：找更大的炸弹
                last_tier = _effective_bomb_tier(last_play) if last_play.is_bomb_type else 0
                for rank in sorted(rank_count.keys()):
                    if rank in bomb_ranks and rank > target_rank:
                        cs = [card for card in cards if card.rank == rank][:
                             BOMB_CONFIG['min_bomb_size']]
                        bomb = Pattern(HandType.BOMB, rank, cards=cs,
                                      bomb_size=len(cs))
                        if _effective_bomb_tier(bomb) >= last_tier:
                            return cs, bomb, False

            # 任何非炸弹牌型都可以被炸弹压
            if target_type not in (HandType.BOMB, HandType.ROCKET):
                # 手牌较少时考虑用炸弹
                if len(cards) <= 8:
                    for rank in sorted(rank_count.keys()):
                        if rank in bomb_ranks:
                            cs = [card for card in cards if card.rank == rank][:
                                 BOMB_CONFIG['min_bomb_size']]
                            pattern = Pattern(HandType.BOMB, rank, cards=cs,
                                             bomb_size=len(cs))
                            return cs, pattern, False

                # 尝试王炸
                small_jokers = [c for c in cards if c.rank == Rank.SMALL_JOKER]
                big_jokers = [c for c in cards if c.rank == Rank.BIG_JOKER]
                if small_jokers and big_jokers:
                    cs = [small_jokers[0], big_jokers[0]]
                    pattern = Pattern(HandType.ROCKET, Rank.BIG_JOKER,
                                     cards=cs, is_rocket=True)
                    return cs, pattern, False

            # 无法压住，不出
            return [], None, True

    # -------------------- 模拟结果评估 --------------------

    def _eval_sim_result(self, hands: list[list[Card]],
                         finished: list[bool],
                         finish_order: list[int],
                         my_idx: int) -> float:
        """
        评估模拟结果

        评分规则：
        - 第1名：100分
        - 第2名：60分
        - 第3名：30分
        - 第4名：10分
        - 未完成：根据手牌数量与对手手牌数量的比较给出中间分

        Args:
            hands: 所有玩家手牌
            finished: 是否已出完
            finish_order: 出完顺序
            my_idx: 自己的索引

        Returns:
            评分（0-100）
        """
        if my_idx in finish_order:
            rank = finish_order.index(my_idx) + 1
            if rank == 1:
                return 100.0
            elif rank == 2:
                return 60.0
            elif rank == 3:
                return 30.0
            else:
                return 10.0

        # 未完成：根据手牌数量估计
        my_remaining = len(hands[my_idx])
        if my_remaining == 0:
            return 90.0  # 应该在finish_order中，但以防万一

        # 计算对手平均剩余牌数
        other_remainings = []
        for i, h in enumerate(hands):
            if i != my_idx and not finished[i]:
                other_remainings.append(len(h))

        if not other_remainings:
            # 其他人都出完了，我是最后一个
            total_players = len(hands)
            return max(5.0, 30.0 / total_players)

        others_avg = sum(other_remainings) / len(other_remainings)

        # 根据相对手牌优势评分
        if my_remaining < others_avg:
            # 我比对手平均少，有优势
            advantage = 1 - (my_remaining / max(1, others_avg))
            return 30.0 + 30.0 * advantage
        else:
            # 我比对手平均多，劣势
            disadvantage = my_remaining / max(1, others_avg)
            return max(5.0, 25.0 / disadvantage)
