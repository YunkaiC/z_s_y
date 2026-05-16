"""
争上游卡牌游戏 - 牌型识别、比较与合法性判断

核心逻辑模块，负责：
1. 识别一组牌的牌型
2. 比较两手牌的大小
3. 判断出牌是否合法
4. 炸弹比较的独立配置逻辑
5. 炸弹是否可拆分的配置规则
6. 合法出牌候选的完整枚举
"""
from collections import Counter
from itertools import combinations
from card import Card
from constants import (
    Rank, HandType,
    MIN_STRAIGHT_LEN, MIN_STRAIGHT_PAIR_LEN, MIN_STRAIGHT_TRIPLE_LEN,
    MIN_TWO_STRAIGHT_PAIR_LEN,
)


# ==================== 炸弹比较配置 ====================
# 可修改此配置来调整炸弹规则
BOMB_CONFIG = {
    # 是否允许把已成炸弹的同点数牌拆开用于单张、对子、三张、顺子等非炸弹牌型
    'allow_split_bombs': True,
    # 是否允许4张同点数炸弹
    'allow_4_bomb': True,
    # 是否允许5张及以上同点数炸弹（两副牌模式）
    'allow_n_bomb': True,
    # 王炸是否能压所有炸弹（True=王炸最大，False=按张数比较）
    'rocket_beats_all_bombs': True,
    # 炸弹最小张数
    'min_bomb_size': 4,
    # 是否允许双王炸（两副牌模式下两张小王+两张大王）
    'allow_double_rocket': True,
    # 双王炸的等级：
    #   0 = 与普通王炸同级
    #   1 = 高于普通王炸但低于5张炸弹（当rocket_beats_all_bombs=False时）
    #       或高于普通王炸（当rocket_beats_all_bombs=True时）
    #   2 = 最高级，任何牌都无法压过
    'double_rocket_level': 2,
}


# ==================== 牌型识别 ====================

class Pattern:
    """
    牌型描述类
    记录一组牌的牌型信息
    """

    def __init__(self, hand_type: HandType, main_rank: Rank,
                 length: int = 0, cards: list[Card] = None,
                 bomb_size: int = 0, is_rocket: bool = False,
                 is_double_rocket: bool = False):
        """
        Args:
            hand_type: 牌型类型
            main_rank: 主牌点数（用于比较大小）
            length: 牌型长度（顺子长度、连对数、连三张数等）
            cards: 组成该牌型的牌列表
            bomb_size: 炸弹张数（仅BOMB类型有效）
            is_rocket: 是否为王炸
            is_double_rocket: 是否为双王炸（两副牌）
        """
        self.hand_type = hand_type
        self.main_rank = main_rank
        self.length = length
        self.cards = cards or []
        self.bomb_size = bomb_size
        self.is_rocket = is_rocket
        self.is_double_rocket = is_double_rocket

    @property
    def is_bomb_type(self) -> bool:
        """是否为炸弹类牌型（含王炸）"""
        return self.hand_type in (HandType.BOMB, HandType.ROCKET)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'hand_type': int(self.hand_type),
            'main_rank': int(self.main_rank),
            'length': self.length,
            'cards': [c.to_dict() for c in self.cards],
            'bomb_size': self.bomb_size,
            'is_rocket': self.is_rocket,
            'is_double_rocket': self.is_double_rocket,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Pattern':
        """从字典反序列化"""
        return cls(
            hand_type=HandType(data['hand_type']),
            main_rank=Rank(data['main_rank']),
            length=data.get('length', 0),
            cards=[Card.from_dict(c) for c in data.get('cards', [])],
            bomb_size=data.get('bomb_size', 0),
            is_rocket=data.get('is_rocket', False),
            is_double_rocket=data.get('is_double_rocket', False),
        )


# ==================== 辅助函数 ====================

def _count_ranks(cards: list[Card]) -> Counter:
    """统计各点数出现的次数"""
    return Counter(c.rank for c in cards)


def _get_rank_cards(cards: list[Card], rank: Rank) -> list[Card]:
    """获取指定点数的所有牌"""
    return [c for c in cards if c.rank == rank]


def _find_consecutive(rank_list: list[Rank], min_len: int) -> list[list[Rank]]:
    """
    在有序点数列表中找出所有长度>=min_len的连续子序列
    注意：2和王不能出现在顺子等连续牌型中

    Args:
        rank_list: 已排序且去重的点数列表
        min_len: 最小连续长度

    Returns:
        所有满足条件的连续子序列列表
    """
    # 过滤掉2和王（它们不能参与顺子）
    valid = [r for r in rank_list if r < Rank.TWO]

    if not valid:
        return []

    results = []
    start = 0
    for i in range(1, len(valid)):
        if valid[i] != valid[i - 1] + 1:
            seq_len = i - start
            if seq_len >= min_len:
                for sub_start in range(start, i):
                    for sub_end in range(sub_start + min_len, i + 1):
                        results.append(valid[sub_start:sub_end])
            start = i

    # 处理最后一段
    seq_len = len(valid) - start
    if seq_len >= min_len:
        for sub_start in range(start, len(valid)):
            for sub_end in range(sub_start + min_len, len(valid) + 1):
                results.append(valid[sub_start:sub_end])

    return results


def _is_bomb_size_allowed(size: int) -> bool:
    """
    检查给定张数的炸弹是否被BOMB_CONFIG允许

    Args:
        size: 炸弹张数

    Returns:
        是否允许
    """
    if size < BOMB_CONFIG['min_bomb_size']:
        return False
    if size == 4:
        return BOMB_CONFIG.get('allow_4_bomb', True)
    if size >= 5:
        return BOMB_CONFIG.get('allow_n_bomb', True)
    return False


def _get_bomb_ranks(rank_count: Counter) -> set[Rank]:
    """
    获取所有"炸弹点数"——即拥有min_bomb_size张以上、且至少能组成一种合法炸弹的点数。
    当配置禁止拆炸弹时，候选生成会用这些点数过滤非炸弹牌型。

    Args:
        rank_count: 手牌点数统计

    Returns:
        炸弹点数集合
    """
    bomb_ranks = set()
    for rank, count in rank_count.items():
        if count < BOMB_CONFIG['min_bomb_size']:
            continue
        # 检查是否存在至少一种合法炸弹张数
        for size in range(BOMB_CONFIG['min_bomb_size'], count + 1):
            if _is_bomb_size_allowed(size):
                bomb_ranks.add(rank)
                break
    return bomb_ranks


def _effective_bomb_tier(pattern: Pattern) -> int:
    """
    获取炸弹/王炸的有效层级，用于统一比较。

    层级规则：
    - 普通N张炸弹：tier = N（4张=4, 5张=5, ...）
    - 普通王炸：tier = 100（当rocket_beats_all_bombs=True）或 4（False）
    - 双王炸：
      - level 0: 与普通王炸同级
      - level 1: 比普通王炸高一级
      - level 2: 999，绝对最高

    Args:
        pattern: 牌型

    Returns:
        有效层级数值，越大越强
    """
    if pattern.is_double_rocket:
        level = BOMB_CONFIG.get('double_rocket_level', 2)
        base = 100 if BOMB_CONFIG.get('rocket_beats_all_bombs', True) else 4
        if level == 0:
            return base  # 与普通王炸同级
        elif level == 1:
            return base + 1  # 高于普通王炸一级
        elif level >= 2:
            return 999  # 绝对最高
        return base

    if pattern.is_rocket:
        return 100 if BOMB_CONFIG.get('rocket_beats_all_bombs', True) else 4

    if pattern.hand_type == HandType.BOMB:
        return pattern.bomb_size

    return 0


# ==================== 牌型识别 ====================

def recognize_pattern(cards: list[Card]) -> Pattern:
    """
    识别一组牌的牌型

    这是核心函数，根据出牌的组成判断其属于哪种牌型。
    注意：此函数只做纯粹的牌型识别，不涉及炸弹拆分限制等游戏规则判断。

    Args:
        cards: 出牌列表

    Returns:
        Pattern对象，hand_type为INVALID表示无效牌型
    """
    if not cards:
        return Pattern(HandType.INVALID, Rank.THREE)

    n = len(cards)
    rank_count = _count_ranks(cards)
    ranks = sorted(rank_count.keys())
    counts = sorted(rank_count.values(), reverse=True)

    # ---------- 王炸 ----------
    small_jokers = sum(1 for c in cards if c.rank == Rank.SMALL_JOKER)
    big_jokers = sum(1 for c in cards if c.rank == Rank.BIG_JOKER)

    # 普通王炸：1小王+1大王
    if n == 2 and small_jokers == 1 and big_jokers == 1:
        return Pattern(HandType.ROCKET, Rank.BIG_JOKER, cards=cards, is_rocket=True)

    # 双王炸：2小王+2大王（两副牌模式）
    if BOMB_CONFIG['allow_double_rocket'] and n == 4 and small_jokers == 2 and big_jokers == 2:
        return Pattern(HandType.ROCKET, Rank.BIG_JOKER, cards=cards,
                       is_rocket=True, is_double_rocket=True)

    # ---------- 单张 ----------
    if n == 1:
        return Pattern(HandType.SINGLE, cards[0].rank, cards=cards)

    # ---------- 对子 ----------
    if n == 2 and len(rank_count) == 1 and counts[0] == 2:
        return Pattern(HandType.PAIR, ranks[0], cards=cards)

    # ---------- 三张 ----------
    if n == 3 and len(rank_count) == 1 and counts[0] == 3:
        return Pattern(HandType.TRIPLE, ranks[0], cards=cards)

    # ---------- 三带一 ----------
    if n == 4 and 3 in counts and len(rank_count) == 2:
        main_rank = [r for r, c in rank_count.items() if c >= 3][0]
        return Pattern(HandType.TRIPLE_ONE, main_rank, cards=cards)

    # ---------- 三带二 ----------
    if n == 5 and 3 in counts and 2 in counts and len(rank_count) == 2:
        main_rank = [r for r, c in rank_count.items() if c >= 3][0]
        return Pattern(HandType.TRIPLE_TWO, main_rank, cards=cards)

    # ---------- 炸弹（4张及以上同点数）----------
    if len(rank_count) == 1 and counts[0] >= BOMB_CONFIG['min_bomb_size']:
        bomb_size = counts[0]
        return Pattern(HandType.BOMB, ranks[0], cards=cards, bomb_size=bomb_size)

    # ---------- 顺子 ----------
    if n >= MIN_STRAIGHT_LEN and all(c == 1 for c in counts):
        sorted_ranks = sorted(ranks)
        if (sorted_ranks[-1] < Rank.TWO and
                sorted_ranks[-1] - sorted_ranks[0] == n - 1):
            return Pattern(HandType.STRAIGHT, sorted_ranks[-1],
                           length=n, cards=cards)

    # ---------- 两连对 ----------
    if n == 4 and all(c == 2 for c in counts):
        sorted_ranks = sorted(ranks)
        if (len(sorted_ranks) == 2 and
                sorted_ranks[-1] < Rank.TWO and
                sorted_ranks[-1] - sorted_ranks[0] == 1):
            return Pattern(HandType.TWO_STRAIGHT_PAIR, sorted_ranks[-1],
                           length=2, cards=cards)

    # ---------- 连对 ----------
    if n >= MIN_STRAIGHT_PAIR_LEN * 2 and all(c == 2 for c in counts):
        sorted_ranks = sorted(ranks)
        num_pairs = len(sorted_ranks)
        if (num_pairs >= MIN_STRAIGHT_PAIR_LEN and
                sorted_ranks[-1] < Rank.TWO and
                sorted_ranks[-1] - sorted_ranks[0] == num_pairs - 1):
            return Pattern(HandType.STRAIGHT_PAIR, sorted_ranks[-1],
                           length=num_pairs, cards=cards)

    # ---------- 连三张 ----------
    if n >= MIN_STRAIGHT_TRIPLE_LEN * 3 and all(c == 3 for c in counts):
        sorted_ranks = sorted(ranks)
        num_triples = len(sorted_ranks)
        if (num_triples >= MIN_STRAIGHT_TRIPLE_LEN and
                sorted_ranks[-1] < Rank.TWO and
                sorted_ranks[-1] - sorted_ranks[0] == num_triples - 1):
            return Pattern(HandType.STRAIGHT_TRIPLE, sorted_ranks[-1],
                           length=num_triples, cards=cards)

    # ---------- 飞机带单 ----------
    triple_ranks = sorted([r for r, c in rank_count.items() if c >= 3])
    if len(triple_ranks) >= 2:
        consecutive_triples = _find_consecutive(triple_ranks, MIN_STRAIGHT_TRIPLE_LEN)
        for seq in consecutive_triples:
            num_triples = len(seq)
            expected_total = num_triples * 4  # 每组三带一
            if n == expected_total:
                remaining = n - num_triples * 3
                if remaining == num_triples:
                    return Pattern(HandType.PLANE_ONE, seq[-1],
                                   length=num_triples, cards=cards)

    # ---------- 飞机带对 ----------
    if len(triple_ranks) >= 2:
        consecutive_triples = _find_consecutive(triple_ranks, MIN_STRAIGHT_TRIPLE_LEN)
        for seq in consecutive_triples:
            num_triples = len(seq)
            expected_total = num_triples * 5  # 每组三带二
            if n == expected_total:
                non_triple_count = {r: c for r, c in rank_count.items() if r not in seq}
                all_pairs = all(c == 2 for c in non_triple_count.values())
                if all_pairs and len(non_triple_count) == num_triples:
                    return Pattern(HandType.PLANE_TWO, seq[-1],
                                   length=num_triples, cards=cards)

    return Pattern(HandType.INVALID, Rank.THREE, cards=cards)


# ==================== 牌型比较 ====================

def compare_hands(play: Pattern, last_play: Pattern) -> int:
    """
    比较两手牌的大小

    Args:
        play: 当前出的牌
        last_play: 上一次出的牌（需要压过的牌）

    Returns:
        正数: play > last_play（可以压住）
        0: play == last_play（平局，一般不可能出现）
        负数: play < last_play（压不住）
    """
    # 无效牌型不能比较
    if play.hand_type == HandType.INVALID:
        return -1
    if last_play.hand_type == HandType.INVALID:
        return 1

    # 判断是否为炸弹类
    play_is_bomb = play.hand_type in (HandType.BOMB, HandType.ROCKET)
    last_is_bomb = last_play.hand_type in (HandType.BOMB, HandType.ROCKET)

    # 炸弹 vs 非炸弹
    if play_is_bomb and not last_is_bomb:
        return 1  # 炸弹压任意普通牌型
    if not play_is_bomb and last_is_bomb:
        return -1  # 普通牌型压不过炸弹

    # 两个都是炸弹类 → 统一炸弹比较
    if play_is_bomb and last_is_bomb:
        return _compare_bombs(play, last_play)

    # ---------- 两个都是普通牌型 ----------
    # 不同牌型不能比较（相同牌型才能压制）
    if play.hand_type != last_play.hand_type:
        return -1

    # 顺子、连对、两连对、连三张、飞机必须长度一致
    if play.hand_type in (HandType.STRAIGHT, HandType.STRAIGHT_PAIR,
                          HandType.TWO_STRAIGHT_PAIR,
                          HandType.STRAIGHT_TRIPLE, HandType.PLANE_ONE,
                          HandType.PLANE_TWO):
        if play.length != last_play.length:
            return -1  # 长度不一致，无法比较

    # 比较主牌点数
    if play.main_rank > last_play.main_rank:
        return 1
    elif play.main_rank < last_play.main_rank:
        return -1

    return -1  # 点数相同，后出的不能压


def _compare_bombs(play: Pattern, last_play: Pattern) -> int:
    """
    比较两个炸弹/王炸的大小

    使用 _effective_bomb_tier 统一比较逻辑：
    1. 先比较层级（张数/王炸等级）
    2. 层级相同则比较点数

    Args:
        play: 当前出的炸弹
        last_play: 上一次出的炸弹

    Returns:
        正数/0/负数
    """
    play_tier = _effective_bomb_tier(play)
    last_tier = _effective_bomb_tier(last_play)

    # 层级高的更大
    if play_tier > last_tier:
        return 1
    elif play_tier < last_tier:
        return -1

    # 层级相同，比较点数
    if play.main_rank > last_play.main_rank:
        return 1
    elif play.main_rank < last_play.main_rank:
        return -1

    return 0


# ==================== 合法出牌判断 ====================

def is_valid_play(cards: list[Card], last_play: Pattern | None,
                  is_free_turn: bool = False,
                  hand_cards: list[Card] = None) -> tuple[bool, Pattern | None, str]:
    """
    判断出牌是否合法

    Args:
        cards: 要出的牌
        last_play: 上一次成功出牌的牌型（None表示自由出牌）
        is_free_turn: 是否为自由出牌轮（新一轮开始）
        hand_cards: 玩家完整手牌（仅在禁止拆炸弹时用于检查）

    Returns:
        (是否合法, 识别出的牌型, 错误信息)
    """
    if not cards:
        return False, None, "没有选择牌"

    # 识别牌型
    pattern = recognize_pattern(cards)

    if pattern.hand_type == HandType.INVALID:
        return False, None, "无效的牌型"

    # ---------- 可选：炸弹拆分限制检查 ----------
    # 默认允许拆炸弹；如果配置为禁止，才阻止炸弹点数参与非炸弹牌型。
    if not BOMB_CONFIG.get('allow_split_bombs', True) and hand_cards is not None:
        hand_rank_count = _count_ranks(hand_cards)
        bomb_ranks = _get_bomb_ranks(hand_rank_count)
        play_rank_count = _count_ranks(cards)

        for rank in play_rank_count:
            if rank in bomb_ranks:
                # 使用了炸弹点数的牌，但出牌本身不是炸弹/王炸类型
                if pattern.hand_type not in (HandType.BOMB, HandType.ROCKET):
                    return False, None, "不能拆分炸弹出牌（4张及以上同点数只能作为炸弹使用）"
                # 是炸弹类型，但还需确认炸弹本身的张数是否合法
                # （例如用4张出5张炸弹点数的牌，是合法的4张炸弹）

    # 自由出牌（新一轮开始），任何合法牌型都可以
    if is_free_turn or last_play is None:
        return True, pattern, ""

    # 需要压过上家
    result = compare_hands(pattern, last_play)
    if result > 0:
        return True, pattern, ""
    elif result == 0:
        return False, None, "出的牌与上家相同，无法压住"
    else:
        return False, None, f"牌型无法压住上家的{HAND_TYPE_DISPLAY.get(last_play.hand_type, '牌')}"


# 延迟导入，避免循环引用
from constants import HAND_TYPE_DISPLAY


# ==================== 炸弹查找 ====================

def find_all_bombs(cards: list[Card], min_tier_to_beat: int = 0,
                   rank_to_beat: Rank = None) -> list[Pattern]:
    """
    找出手牌中所有可能的炸弹，遵循BOMB_CONFIG配置。

    Args:
        cards: 手牌列表
        min_tier_to_beat: 需要压过的炸弹层级（0表示不限）
        rank_to_beat: 需要压过的炸弹点数（同层级时比较）

    Returns:
        所有可能的炸弹牌型列表
    """
    bombs = []
    rank_count = _count_ranks(cards)

    # 普通炸弹
    for rank, count in rank_count.items():
        if count < BOMB_CONFIG['min_bomb_size']:
            continue
        rank_cards = [c for c in cards if c.rank == rank]
        # 枚举所有合法炸弹张数
        for size in range(BOMB_CONFIG['min_bomb_size'], count + 1):
            if not _is_bomb_size_allowed(size):
                continue
            bomb = Pattern(
                HandType.BOMB, rank,
                cards=rank_cards[:size],
                bomb_size=size
            )
            # 如果需要压过特定炸弹，只返回能赢的
            if min_tier_to_beat > 0:
                bomb_tier = _effective_bomb_tier(bomb)
                if bomb_tier < min_tier_to_beat:
                    continue
                if bomb_tier == min_tier_to_beat and rank_to_beat is not None:
                    if rank <= rank_to_beat:
                        continue
            bombs.append(bomb)

    # 王炸
    small_jokers = [c for c in cards if c.rank == Rank.SMALL_JOKER]
    big_jokers = [c for c in cards if c.rank == Rank.BIG_JOKER]

    if len(small_jokers) >= 1 and len(big_jokers) >= 1:
        rocket = Pattern(
            HandType.ROCKET, Rank.BIG_JOKER,
            cards=small_jokers[:1] + big_jokers[:1], is_rocket=True
        )
        if min_tier_to_beat > 0:
            rocket_tier = _effective_bomb_tier(rocket)
            if rocket_tier > min_tier_to_beat or (
                    rocket_tier == min_tier_to_beat and
                    (rank_to_beat is None or Rank.BIG_JOKER > rank_to_beat)):
                bombs.append(rocket)
        else:
            bombs.append(rocket)

    # 双王炸（两副牌模式）
    if (BOMB_CONFIG['allow_double_rocket'] and
            len(small_jokers) >= 2 and len(big_jokers) >= 2):
        double_rocket = Pattern(
            HandType.ROCKET, Rank.BIG_JOKER,
            cards=small_jokers[:2] + big_jokers[:2],
            is_rocket=True, is_double_rocket=True
        )
        if min_tier_to_beat > 0:
            dr_tier = _effective_bomb_tier(double_rocket)
            if dr_tier > min_tier_to_beat or (
                    dr_tier == min_tier_to_beat and
                    (rank_to_beat is None or Rank.BIG_JOKER > rank_to_beat)):
                bombs.append(double_rocket)
        else:
            bombs.append(double_rocket)

    return bombs


# ==================== 合法出牌候选查找 ====================

def find_all_valid_plays(cards: list[Card], last_play: Pattern | None = None) -> list[Pattern]:
    """
    找出手牌中所有合法出牌组合

    核心规则：
    - 默认允许拆炸弹；如需禁止，可设置 BOMB_CONFIG['allow_split_bombs'] = False
    - 所有候选牌型必须通过 compare_hands() 验证能压过上家
    - 尽量枚举合理的带牌组合（三带一/三带二/飞机带单/飞机带对）

    Args:
        cards: 手牌列表
        last_play: 上一次出牌的牌型（None表示自由出牌）

    Returns:
        所有合法出牌的Pattern列表
    """
    valid_plays = []
    rank_count = _count_ranks(cards)
    bomb_ranks = (
        set() if BOMB_CONFIG.get('allow_split_bombs', True)
        else _get_bomb_ranks(rank_count)
    )

    if last_play is None:
        _find_free_plays(cards, rank_count, bomb_ranks, valid_plays)
    else:
        _find_follow_plays(cards, rank_count, bomb_ranks, last_play, valid_plays)

    # ---------- 最终验证 ----------
    # 确保每个候选出牌都能通过 compare_hands 真正压过上家
    if last_play is not None:
        validated = []
        for play in valid_plays:
            if compare_hands(play, last_play) > 0:
                validated.append(play)
        valid_plays = validated

    return valid_plays


def _find_free_plays(cards: list[Card], rank_count: Counter,
                     bomb_ranks: set[Rank], valid_plays: list[Pattern]):
    """
    自由出牌时枚举所有合法牌型

    bomb_ranks 仅在配置禁止拆炸弹时非空；默认允许炸弹点数参与非炸弹牌型。

    Args:
        cards: 手牌
        rank_count: 手牌点数统计
        bomb_ranks: 炸弹点数集合
        valid_plays: 输出的合法牌型列表
    """
    # ========== 单张 ==========
    for rank in rank_count:
        if rank in bomb_ranks:
            continue
        rank_cards = _get_rank_cards(cards, rank)
        valid_plays.append(Pattern(HandType.SINGLE, rank, cards=[rank_cards[0]]))

    # ========== 对子 ==========
    for rank, count in rank_count.items():
        if rank in bomb_ranks:
            continue
        if count >= 2:
            pair_cards = _get_rank_cards(cards, rank)[:2]
            valid_plays.append(Pattern(HandType.PAIR, rank, cards=pair_cards))

    # ========== 三张 ==========
    for rank, count in rank_count.items():
        if rank in bomb_ranks:
            continue
        if count >= 3:
            triple_cards = _get_rank_cards(cards, rank)[:3]
            valid_plays.append(Pattern(HandType.TRIPLE, rank, cards=triple_cards))

    # ========== 三带一（枚举所有带牌）==========
    _find_triple_one_plays(cards, rank_count, bomb_ranks, valid_plays, min_rank=None)

    # ========== 三带二（枚举所有带对）==========
    _find_triple_two_plays(cards, rank_count, bomb_ranks, valid_plays, min_rank=None)

    # ========== 顺子 ==========
    single_ranks = sorted([r for r in rank_count if r < Rank.TWO and r not in bomb_ranks])
    consecutive = _find_consecutive(single_ranks, MIN_STRAIGHT_LEN)
    for seq in consecutive:
        seq_cards = []
        for r in seq:
            seq_cards.append(_get_rank_cards(cards, r)[0])
        valid_plays.append(Pattern(
            HandType.STRAIGHT, seq[-1],
            length=len(seq), cards=seq_cards
        ))

    # ========== 两连对（2连续对子，如5566）==========
    pair_ranks_for_two = sorted([r for r, c in rank_count.items()
                                  if c >= 2 and r < Rank.TWO and r not in bomb_ranks])
    for i in range(len(pair_ranks_for_two) - 1):
        if pair_ranks_for_two[i + 1] == pair_ranks_for_two[i] + 1:
            r1 = pair_ranks_for_two[i]
            r2 = pair_ranks_for_two[i + 1]
            seq_cards = _get_rank_cards(cards, r1)[:2] + _get_rank_cards(cards, r2)[:2]
            valid_plays.append(Pattern(
                HandType.TWO_STRAIGHT_PAIR, r2,
                length=2, cards=seq_cards
            ))

    # ========== 连对（3对及以上）==========
    pair_ranks = sorted([r for r, c in rank_count.items()
                          if c >= 2 and r < Rank.TWO and r not in bomb_ranks])
    consecutive = _find_consecutive(pair_ranks, MIN_STRAIGHT_PAIR_LEN)
    for seq in consecutive:
        seq_cards = []
        for r in seq:
            seq_cards.extend(_get_rank_cards(cards, r)[:2])
        valid_plays.append(Pattern(
            HandType.STRAIGHT_PAIR, seq[-1],
            length=len(seq), cards=seq_cards
        ))

    # ========== 连三张 ==========
    triple_ranks = sorted([r for r, c in rank_count.items()
                            if c >= 3 and r < Rank.TWO and r not in bomb_ranks])
    consecutive = _find_consecutive(triple_ranks, MIN_STRAIGHT_TRIPLE_LEN)
    for seq in consecutive:
        seq_cards = []
        for r in seq:
            seq_cards.extend(_get_rank_cards(cards, r)[:3])
        valid_plays.append(Pattern(
            HandType.STRAIGHT_TRIPLE, seq[-1],
            length=len(seq), cards=seq_cards
        ))

    # ========== 飞机带单（枚举带牌组合）==========
    _find_plane_one_plays(cards, rank_count, bomb_ranks, valid_plays, min_rank=None)

    # ========== 飞机带对（枚举带对组合）==========
    _find_plane_two_plays(cards, rank_count, bomb_ranks, valid_plays, min_rank=None)

    # ========== 炸弹 ==========
    valid_plays.extend(find_all_bombs(cards))


def _find_follow_plays(cards: list[Card], rank_count: Counter,
                       bomb_ranks: set[Rank], last_play: Pattern,
                       valid_plays: list[Pattern]):
    """
    跟牌时枚举所有能压住上家的合法牌型

    bomb_ranks 仅在配置禁止拆炸弹时非空。
    最终验证由 find_all_valid_plays 中的 compare_hands 检查保证。

    Args:
        cards: 手牌
        rank_count: 手牌点数统计
        bomb_ranks: 炸弹点数集合
        last_play: 上一次出牌
        valid_plays: 输出的合法牌型列表
    """
    target_type = last_play.hand_type

    if target_type == HandType.SINGLE:
        for rank in sorted(rank_count.keys()):
            if rank in bomb_ranks:
                continue
            if rank > last_play.main_rank:
                rank_cards = _get_rank_cards(cards, rank)
                valid_plays.append(Pattern(HandType.SINGLE, rank, cards=[rank_cards[0]]))

    elif target_type == HandType.PAIR:
        for rank in sorted(rank_count.keys()):
            if rank in bomb_ranks:
                continue
            if rank > last_play.main_rank and rank_count[rank] >= 2:
                pair_cards = _get_rank_cards(cards, rank)[:2]
                valid_plays.append(Pattern(HandType.PAIR, rank, cards=pair_cards))

    elif target_type == HandType.TRIPLE:
        for rank in sorted(rank_count.keys()):
            if rank in bomb_ranks:
                continue
            if rank > last_play.main_rank and rank_count[rank] >= 3:
                triple_cards = _get_rank_cards(cards, rank)[:3]
                valid_plays.append(Pattern(HandType.TRIPLE, rank, cards=triple_cards))

    elif target_type == HandType.TRIPLE_ONE:
        _find_triple_one_plays(cards, rank_count, bomb_ranks, valid_plays,
                               min_rank=last_play.main_rank)

    elif target_type == HandType.TRIPLE_TWO:
        _find_triple_two_plays(cards, rank_count, bomb_ranks, valid_plays,
                               min_rank=last_play.main_rank)

    elif target_type == HandType.STRAIGHT:
        target_len = last_play.length
        single_ranks = sorted([r for r in rank_count if r < Rank.TWO and r not in bomb_ranks])
        consecutive = _find_consecutive(single_ranks, target_len)
        for seq in consecutive:
            if len(seq) == target_len and seq[-1] > last_play.main_rank:
                seq_cards = []
                for r in seq:
                    seq_cards.append(_get_rank_cards(cards, r)[0])
                valid_plays.append(Pattern(
                    HandType.STRAIGHT, seq[-1],
                    length=target_len, cards=seq_cards
                ))

    elif target_type == HandType.STRAIGHT_PAIR:
        target_len = last_play.length
        pair_ranks = sorted([r for r, c in rank_count.items()
                              if c >= 2 and r < Rank.TWO and r not in bomb_ranks])
        consecutive = _find_consecutive(pair_ranks, target_len)
        for seq in consecutive:
            if len(seq) == target_len and seq[-1] > last_play.main_rank:
                seq_cards = []
                for r in seq:
                    seq_cards.extend(_get_rank_cards(cards, r)[:2])
                valid_plays.append(Pattern(
                    HandType.STRAIGHT_PAIR, seq[-1],
                    length=target_len, cards=seq_cards
                ))

    elif target_type == HandType.TWO_STRAIGHT_PAIR:
        pair_ranks = sorted([r for r, c in rank_count.items()
                              if c >= 2 and r < Rank.TWO and r not in bomb_ranks])
        for i in range(len(pair_ranks) - 1):
            if pair_ranks[i + 1] == pair_ranks[i] + 1:
                if pair_ranks[i + 1] > last_play.main_rank:
                    r1 = pair_ranks[i]
                    r2 = pair_ranks[i + 1]
                    seq_cards = _get_rank_cards(cards, r1)[:2] + _get_rank_cards(cards, r2)[:2]
                    valid_plays.append(Pattern(
                        HandType.TWO_STRAIGHT_PAIR, r2,
                        length=2, cards=seq_cards
                    ))

    elif target_type == HandType.STRAIGHT_TRIPLE:
        target_len = last_play.length
        triple_ranks = sorted([r for r, c in rank_count.items()
                                if c >= 3 and r < Rank.TWO and r not in bomb_ranks])
        consecutive = _find_consecutive(triple_ranks, target_len)
        for seq in consecutive:
            if len(seq) == target_len and seq[-1] > last_play.main_rank:
                seq_cards = []
                for r in seq:
                    seq_cards.extend(_get_rank_cards(cards, r)[:3])
                valid_plays.append(Pattern(
                    HandType.STRAIGHT_TRIPLE, seq[-1],
                    length=target_len, cards=seq_cards
                ))

    elif target_type == HandType.PLANE_ONE:
        _find_plane_one_plays(cards, rank_count, bomb_ranks, valid_plays,
                              min_rank=last_play.main_rank, target_len=last_play.length)

    elif target_type == HandType.PLANE_TWO:
        _find_plane_two_plays(cards, rank_count, bomb_ranks, valid_plays,
                              min_rank=last_play.main_rank, target_len=last_play.length)

    elif target_type == HandType.BOMB:
        # 使用统一的炸弹查找，只返回能压过上家炸弹的
        last_tier = _effective_bomb_tier(last_play)
        bomb_plays = find_all_bombs(cards, min_tier_to_beat=last_tier,
                                     rank_to_beat=last_play.main_rank)
        valid_plays.extend(bomb_plays)

    elif target_type == HandType.ROCKET:
        # 王炸只能被双王炸压（如果允许且等级足够）
        if BOMB_CONFIG['allow_double_rocket'] and not last_play.is_double_rocket:
            last_tier = _effective_bomb_tier(last_play)
            small_jokers = [c for c in cards if c.rank == Rank.SMALL_JOKER]
            big_jokers = [c for c in cards if c.rank == Rank.BIG_JOKER]
            if len(small_jokers) >= 2 and len(big_jokers) >= 2:
                double_rocket = Pattern(
                    HandType.ROCKET, Rank.BIG_JOKER,
                    cards=small_jokers[:2] + big_jokers[:2],
                    is_rocket=True, is_double_rocket=True
                )
                if _effective_bomb_tier(double_rocket) > last_tier:
                    valid_plays.append(double_rocket)

    # ========== 任何非炸弹牌型都可以被炸弹/王炸压 ==========
    if target_type not in (HandType.BOMB, HandType.ROCKET):
        bomb_plays = find_all_bombs(cards)
        valid_plays.extend(bomb_plays)
    # 注意：跟炸弹时，find_all_bombs 已经包含了能赢的王炸/双王炸，无需重复添加


# ==================== 组合牌型候选生成 ====================

def _find_triple_one_plays(cards: list[Card], rank_count: Counter,
                            bomb_ranks: set[Rank], valid_plays: list[Pattern],
                            min_rank: Rank = None):
    """
    生成三带一的所有候选组合

    枚举每个合法三张 + 每个可用点数的带牌。
    不再只取第一种带牌方案，而是尽量枚举合理组合。

    Args:
        cards: 手牌
        rank_count: 点数统计
        bomb_ranks: 炸弹点数集合
        valid_plays: 输出列表
        min_rank: 跟牌时三张主牌需要大于此点数（None表示自由出牌）
    """
    for rank, count in rank_count.items():
        if rank in bomb_ranks:
            continue
        if count < 3:
            continue
        if min_rank is not None and rank <= min_rank:
            continue

        triple_cards = _get_rank_cards(cards, rank)[:3]

        # 枚举所有可用点数、非三张本身点数的单牌
        other_ranks = [r for r in rank_count if r != rank and r not in bomb_ranks]
        for kicker_rank in other_ranks:
            kicker_cards = _get_rank_cards(cards, kicker_rank)[:1]
            valid_plays.append(Pattern(
                HandType.TRIPLE_ONE, rank,
                cards=triple_cards + kicker_cards
            ))


def _find_triple_two_plays(cards: list[Card], rank_count: Counter,
                            bomb_ranks: set[Rank], valid_plays: list[Pattern],
                            min_rank: Rank = None):
    """
    生成三带二的所有候选组合

    枚举每个合法三张 + 每个可用点数的对子带牌。

    Args:
        cards: 手牌
        rank_count: 点数统计
        bomb_ranks: 炸弹点数集合
        valid_plays: 输出列表
        min_rank: 跟牌时三张主牌需要大于此点数
    """
    for rank, count in rank_count.items():
        if rank in bomb_ranks:
            continue
        if count < 3:
            continue
        if min_rank is not None and rank <= min_rank:
            continue

        triple_cards = _get_rank_cards(cards, rank)[:3]

        # 枚举所有可用点数、非三张本身点数的对子
        for other_rank, other_count in rank_count.items():
            if other_rank == rank or other_rank in bomb_ranks:
                continue
            if other_count < 2:
                continue
            pair_cards = _get_rank_cards(cards, other_rank)[:2]
            valid_plays.append(Pattern(
                HandType.TRIPLE_TWO, rank,
                cards=triple_cards + pair_cards
            ))


def _find_plane_one_plays(cards: list[Card], rank_count: Counter,
                           bomb_ranks: set[Rank], valid_plays: list[Pattern],
                           min_rank: Rank = None, target_len: int = None):
    """
    生成飞机带单的所有候选组合

    枚举连续三张 + 不同带牌单牌的组合。
    限制每种飞机骨架最多生成 10 种不同带牌方案，避免组合爆炸。

    Args:
        cards: 手牌
        rank_count: 点数统计
        bomb_ranks: 炸弹点数集合
        valid_plays: 输出列表
        min_rank: 跟牌时飞机尾部点数需大于此值
        target_len: 跟牌时飞机长度需匹配
    """
    triple_ranks = sorted([r for r, c in rank_count.items()
                            if c >= 3 and r < Rank.TWO and r not in bomb_ranks])

    if len(triple_ranks) < MIN_STRAIGHT_TRIPLE_LEN:
        return

    plane_found = set()  # 用 (尾部点数, 长度) 去重

    # 尝试不同长度的连续三张（从最长到最短）
    for seq_len in range(len(triple_ranks), MIN_STRAIGHT_TRIPLE_LEN - 1, -1):
        for start_idx in range(len(triple_ranks) - seq_len + 1):
            sub_seq = triple_ranks[start_idx:start_idx + seq_len]

            # 检查是否连续
            is_consecutive = all(
                sub_seq[i + 1] == sub_seq[i] + 1 for i in range(len(sub_seq) - 1)
            )
            if not is_consecutive:
                continue

            num_triples = len(sub_seq)
            dedup_key = (sub_seq[-1], num_triples)
            if dedup_key in plane_found:
                continue

            # 跟牌时检查长度和点数
            if target_len is not None and num_triples != target_len:
                continue
            if min_rank is not None and sub_seq[-1] <= min_rank:
                continue

            # 收集连续三张的牌
            seq_cards = []
            for r in sub_seq:
                seq_cards.extend(_get_rank_cards(cards, r)[:3])

            # 找单牌（可用点数、非三张本身点数）
            used_ranks = set(sub_seq)
            remaining_ranks = [r for r in rank_count
                               if r not in used_ranks and r not in bomb_ranks]

            # 收集可用的带牌
            available_kickers = []
            for r in remaining_ranks:
                available_kickers.extend(_get_rank_cards(cards, r))
            # 三张中多余的牌（4张中第4张）也可作为带牌
            for r in sub_seq:
                extra = _get_rank_cards(cards, r)[3:]
                available_kickers.extend(extra)

            if len(available_kickers) < num_triples:
                continue

            # 生成带牌组合（限制数量避免爆炸）
            kicker_combos = list(combinations(range(len(available_kickers)), num_triples))
            # 最多取 10 种组合
            kicker_combos = kicker_combos[:10]

            for combo in kicker_combos:
                kicker_cards = [available_kickers[i] for i in combo]
                plane_cards = seq_cards + kicker_cards
                # 验证牌型
                pattern = recognize_pattern(plane_cards)
                if pattern.hand_type == HandType.PLANE_ONE:
                    valid_plays.append(Pattern(
                        HandType.PLANE_ONE, sub_seq[-1],
                        length=num_triples, cards=plane_cards
                    ))
                    plane_found.add(dedup_key)

            # 即使没有枚举到有效组合，也标记此骨架已尝试
            if dedup_key not in plane_found:
                plane_found.add(dedup_key)


def _find_plane_two_plays(cards: list[Card], rank_count: Counter,
                           bomb_ranks: set[Rank], valid_plays: list[Pattern],
                           min_rank: Rank = None, target_len: int = None):
    """
    生成飞机带对的所有候选组合

    枚举连续三张 + 不同带牌对子的组合。
    限制每种飞机骨架最多生成 10 种不同带对方案。

    Args:
        cards: 手牌
        rank_count: 点数统计
        bomb_ranks: 炸弹点数集合
        valid_plays: 输出列表
        min_rank: 跟牌时飞机尾部点数需大于此值
        target_len: 跟牌时飞机长度需匹配
    """
    triple_ranks = sorted([r for r, c in rank_count.items()
                            if c >= 3 and r < Rank.TWO and r not in bomb_ranks])

    if len(triple_ranks) < MIN_STRAIGHT_TRIPLE_LEN:
        return

    plane_found = set()

    for seq_len in range(len(triple_ranks), MIN_STRAIGHT_TRIPLE_LEN - 1, -1):
        for start_idx in range(len(triple_ranks) - seq_len + 1):
            sub_seq = triple_ranks[start_idx:start_idx + seq_len]

            is_consecutive = all(
                sub_seq[i + 1] == sub_seq[i] + 1 for i in range(len(sub_seq) - 1)
            )
            if not is_consecutive:
                continue

            num_triples = len(sub_seq)
            dedup_key = (sub_seq[-1], num_triples)
            if dedup_key in plane_found:
                continue

            if target_len is not None and num_triples != target_len:
                continue
            if min_rank is not None and sub_seq[-1] <= min_rank:
                continue

            seq_cards = []
            for r in sub_seq:
                seq_cards.extend(_get_rank_cards(cards, r)[:3])

            # 找对子作为带牌（可用点数、非三张本身点数）
            used_ranks = set(sub_seq)
            pair_ranks_remaining = [
                r for r, c in rank_count.items()
                if r not in used_ranks and r not in bomb_ranks and c >= 2
            ]

            if len(pair_ranks_remaining) < num_triples:
                continue

            # 生成对子组合
            pair_combos = list(combinations(pair_ranks_remaining, num_triples))
            pair_combos = pair_combos[:10]  # 限制数量

            for pair_set in pair_combos:
                plane_cards = list(seq_cards)
                for pr in pair_set:
                    plane_cards.extend(_get_rank_cards(cards, pr)[:2])
                # 验证牌型
                pattern = recognize_pattern(plane_cards)
                if pattern.hand_type == HandType.PLANE_TWO:
                    valid_plays.append(Pattern(
                        HandType.PLANE_TWO, sub_seq[-1],
                        length=num_triples, cards=plane_cards
                    ))
                    plane_found.add(dedup_key)

            if dedup_key not in plane_found:
                plane_found.add(dedup_key)
