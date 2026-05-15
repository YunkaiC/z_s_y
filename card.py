"""
争上游卡牌游戏 - 扑克牌类
"""
from constants import Suit, Rank, RANK_DISPLAY, SUIT_DISPLAY, SUIT_COLORS


class Card:
    """
    扑克牌类
    每张牌由花色(suit)和点数(rank)唯一确定
    支持比较操作（按点数大小比较）
    """
    
    # 用于JSON序列化的分隔符
    SEP = "-"
    
    def __init__(self, suit: Suit, rank: Rank, deck_id: int = 0):
        """
        初始化一张牌
        
        Args:
            suit: 花色
            rank: 点数
            deck_id: 牌副编号（两副牌模式下区分同点数同花色的牌）
        """
        self.suit = suit
        self.rank = rank
        self.deck_id = deck_id  # 用于区分两副牌中完全相同的牌
    
    @property
    def is_joker(self) -> bool:
        """是否为王牌"""
        return self.suit == Suit.JOKER
    
    @property
    def is_big_joker(self) -> bool:
        """是否为大王"""
        return self.rank == Rank.BIG_JOKER
    
    @property
    def is_small_joker(self) -> bool:
        """是否为小王"""
        return self.rank == Rank.SMALL_JOKER
    
    @property
    def display_rank(self) -> str:
        """获取点数显示文本"""
        return RANK_DISPLAY.get(self.rank, '?')
    
    @property
    def display_suit(self) -> str:
        """获取花色显示文本"""
        return SUIT_DISPLAY.get(self.suit, '?')
    
    @property
    def color(self) -> tuple:
        """获取牌面文字颜色"""
        if self.is_big_joker:
            return (220, 20, 20)  # 红色
        elif self.is_small_joker:
            return (0, 0, 0)      # 黑色
        return SUIT_COLORS.get(self.suit, (0, 0, 0))
    
    @property
    def sort_key(self) -> tuple:
        """
        排序键，用于手牌排序
        先按点数升序，再按花色升序，再按牌副编号
        """
        return (self.rank, self.suit, self.deck_id)
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank and self.deck_id == other.deck_id
    
    def __lt__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.sort_key < other.sort_key
    
    def __le__(self, other):
        return self == other or self < other
    
    def __gt__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.sort_key > other.sort_key
    
    def __ge__(self, other):
        return self == other or self > other
    
    def __hash__(self):
        return hash((self.suit, self.rank, self.deck_id))
    
    def __repr__(self):
        if self.is_joker:
            return f"[{self.display_rank}]"
        return f"[{self.display_suit}{self.display_rank}]"
    
    def __str__(self):
        return self.__repr__()
    
    def to_dict(self) -> dict:
        """序列化为字典，用于网络传输"""
        return {
            'suit': int(self.suit),
            'rank': int(self.rank),
            'deck_id': self.deck_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Card':
        """从字典反序列化"""
        return cls(
            suit=Suit(data['suit']),
            rank=Rank(data['rank']),
            deck_id=data.get('deck_id', 0),
        )
    
    def to_str(self) -> str:
        """序列化为字符串"""
        return f"{int(self.suit)}{self.SEP}{int(self.rank)}{self.SEP}{self.deck_id}"
    
    @classmethod
    def from_str(cls, s: str) -> 'Card':
        """从字符串反序列化"""
        parts = s.split(cls.SEP)
        return cls(
            suit=Suit(int(parts[0])),
            rank=Rank(int(parts[1])),
            deck_id=int(parts[2]),
        )
