"""
争上游卡牌游戏 - 牌组类
"""
import random
from card import Card
from constants import Suit, Rank, MAX_DECKS


class Deck:
    """
    牌组类
    支持创建1副牌(54张)或2副牌(108张)
    支持洗牌和发牌
    """
    
    def __init__(self, num_decks: int = 1):
        """
        初始化牌组
        
        Args:
            num_decks: 牌副数，1或2
        """
        if num_decks not in (1, 2):
            raise ValueError(f"牌副数必须为1或2，当前为{num_decks}")
        self.num_decks = num_decks
        self.cards: list[Card] = []
        self._build()
    
    def _build(self):
        """构建牌组"""
        self.cards = []
        
        for deck_id in range(self.num_decks):
            # 添加普通牌（52张：4花色 × 13点数）
            for suit in [Suit.SPADE, Suit.HEART, Suit.DIAMOND, Suit.CLUB]:
                for rank in range(Rank.THREE, Rank.TWO + 1):  # 3到2
                    self.cards.append(Card(suit, Rank(rank), deck_id))
            
            # 添加王牌（2张：小王 + 大王）
            self.cards.append(Card(Suit.JOKER, Rank.SMALL_JOKER, deck_id))
            self.cards.append(Card(Suit.JOKER, Rank.BIG_JOKER, deck_id))
    
    def shuffle(self):
        """洗牌"""
        random.shuffle(self.cards)
    
    def deal(self, num_players: int) -> list[list[Card]]:
        """
        发牌，将所有牌平均分给玩家
        
        Args:
            num_players: 玩家数量
            
        Returns:
            每位玩家的手牌列表
        """
        total = len(self.cards)
        base = total // num_players
        remainder = total % num_players
        
        hands = []
        idx = 0
        for i in range(num_players):
            count = base + (1 if i < remainder else 0)
            hand = self.cards[idx:idx + count]
            # 对手牌排序
            hand.sort(key=lambda c: c.sort_key)
            hands.append(hand)
            idx += count
        
        return hands
    
    @property
    def total_cards(self) -> int:
        """总牌数"""
        return len(self.cards)
    
    def find_spade_three(self) -> Card | None:
        """
        找到黑桃3（用于确定先手玩家）
        两副牌模式下返回第一张黑桃3
        
        Returns:
            黑桃3牌，如果不存在返回None
        """
        for card in self.cards:
            if card.suit == Suit.SPADE and card.rank == Rank.THREE:
                return card
        return None
