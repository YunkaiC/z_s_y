"""
争上游卡牌游戏 - 常量定义
"""
from enum import IntEnum

# ==================== 牌面相关常量 ====================

class Suit(IntEnum):
    """花色枚举"""
    SPADE = 0      # 黑桃 ♠
    HEART = 1      # 红心 ♥
    DIAMOND = 2    # 方块 ♦
    CLUB = 3       # 梅花 ♣
    JOKER = 4      # 王（无花色）

class Rank(IntEnum):
    """点数枚举，值越大牌越大"""
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 15
    SMALL_JOKER = 16   # 小王
    BIG_JOKER = 17     # 大王

# 点数显示名称映射
RANK_DISPLAY = {
    Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5', Rank.SIX: '6',
    Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9', Rank.TEN: '10',
    Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K', Rank.ACE: 'A',
    Rank.TWO: '2', Rank.SMALL_JOKER: '小王', Rank.BIG_JOKER: '大王',
}

# 花色显示名称映射
SUIT_DISPLAY = {
    Suit.SPADE: '♠', Suit.HEART: '♥', Suit.DIAMOND: '♦', Suit.CLUB: '♣', Suit.JOKER: '',
}

# 花色颜色 (pygame颜色)
SUIT_COLORS = {
    Suit.SPADE: (0, 0, 0),       # 黑
    Suit.HEART: (220, 20, 20),   # 红
    Suit.DIAMOND: (220, 20, 20), # 红
    Suit.CLUB: (0, 0, 0),        # 黑
    Suit.JOKER: (0, 0, 0),       # 黑（实际会根据大小王变色）
}

# ==================== 牌型常量 ====================

class HandType(IntEnum):
    """牌型枚举"""
    INVALID = 0           # 无效
    SINGLE = 1            # 单张
    PAIR = 2              # 对子
    TRIPLE = 3            # 三张
    TRIPLE_ONE = 4        # 三带一
    TRIPLE_TWO = 5        # 三带二
    STRAIGHT = 6          # 顺子（至少5张）
    STRAIGHT_PAIR = 7     # 连对（至少3连对）
    STRAIGHT_TRIPLE = 8   # 连三张（至少2连三张）
    PLANE_ONE = 9         # 飞机带单（连续三张+等数量单牌）
    PLANE_TWO = 10        # 飞机带对（连续三张+等数量对子）
    BOMB = 11             # 炸弹（4张及以上同点数）
    ROCKET = 12           # 王炸（小王+大王）
    TWO_STRAIGHT_PAIR = 13  # 两连对（2连续对子，如5566）

HAND_TYPE_DISPLAY = {
    HandType.INVALID: '无效',
    HandType.SINGLE: '单张',
    HandType.PAIR: '对子',
    HandType.TRIPLE: '三张',
    HandType.TRIPLE_ONE: '三带一',
    HandType.TRIPLE_TWO: '三带二',
    HandType.STRAIGHT: '顺子',
    HandType.STRAIGHT_PAIR: '连对',
    HandType.STRAIGHT_TRIPLE: '连三张',
    HandType.PLANE_ONE: '飞机带单',
    HandType.PLANE_TWO: '飞机带对',
    HandType.BOMB: '炸弹',
    HandType.ROCKET: '王炸',
    HandType.TWO_STRAIGHT_PAIR: '两连对',
}

# ==================== 网络常量 ====================

DEFAULT_PORT = 9999
BUFFER_SIZE = 65536

# 消息类型
class MsgType:
    """网络消息类型"""
    # 房间相关
    CREATE_ROOM = "create_room"
    JOIN_ROOM = "join_room"
    LEAVE_ROOM = "leave_room"
    ROOM_INFO = "room_info"
    PLAYER_JOINED = "player_joined"
    PLAYER_LEFT = "player_left"
    PLAYER_READY = "player_ready"
    
    # 游戏流程
    GAME_START = "game_start"
    DEAL_CARDS = "deal_cards"
    YOUR_TURN = "your_turn"
    PLAY_CARDS = "play_cards"
    PASS = "pass"
    PLAY_RESULT = "play_result"
    STATE_UPDATE = "state_update"
    TURN_CHANGE = "turn_change"
    ROUND_WIN = "round_win"
    PLAYER_FINISH = "player_finish"
    GAME_OVER = "game_over"
    
    # 系统
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    RECONNECT = "reconnect"

# ==================== UI常量 ====================

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800
FPS = 60

# 颜色
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (34, 139, 34)        # 桌面绿色
COLOR_DARK_GREEN = (0, 100, 0)
COLOR_RED = (220, 20, 20)
COLOR_BLUE = (30, 100, 200)
COLOR_GRAY = (180, 180, 180)
COLOR_DARK_GRAY = (100, 100, 100)
COLOR_LIGHT_GRAY = (220, 220, 220)
COLOR_YELLOW = (255, 215, 0)
COLOR_GOLD = (255, 200, 0)
COLOR_ORANGE = (255, 140, 0)

# 牌的尺寸
CARD_WIDTH = 70
CARD_HEIGHT = 100
CARD_GAP = 25  # 手牌之间的间距

# ==================== 游戏设置常量 ====================

MIN_PLAYERS = 3
MAX_PLAYERS = 4
MIN_DECKS = 1
MAX_DECKS = 2

# 顺子最少张数
MIN_STRAIGHT_LEN = 5
# 连对最少连数
MIN_STRAIGHT_PAIR_LEN = 3  # 即至少3对
# 两连对连数（固定为2）
MIN_TWO_STRAIGHT_PAIR_LEN = 2  # 即2对
# 连三张最少连数
MIN_STRAIGHT_TRIPLE_LEN = 2  # 即至少2个三张

# 先手规则
class FirstPlayRule:
    SPADE_THREE = "spade_three"  # 持有黑桃3先出
    RANDOM = "random"           # 随机先手
