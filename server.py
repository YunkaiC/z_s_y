"""
争上游卡牌游戏 - FastAPI + WebSocket 局域网联机服务器

启动方式: python server.py
玩家通过浏览器访问 http://<局域网IP>:8000 加入游戏
"""
import asyncio
import json
import uuid
import socket
import os
from typing import Optional
from contextlib import asynccontextmanager

# Get the directory of this file for resolving paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from card import Card
from deck import Deck
from constants import Suit, Rank, HandType, RANK_DISPLAY, SUIT_DISPLAY, HAND_TYPE_DISPLAY, FirstPlayRule
from game_logic import recognize_pattern, compare_hands, is_valid_play, Pattern, find_all_valid_plays
from ai_engine import AIEngine


# ==================== 房间与玩家管理 ====================

class Player:
    def __init__(self, player_id: str, name: str, ws: WebSocket):
        self.id = player_id
        self.name = name
        self.ws = ws
        self.cards: list[Card] = []
        self.ready = False
        self.rank = 0  # 0=未完成, >0=名次
        self.is_connected = True

    @property
    def card_count(self) -> int:
        return len(self.cards)

    def to_dict(self, include_cards=False) -> dict:
        d = {
            'id': self.id,
            'name': self.name,
            'card_count': self.card_count,
            'ready': self.ready,
            'rank': self.rank,
            'is_connected': self.is_connected,
        }
        if include_cards:
            d['cards'] = [c.to_dict() for c in self.cards]
        return d


class GameRoom:
    def __init__(self, room_id: str, host_id: str, num_decks: int = 1,
                 num_players: int = 3, first_play_rule: str = "spade_three",
                 ai_fill: bool = True, ai_difficulty: str = "normal"):
        self.id = room_id
        self.host_id = host_id
        self.num_decks = num_decks
        self.num_players = num_players
        self.first_play_rule = first_play_rule
        self.ai_fill = ai_fill
        self.ai_difficulty = ai_difficulty  # 'easy', 'normal', 'hard'

        self.players: dict[str, Player] = {}  # id -> Player
        self.player_order: list[str] = []  # 出牌顺序

        # 游戏状态
        self.started = False
        self.current_player_idx = 0
        self.last_play: Optional[Pattern] = None
        self.last_play_player_id: Optional[str] = None
        self.consecutive_passes = 0
        self.is_free_turn = True
        self.play_history: list[dict] = []
        self.ranked_players: list[str] = []
        self.game_over = False

        # AI
        self.ai_engine = AIEngine(difficulty=ai_difficulty)
        self.ai_players: dict[str, dict] = {}  # AI player info
        self.ai_timer: Optional[asyncio.Task] = None
        self.ai_timeout_timer: Optional[asyncio.Task] = None  # 超时保护定时器

    @property
    def real_player_count(self) -> int:
        return len([p for p in self.players.values() if p.id not in self.ai_players])

    def add_player(self, player: Player) -> bool:
        total = len(self.players) + len(self.ai_players)
        if total >= self.num_players:
            return False
        if player.id in self.players:
            return False
        self.players[player.id] = player
        self.player_order.append(player.id)
        return True

    def remove_player(self, player_id: str):
        if player_id in self.players:
            del self.players[player_id]
            if player_id in self.player_order:
                self.player_order.remove(player_id)

    def add_ai_player(self, ai_name: str = None) -> Optional[str]:
        """手动添加一个AI玩家到房间"""
        total = len(self.players) + len(self.ai_players)
        if total >= self.num_players:
            return None
        ai_names = ['🤖 AI-小红', '🤖 AI-小蓝', '🤖 AI-小绿']
        # 找一个还没用的名字
        used_names = {ai['name'] for ai in self.ai_players.values()}
        if ai_name is None:
            for name in ai_names:
                if name not in used_names:
                    ai_name = name
                    break
            if ai_name is None:
                ai_name = f'🤖 AI-{len(self.ai_players)+1}'
        ai_id = f"ai_{uuid.uuid4().hex[:8]}"
        ai_info = {
            'id': ai_id,
            'name': ai_name,
            'cards': [],
            'rank': 0,
        }
        self.ai_players[ai_id] = ai_info
        self.player_order.append(ai_id)
        return ai_id

    def remove_ai_player(self, ai_id: str) -> bool:
        """移除一个AI玩家"""
        if ai_id not in self.ai_players:
            return False
        del self.ai_players[ai_id]
        if ai_id in self.player_order:
            self.player_order.remove(ai_id)
        return True

    def fill_ai_players(self):
        """用AI补齐空位"""
        ai_count = self.num_players - len(self.players) - len(self.ai_players)
        for i in range(ai_count):
            self.add_ai_player()

    def get_current_player_id(self) -> Optional[str]:
        if not self.player_order or self.game_over:
            return None
        return self.player_order[self.current_player_idx % len(self.player_order)]

    def get_current_player_name(self) -> str:
        pid = self.get_current_player_id()
        if pid in self.players:
            return self.players[pid].name
        if pid in self.ai_players:
            return self.ai_players[pid]['name']
        return "?"

    def _count_active_players(self) -> int:
        """统计仍在场的活跃玩家数量"""
        return sum(1 for pid in self.player_order
                   if (pid in self.players and self.players[pid].rank == 0) or
                   (pid in self.ai_players and self.ai_players[pid]['rank'] == 0))

    def _is_player_active(self, player_id: str) -> bool:
        """检查指定玩家是否仍在场"""
        if player_id in self.players:
            return self.players[player_id].rank == 0
        if player_id in self.ai_players:
            return self.ai_players[player_id]['rank'] == 0
        return False

    def next_turn(self):
        """
        移动到下一个未完成玩家，并检查是否应切换为自由出牌。

        自由出牌触发规则：
        - 某玩家出牌后，所有其他活跃玩家都选择了"不要"
        - 此时轮次重新回到出牌者（或其下家，若已完赛），由其自由出牌
        - 关键：如果出牌者已完赛，需要所有剩余活跃玩家都不要才能触发
        """
        total = len(self.player_order)
        for _ in range(total):
            self.current_player_idx = (self.current_player_idx + 1) % total
            pid = self.player_order[self.current_player_idx]
            # 跳过已完成的玩家
            if pid in self.players and self.players[pid].rank > 0:
                continue
            if pid in self.ai_players and self.ai_players[pid]['rank'] > 0:
                continue
            break

        # 检查是否应切换为自由出牌
        if self.last_play is not None and self.last_play_player_id is not None:
            active_count = self._count_active_players()
            last_player_active = self._is_player_active(self.last_play_player_id)

            # 如果上一位出牌者仍在场：需要 active_count - 1 人不要
            #   （除了出牌者自己，其他所有人都不要）
            # 如果上一位出牌者已完赛：需要 active_count 人不要
            #   （所有剩余活跃玩家都不要，因为出牌者已不在场）
            threshold = active_count - 1 if last_player_active else active_count

            if self.consecutive_passes >= threshold:
                self.is_free_turn = True
                self.last_play = None
                self.last_play_player_id = None
                self.consecutive_passes = 0

    def to_dict(self, for_player_id: Optional[str] = None) -> dict:
        """序列化房间状态，只给指定玩家看自己的手牌"""
        players_info = []
        for pid in self.player_order:
            if pid in self.players:
                p = self.players[pid]
                include_cards = (pid == for_player_id)
                players_info.append(p.to_dict(include_cards=include_cards))
            elif pid in self.ai_players:
                ai = self.ai_players[pid]
                info = {
                    'id': ai['id'],
                    'name': ai['name'],
                    'card_count': len(ai['cards']),
                    'ready': True,
                    'rank': ai['rank'],
                    'is_connected': True,
                    'is_ai': True,
                }
                if pid == for_player_id:
                    info['cards'] = [c.to_dict() for c in ai['cards']]
                players_info.append(info)

        last_play_info = None
        if self.last_play:
            last_play_info = {
                'hand_type': int(self.last_play.hand_type),
                'hand_type_name': HAND_TYPE_DISPLAY.get(self.last_play.hand_type, '?'),
                'main_rank': int(self.last_play.main_rank),
                'cards': [c.to_dict() for c in self.last_play.cards],
                'bomb_size': self.last_play.bomb_size,
                'is_rocket': self.last_play.is_rocket,
            }

        return {
            'room_id': self.id,
            'host_id': self.host_id,
            'num_decks': self.num_decks,
            'num_players': self.num_players,
            'first_play_rule': self.first_play_rule,
            'started': self.started,
            'game_over': self.game_over,
            'current_player_id': self.get_current_player_id(),
            'current_player_name': self.get_current_player_name(),
            'is_free_turn': self.is_free_turn,
            'last_play': last_play_info,
            'last_play_player_id': self.last_play_player_id,
            'players': players_info,
            'play_history': self.play_history[-10:],  # 最近10条
            'ranked_players': self.ranked_players,
            'ai_difficulty': self.ai_difficulty,
        }


# ==================== 全局房间管理 ====================

rooms: dict[str, GameRoom] = {}
player_room_map: dict[str, str] = {}  # player_id -> room_id


def get_local_ip() -> str:
    """获取本机局域网IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n{'='*50}")
    print(f"  争上游 - 局域网联机服务器已启动")
    print(f"  本机IP: {get_local_ip()}")
    print(f"  访问地址: http://{get_local_ip()}:8000")
    print(f"{'='*50}\n")
    yield

app = FastAPI(title="争上游 - 局域网联机", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/ip")
async def get_ip():
    return {"ip": get_local_ip(), "port": 8000}


# ==================== 游戏逻辑 ====================

async def broadcast_room(room: GameRoom, exclude_id: Optional[str] = None):
    """向房间内所有在线玩家广播房间状态"""
    for pid, player in room.players.items():
        if pid == exclude_id or not player.is_connected:
            continue
        try:
            state = room.to_dict(for_player_id=pid)
            await player.ws.send_json({
                'type': 'room_update',
                'data': state,
            })
        except Exception:
            pass


async def send_to_player(room: GameRoom, player_id: str, msg: dict):
    """给指定玩家发消息"""
    if player_id in room.players and room.players[player_id].is_connected:
        try:
            await room.players[player_id].ws.send_json(msg)
        except Exception:
            pass


async def start_game(room: GameRoom):
    """开始游戏"""
    # 补齐AI
    if room.ai_fill:
        room.fill_ai_players()

    # 创建牌组并洗牌
    deck = Deck(room.num_decks)
    deck.shuffle()

    # 发牌
    total_players = len(room.player_order)
    hands = deck.deal(total_players)

    for i, pid in enumerate(room.player_order):
        if pid in room.players:
            room.players[pid].cards = hands[i]
            room.players[pid].rank = 0
        elif pid in room.ai_players:
            room.ai_players[pid]['cards'] = hands[i]
            room.ai_players[pid]['rank'] = 0

    # 确定先手
    if room.first_play_rule == FirstPlayRule.SPADE_THREE:
        for i, pid in enumerate(room.player_order):
            cards = room.players[pid].cards if pid in room.players else room.ai_players[pid]['cards']
            if any(c.suit == Suit.SPADE and c.rank == Rank.THREE for c in cards):
                room.current_player_idx = i
                break
    else:
        import random
        room.current_player_idx = random.randint(0, total_players - 1)

    room.started = True
    room.game_over = False
    room.is_free_turn = True
    room.last_play = None
    room.last_play_player_id = None
    room.consecutive_passes = 0
    room.play_history = []
    room.ranked_players = []

    # 广播游戏开始
    for pid, player in room.players.items():
        if player.is_connected:
            state = room.to_dict(for_player_id=pid)
            try:
                await player.ws.send_json({
                    'type': 'game_start',
                    'data': state,
                })
            except Exception:
                pass

    # 如果当前是AI，触发AI出牌
    current_id = room.get_current_player_id()
    if current_id in room.ai_players:
        await schedule_ai_play(room)


async def process_play(room: GameRoom, player_id: str, card_dicts: list[dict]) -> dict:
    """处理玩家出牌"""
    if room.game_over:
        return {'success': False, 'error': '游戏已结束'}

    if room.get_current_player_id() != player_id:
        return {'success': False, 'error': '不是你的回合'}

    # 获取玩家手牌
    if player_id in room.players:
        player_cards = room.players[player_id].cards
        player_name = room.players[player_id].name
    elif player_id in room.ai_players:
        player_cards = room.ai_players[player_id]['cards']
        player_name = room.ai_players[player_id]['name']
    else:
        return {'success': False, 'error': '玩家不存在'}

    # 反序列化出的牌
    played_cards = [Card.from_dict(d) for d in card_dicts]

    # 验证牌是否在手中
    played_set = set((c.suit, c.rank, c.deck_id) for c in played_cards)
    hand_set = set((c.suit, c.rank, c.deck_id) for c in player_cards)
    if not played_set.issubset(hand_set):
        return {'success': False, 'error': '出牌不在手中'}

    # 验证牌型（传入完整手牌以检查炸弹不可拆分规则）
    valid, pattern, error = is_valid_play(
        played_cards, room.last_play, room.is_free_turn,
        hand_cards=player_cards
    )

    if not valid:
        return {'success': False, 'error': error}

    # 从手牌中移除出的牌
    remaining = [c for c in player_cards if (c.suit, c.rank, c.deck_id) not in played_set]
    if player_id in room.players:
        room.players[player_id].cards = remaining
    else:
        room.ai_players[player_id]['cards'] = remaining

    # 更新游戏状态
    room.last_play = pattern
    room.last_play_player_id = player_id
    room.consecutive_passes = 0
    room.is_free_turn = False

    # 记录出牌历史
    history_entry = {
        'player_id': player_id,
        'player_name': player_name,
        'cards': [c.to_dict() for c in played_cards],
        'hand_type': int(pattern.hand_type),
        'hand_type_name': HAND_TYPE_DISPLAY.get(pattern.hand_type, '?'),
        'passed': False,
    }
    room.play_history.append(history_entry)

    # 检查玩家是否出完牌
    card_count = len(remaining)
    if card_count == 0:
        rank = len(room.ranked_players) + 1
        if player_id in room.players:
            room.players[player_id].rank = rank
        else:
            room.ai_players[player_id]['rank'] = rank
        room.ranked_players.append(player_id)

    # 移动到下一个玩家
    room.next_turn()

    # 检查游戏是否结束
    active_count = room._count_active_players()

    if active_count <= 1:
        for pid in room.player_order:
            if pid in room.players and room.players[pid].rank == 0:
                room.players[pid].rank = len(room.ranked_players) + 1
                room.ranked_players.append(pid)
            elif pid in room.ai_players and room.ai_players[pid]['rank'] == 0:
                room.ai_players[pid]['rank'] = len(room.ranked_players) + 1
                room.ranked_players.append(pid)
        room.game_over = True

    # 广播状态更新
    await broadcast_room(room)

    # 如果游戏结束，广播game_over
    if room.game_over:
        rankings = []
        for i, pid in enumerate(room.ranked_players):
            name = room.players[pid].name if pid in room.players else room.ai_players[pid]['name']
            rankings.append({'rank': i + 1, 'player_id': pid, 'name': name})
        for pid, player in room.players.items():
            if player.is_connected:
                try:
                    await player.ws.send_json({
                        'type': 'game_over',
                        'data': {
                            'rankings': rankings,
                            'total_turns': len(room.play_history),
                        },
                    })
                except Exception:
                    pass
        return {'success': True, 'game_over': True}

    # 如果下一个是AI，触发AI出牌
    current_id = room.get_current_player_id()
    if current_id in room.ai_players:
        await schedule_ai_play(room)

    return {'success': True}


async def process_pass(room: GameRoom, player_id: str) -> dict:
    """处理玩家不要（过）"""
    if room.game_over:
        return {'success': False, 'error': '游戏已结束'}

    if room.get_current_player_id() != player_id:
        return {'success': False, 'error': '不是你的回合'}

    if room.is_free_turn:
        return {'success': False, 'error': '自由出牌轮不能不出'}

    player_name = room.players[player_id].name if player_id in room.players else room.ai_players[player_id]['name']

    room.consecutive_passes += 1

    history_entry = {
        'player_id': player_id,
        'player_name': player_name,
        'cards': [],
        'hand_type': 0,
        'hand_type_name': '',
        'passed': True,
    }
    room.play_history.append(history_entry)

    room.next_turn()

    await broadcast_room(room)

    current_id = room.get_current_player_id()
    if current_id in room.ai_players:
        await schedule_ai_play(room)

    return {'success': True}


async def get_hint(room: GameRoom, player_id: str) -> list[dict]:
    """获取提示"""
    if player_id in room.players:
        cards = room.players[player_id].cards
    elif player_id in room.ai_players:
        cards = room.ai_players[player_id]['cards']
    else:
        return []

    valid_plays = find_all_valid_plays(cards, room.last_play if not room.is_free_turn else None)
    hints = []
    for play in valid_plays[:5]:  # 最多返回5个提示
        hints.append({
            'cards': [c.to_dict() for c in play.cards],
            'hand_type_name': HAND_TYPE_DISPLAY.get(play.hand_type, '?'),
        })
    return hints


async def schedule_ai_play(room: GameRoom):
    """
    调度AI出牌（延迟1秒）+ 超时保护

    关键设计：每次调用都会取消之前的定时器，确保只有一个AI出牌任务在排队。
    这避免了因 schedule_ai_play 在 _ai_play_after_delay 执行期间被调用
    而导致下一个AI无法被调度的问题。
    """
    # 取消之前的AI出牌定时器（如果还在等待中）
    if room.ai_timer and not room.ai_timer.done():
        room.ai_timer.cancel()
    # 取消之前的超时保护
    if room.ai_timeout_timer and not room.ai_timeout_timer.done():
        room.ai_timeout_timer.cancel()

    room.ai_timer = asyncio.create_task(_ai_play_after_delay(room))
    room.ai_timeout_timer = asyncio.create_task(_ai_timeout_guard(room))


async def _ai_timeout_guard(room: GameRoom):
    """AI出牌超时保护：5秒后强制出牌"""
    await asyncio.sleep(5.0)
    if room.game_over:
        return
    # 超时保护：强制为当前AI执行出牌
    current_id = room.get_current_player_id()
    if current_id and current_id in room.ai_players:
        print(f"[TIMEOUT] AI play timeout, forcing action for {current_id}")
        await _ai_force_play(room, current_id)


async def _ai_force_play(room: GameRoom, ai_id: str):
    """
    强制AI出牌（超时或异常时的兜底）

    改进：不仅尝试不出，还尝试找到合法出牌。
    只有在真的无法出牌时才选择不要。
    """
    if room.game_over:
        return
    if room.get_current_player_id() != ai_id:
        return
    if ai_id not in room.ai_players:
        return

    cards = room.ai_players[ai_id]['cards']
    if not cards:
        return

    if room.is_free_turn:
        # 自由出牌：必须出牌
        # 先尝试 find_all_valid_plays
        valid_plays = find_all_valid_plays(cards, None)
        if valid_plays:
            # 选最小的出
            valid_plays.sort(key=lambda p: (p.main_rank, len(p.cards)))
            card_dicts = [c.to_dict() for c in valid_plays[0].cards]
            result = await process_play(room, ai_id, card_dicts)
            if result.get('success'):
                return
        # 兜底：出最小的单张
        sorted_cards = sorted(cards, key=lambda c: c.sort_key)
        for card in sorted_cards:
            result = await process_play(room, ai_id, [card.to_dict()])
            if result.get('success'):
                return
    else:
        # 跟牌：先尝试找到能压的牌
        valid_plays = find_all_valid_plays(cards, room.last_play)
        if valid_plays:
            # 优先出非炸弹的最小牌
            non_bomb = [p for p in valid_plays if p.hand_type not in (HandType.BOMB, HandType.ROCKET)]
            plays_to_try = non_bomb if non_bomb else valid_plays
            plays_to_try.sort(key=lambda p: (p.main_rank, len(p.cards)))
            card_dicts = [c.to_dict() for c in plays_to_try[0].cards]
            result = await process_play(room, ai_id, card_dicts)
            if result.get('success'):
                return
        # 无法压住，选择不出
        await process_pass(room, ai_id)


async def _ai_play_after_delay(room: GameRoom):
    """
    AI延迟后出牌

    关键设计：在执行任何游戏逻辑之前，先将 room.ai_timer 置为 None。
    这确保了当 process_play / process_pass 内部调用 schedule_ai_play 时，
    不会因为"上一个定时器还在运行"而被拒绝调度。
    """
    await asyncio.sleep(1.0)

    # 立即清除定时器引用，允许后续 schedule_ai_play 调用创建新定时器
    room.ai_timer = None

    if room.game_over:
        return

    current_id = room.get_current_player_id()
    if current_id not in room.ai_players:
        return

    ai = room.ai_players[current_id]
    cards = ai['cards']

    if not cards:
        return

    # 计算对手最少手牌数（包含所有玩家：人类+AI）
    opponent_min = 999
    # 收集对手手牌数量（用于Monte Carlo模拟的隐藏手牌分配）
    my_player_idx = room.player_order.index(current_id) if current_id in room.player_order else 0
    opponent_card_counts = []  # [(player_idx, card_count), ...]

    for i, pid in enumerate(room.player_order):
        if pid == current_id:
            continue
        if pid in room.players and room.players[pid].rank == 0:
            opp_count = len(room.players[pid].cards)
            opponent_min = min(opponent_min, opp_count)
            opponent_card_counts.append((i, opp_count))
        elif pid in room.ai_players and room.ai_players[pid]['rank'] == 0:
            opp_count = len(room.ai_players[pid]['cards'])
            opponent_min = min(opponent_min, opp_count)
            opponent_card_counts.append((i, opp_count))

    # 收集已出牌（用于Monte Carlo模拟的隐藏手牌精确分配）
    played_cards = []
    for entry in room.play_history:
        if not entry.get('passed', False):
            for card_dict in entry.get('cards', []):
                played_cards.append(Card.from_dict(card_dict))

    # AI决策 - 扩展game_state以支持Monte Carlo模拟
    game_state = {
        'opponent_min_cards': opponent_min,
        'num_decks': room.num_decks,
        'opponent_card_counts': opponent_card_counts,
        'my_player_idx': my_player_idx,
        'played_cards': played_cards,
    }
    play_cards, should_pass = room.ai_engine.decide(
        cards, room.last_play, room.is_free_turn, game_state
    )

    if should_pass and not room.is_free_turn:
        await process_pass(room, current_id)
    else:
        # 尝试出牌
        result = None
        if play_cards:
            card_dicts = [c.to_dict() for c in play_cards]
            result = await process_play(room, current_id, card_dicts)
            if result.get('success'):
                return  # 出牌成功

        # 出牌失败或没有返回牌
        if room.is_free_turn:
            # 自由出牌必须出牌，不能不出
            # 尝试使用find_all_valid_plays找到合法出牌
            valid_plays = find_all_valid_plays(cards, None)
            if valid_plays:
                # 按消耗牌数从少到多排序，优先出小牌
                valid_plays.sort(key=lambda p: (p.main_rank, len(p.cards)))
                for play in valid_plays:
                    card_dicts = [c.to_dict() for c in play.cards]
                    result = await process_play(room, current_id, card_dicts)
                    if result.get('success'):
                        return

            # 最后兜底：出最小的单张
            sorted_cards = sorted(cards, key=lambda c: c.sort_key)
            for card in sorted_cards:
                result = await process_play(room, current_id, [card.to_dict()])
                if result.get('success'):
                    return

            # 极端情况：所有出牌都失败了
            print(f"[ERROR] AI {current_id} cannot play any card on free turn! Cards: {len(cards)}")
        else:
            # 跟牌失败，尝试 find_all_valid_plays 再找一次
            valid_plays = find_all_valid_plays(cards, room.last_play)
            if valid_plays:
                non_bomb = [p for p in valid_plays if p.hand_type not in (HandType.BOMB, HandType.ROCKET)]
                plays_to_try = non_bomb if non_bomb else valid_plays
                plays_to_try.sort(key=lambda p: (p.main_rank, len(p.cards)))
                for play in plays_to_try[:3]:
                    card_dicts = [c.to_dict() for c in play.cards]
                    result = await process_play(room, current_id, card_dicts)
                    if result.get('success'):
                        return
            # 确实无法压住，选择不出
            await process_pass(room, current_id)


# ==================== WebSocket 处理 ====================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    player_id = None
    room_id = None

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get('type', '')

            # ---- 创建房间 ----
            if msg_type == 'create_room':
                player_id = f"p_{uuid.uuid4().hex[:8]}"
                player_name = data.get('name', '玩家1')
                num_decks = data.get('num_decks', 1)
                num_players = data.get('num_players', 3)
                first_play_rule = data.get('first_play_rule', 'spade_three')
                ai_fill = data.get('ai_fill', True)
                ai_difficulty = data.get('ai_difficulty', 'normal')

                room_id = f"r_{uuid.uuid4().hex[:6]}"
                room = GameRoom(room_id, player_id, num_decks, num_players, first_play_rule, ai_fill, ai_difficulty)
                rooms[room_id] = room

                player = Player(player_id, player_name, ws)
                room.add_player(player)
                player_room_map[player_id] = room_id

                await ws.send_json({
                    'type': 'room_created',
                    'data': {
                        'room_id': room_id,
                        'player_id': player_id,
                        'room': room.to_dict(for_player_id=player_id),
                    }
                })
                print(f"[Room {room_id}] Created by {player_name}")

            # ---- 加入房间 ----
            elif msg_type == 'join_room':
                room_id_input = data.get('room_id', '')
                if not room_id_input and len(rooms) == 1:
                    room_id = list(rooms.keys())[0]
                else:
                    room_id = room_id_input

                if room_id not in rooms:
                    await ws.send_json({'type': 'error', 'data': {'message': '房间不存在'}})
                    continue

                room = rooms[room_id]
                if room.started:
                    await ws.send_json({'type': 'error', 'data': {'message': '游戏已开始，无法加入'}})
                    continue

                total_in_room = len(room.players) + len(room.ai_players)
                if total_in_room >= room.num_players:
                    await ws.send_json({'type': 'error', 'data': {'message': '房间已满'}})
                    continue

                player_id = f"p_{uuid.uuid4().hex[:8]}"
                player_name = data.get('name', f'玩家{len(room.players)+1}')

                player = Player(player_id, player_name, ws)
                room.add_player(player)
                player_room_map[player_id] = room_id

                await ws.send_json({
                    'type': 'room_joined',
                    'data': {
                        'room_id': room_id,
                        'player_id': player_id,
                        'room': room.to_dict(for_player_id=player_id),
                    }
                })
                await broadcast_room(room, exclude_id=player_id)
                print(f"[Room {room_id}] {player_name} joined")

            # ---- 添加AI玩家 ----
            elif msg_type == 'add_ai':
                if room_id and room_id in rooms:
                    room = rooms[room_id]
                    if not room.started and player_id == room.host_id:
                        ai_name = data.get('name')
                        ai_id = room.add_ai_player(ai_name)
                        if ai_id:
                            await broadcast_room(room)
                        else:
                            await send_to_player(room, player_id, {
                                'type': 'error',
                                'data': {'message': '房间已满，无法添加AI'},
                            })

            # ---- 移除AI玩家 ----
            elif msg_type == 'remove_ai':
                if room_id and room_id in rooms:
                    room = rooms[room_id]
                    if not room.started and player_id == room.host_id:
                        ai_id = data.get('ai_id')
                        if ai_id and room.remove_ai_player(ai_id):
                            await broadcast_room(room)
                        else:
                            await send_to_player(room, player_id, {
                                'type': 'error',
                                'data': {'message': '移除AI失败'},
                            })

            # ---- 准备 ----
            elif msg_type == 'ready':
                if room_id and room_id in rooms and player_id in rooms[room_id].players:
                    room = rooms[room_id]
                    room.players[player_id].ready = not room.players[player_id].ready
                    await broadcast_room(room)

                    all_ready = all(p.ready for p in room.players.values())
                    total_players = len(room.players) + len(room.ai_players)
                    can_start = all_ready and len(room.players) >= 1 and total_players >= 2
                    if can_start:
                        for pid, p in room.players.items():
                            if p.is_connected:
                                await p.ws.send_json({
                                    'type': 'all_ready',
                                    'data': {'can_start': True}
                                })

            # ---- 开始游戏 ----
            elif msg_type == 'start_game':
                if room_id and room_id in rooms:
                    room = rooms[room_id]
                    if room.host_id == player_id:
                        await start_game(room)
                        print(f"[Room {room_id}] Game started")

            # ---- 出牌 ----
            elif msg_type == 'play_cards':
                if room_id and room_id in rooms:
                    room = rooms[room_id]
                    card_dicts = data.get('cards', [])
                    result = await process_play(room, player_id, card_dicts)
                    if not result.get('success'):
                        await send_to_player(room, player_id, {
                            'type': 'play_result',
                            'data': result,
                        })

            # ---- 不要 ----
            elif msg_type == 'pass':
                if room_id and room_id in rooms:
                    room = rooms[room_id]
                    result = await process_pass(room, player_id)
                    if not result.get('success'):
                        await send_to_player(room, player_id, {
                            'type': 'play_result',
                            'data': result,
                        })

            # ---- 提示 ----
            elif msg_type == 'get_hint':
                if room_id and room_id in rooms:
                    room = rooms[room_id]
                    hints = await get_hint(room, player_id)
                    await send_to_player(room, player_id, {
                        'type': 'hint',
                        'data': {'hints': hints},
                    })

            # ---- 心跳 ----
            elif msg_type == 'ping':
                await ws.send_json({'type': 'pong'})

    except WebSocketDisconnect:
        print(f"[WS] Player {player_id} disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        # 清理
        if player_id and room_id and room_id in rooms:
            room = rooms[room_id]
            if player_id in room.players:
                room.players[player_id].is_connected = False
                if not room.started:
                    room.remove_player(player_id)
                    del player_room_map[player_id]
                    await broadcast_room(room)
                else:
                    await broadcast_room(room)

            active_players = [p for p in room.players.values() if p.is_connected]
            if not active_players and not room.started:
                del rooms[room_id]
                print(f"[Room {room_id}] Deleted (empty)")


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"\n{'='*50}")
    print(f"  争上游 - 局域网联机服务器")
    print(f"  本机局域网地址: http://{get_local_ip()}:{port}")
    print(f"  同一WiFi下的玩家在浏览器输入上述地址即可加入")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
