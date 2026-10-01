"""Procedural GPO skill cards for the semantic coach.

These are ENGINEERED reusable skills.  A small VLM does not need to memorize
the whole game or invent long plans from scratch; it identifies the current
situation and selects the most relevant skill card.  The existing supervisor
and executor still validate every concrete action.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillCard:
    skill_id: str
    title: str
    when: str
    procedure: tuple[str, ...]
    success: tuple[str, ...]
    recovery: tuple[str, ...]
    actions: tuple[str, ...]
    keywords: tuple[str, ...]

    def compact(self) -> str:
        proc = " -> ".join(self.procedure)
        success = "; ".join(self.success)
        recovery = "; ".join(self.recovery)
        actions = ",".join(self.actions)
        return (
            f"[{self.skill_id}] {self.title}\n"
            f"WHEN: {self.when}\n"
            f"DO: {proc}\n"
            f"SUCCESS: {success}\n"
            f"IF FAIL: {recovery}\n"
            f"ALLOWED: {actions}"
        )


SKILL_CARDS: tuple[SkillCard, ...] = (
    SkillCard(
        "quest_accept",
        "Accept a visible quest safely",
        "A yellow QUEST NPC/marker is visible and no active quest is confirmed.",
        (
            "NAVIGATE_OBJECTIVE until the quest giver is close and centered",
            "TAKE_QUEST or INTERACT",
            "REOBSERVE instead of pressing T repeatedly",
        ),
        (
            "quest UI/objective changes",
            "green objective appears",
            "active quest state becomes visible",
        ),
        (
            "if T causes no visible change, move slightly closer and REOBSERVE",
            "if the NPC/marker disappears, stop and reacquire instead of spamming",
        ),
        ("NAVIGATE_OBJECTIVE", "TAKE_QUEST", "INTERACT", "REOBSERVE"),
        ("quest", "yellow", "npc", "interact", "t", "quest giver"),
    ),
    SkillCard(
        "quest_travel",
        "Travel to the active quest objective",
        "A green recommended objective/waypoint is visible.",
        (
            "let the fruit-fly brain own ordinary left/right/approach steering",
            "SPRINT only on a clear long path",
            "REOBSERVE when the objective changes or progress stops",
        ),
        (
            "green objective becomes close",
            "quest target changes to a red enemy marker",
            "quest giver or destination becomes visible",
        ),
        (
            "if visual progress stalls, switch to obstacle_recovery",
            "if objective disappears, REOBSERVE rather than moving randomly",
        ),
        ("NAVIGATE_OBJECTIVE", "SPRINT", "REOBSERVE"),
        ("green", "waypoint", "objective", "travel", "navigate", "island"),
    ),
    SkillCard(
        "quest_combat",
        "Fight only the confirmed quest enemy",
        "A red quest/enemy marker identifies the current NPC objective.",
        (
            "approach using the fly's orientation",
            "FIGHT_QUEST_TARGET at useful range",
            "use BLOCK or EVADE against incoming pressure",
            "USE_OBSERVED_ABILITY only when both HUD label and key are visible",
            "REOBSERVE after knockdown/death/quest progress",
        ),
        (
            "quest counter/progress changes",
            "red target disappears because it is defeated",
            "new quest objective appears",
        ),
        (
            "if attacks do not affect a Logia-like target, verify Haki ownership",
            "if health becomes critical, disengage/pause rather than facetank",
            "never retarget an ordinary player",
        ),
        (
            "NAVIGATE_OBJECTIVE", "FIGHT_QUEST_TARGET", "BLOCK", "EVADE",
            "USE_OBSERVED_ABILITY", "USE_HAKI", "REOBSERVE",
        ),
        ("red", "enemy", "combat", "fight", "bandit", "boss", "health"),
    ),
    SkillCard(
        "obstacle_recovery",
        "Recover from walls, ledges and no-progress movement",
        "Movement is not changing the scene or a physical obstacle blocks progress.",
        (
            "low ledge or step: JUMP once",
            "open lateral route: GO_AROUND",
            "after another failed contact: BACKTRACK then REOBSERVE",
            "CLIMB only when visual semantics explicitly identify a climbable wall",
        ),
        (
            "scene position changes",
            "objective bearing/distance improves",
            "character clears the obstacle",
        ),
        (
            "do not repeat jump/climb indefinitely",
            "after two failed attempts change strategy",
            "never use autonomous camera motion to search",
        ),
        ("JUMP", "CLIMB", "GO_AROUND", "BACKTRACK", "REOBSERVE"),
        ("wall", "stuck", "blocked", "ledge", "climb", "jump", "obstacle"),
    ),
    SkillCard(
        "shop_purchase",
        "Buy a useful visible progression item",
        "An in-game shop/dialog is visibly open and a needed item/button is readable.",
        (
            "identify exact item and visible price",
            "verify it matches the current progression goal",
            "BUY_ITEM/UI_CLICK only on the clearly labeled in-game button",
            "REOBSERVE and verify inventory/equipment/state changed",
        ),
        (
            "item appears in inventory/equipment",
            "currency decreases by the expected amount",
            "dialog confirms purchase",
        ),
        (
            "if price/item is unclear, WAIT or REOBSERVE",
            "never confirm Robux/gamepass/account/external-link/trade UI",
        ),
        ("BUY_ITEM", "UI_CLICK", "REOBSERVE", "WAIT"),
        ("shop", "buy", "price", "peli", "item", "accessory", "weapon"),
    ),
    SkillCard(
        "equipment_use",
        "Equip and use a known visible loadout",
        "A hotbar/equipment move is visibly identified.",
        (
            "EQUIP_SLOT only when the intended slot/item is known",
            "USE_OBSERVED_ABILITY only if both move name and key are readable",
            "REOBSERVE after changing fruit/style/weapon because bindings may change",
        ),
        (
            "intended item is visibly equipped",
            "HUD shows the expected named moves",
        ),
        (
            "never infer a move from a key alone",
            "if HUD is ambiguous, keep using safe basic controls",
        ),
        ("EQUIP_SLOT", "USE_OBSERVED_ABILITY", "REOBSERVE"),
        ("equip", "hotbar", "weapon", "sword", "fruit", "style", "ability", "move"),
    ),
    SkillCard(
        "ship_travel",
        "Board and use a ship for sea travel",
        "Sea travel is required and an owned/spawned ship or seat is visible.",
        (
            "approach the ship/seat",
            "BOARD_SHIP when close",
            "TRAVEL only after seated/attached state is visible",
            "REOBSERVE when the destination/island changes",
        ),
        (
            "character is attached to the ship",
            "ship begins moving toward the intended route",
            "destination island/marker becomes nearer",
        ),
        (
            "if boarding fails, reposition and retry once",
            "if ship is badly damaged, prioritize repair/safe travel",
            "do not buy Robux transport autonomously",
        ),
        ("BOARD_SHIP", "TRAVEL", "REOBSERVE"),
        ("ship", "boat", "sea", "sail", "board", "island", "travel"),
    ),
    SkillCard(
        "survival_defense",
        "Preserve health while continuing the quest",
        "Health drops, an attack is incoming, or melee pressure is high.",
        (
            "BLOCK when timing/range favors defense",
            "EVADE when continuing to stand still is unsafe",
            "REOBSERVE after the defense before choosing the next attack",
        ),
        (
            "health stops dropping",
            "distance to enemy improves",
            "block/evade creates a safe attack window",
        ),
        (
            "critical health means disengage/pause",
            "do not let the semantic coach override the hard F12 safety stop",
        ),
        ("BLOCK", "EVADE", "REOBSERVE", "WAIT"),
        ("damage", "health", "block", "evade", "defense", "hit"),
    ),
)


def select_skill_cards(context: str, max_cards: int = 5) -> list[SkillCard]:
    """Return the most relevant procedural cards for a compact scene/state."""
    text = str(context or "").lower()
    scored = []
    for idx, card in enumerate(SKILL_CARDS):
        score = 0
        for kw in card.keywords:
            if kw in text:
                score += 3 if len(kw) >= 6 else 2

        # Important state patterns get a stronger deterministic nudge.
        if card.skill_id == "quest_accept" and (
                "quest_marker" in text or "yellow_quest" in text):
            score += 7
        if card.skill_id == "quest_travel" and (
                "recommended_quest_waypoint" in text or "green" in text):
            score += 7
        if card.skill_id == "quest_combat" and "quest_enemy_marker" in text:
            score += 8
        if card.skill_id == "obstacle_recovery" and (
                "stuck" in text or "stall" in text or "blocked" in text):
            score += 7

        if score:
            scored.append((-score, idx, card))

    scored.sort()
    selected = [x[2] for x in scored[:max(1, int(max_cards))]]

    # The quest loop is the default skill family if the scene has little text.
    if not selected:
        selected = [SKILL_CARDS[0], SKILL_CARDS[1], SKILL_CARDS[3]]
    return selected


def render_skill_cards(context: str, max_cards: int = 5) -> str:
    return "\n\n".join(
        card.compact() for card in select_skill_cards(context, max_cards)
    )


def procedural_skill_plan(obs: dict, quest: dict | None = None) -> dict | None:
    """Return a high-confidence skill candidate for obvious GPO situations.

    These candidates are now fed to the local SmolVLM text skill selector.
    They remain the fail-safe if local inference stalls.

    Important live correction: a red NPC diamond is NOT sufficient proof of
    an active quest target.  If a yellow quest-giver cue is simultaneously
    visible, accepting/reacquiring the quest takes priority over combat.
    """
    obs = dict(obs or {})
    quest = dict(quest or {})
    target = dict(obs.get("target") or {})
    notes = dict(obs.get("notes") or {})
    ui = dict(obs.get("ui") or {})
    player = dict(obs.get("player") or {})

    target_type = str(target.get("type") or "none")
    try:
        proximity = float(target.get("distance"))
    except (TypeError, ValueError):
        proximity = None
    try:
        direction = float(target.get("direction"))
    except (TypeError, ValueError):
        direction = None

    yellow = bool(notes.get("quest_marker_detected"))
    try:
        yellow_proximity = float(notes.get("quest_marker_proximity"))
    except (TypeError, ValueError):
        yellow_proximity = None
    try:
        yellow_direction = float(notes.get("quest_marker_direction"))
    except (TypeError, ValueError):
        yellow_direction = None

    if bool(ui.get("dialogue")) or bool(ui.get("menu")):
        return None

    try:
        health = float(player.get("health"))
    except (TypeError, ValueError):
        health = None
    if health is not None and health <= 0.18:
        return {
            "scene": "critical_health",
            "objective": "survive and create distance",
            "target": target_type,
            "skill": "REOBSERVE",
            "confidence": 0.99,
            "explanation": "Hard survival/reflex layer owns critical-health defense.",
            "next_after_success": "resume the quest only after health is safe",
            "skill_card": "survival_defense",
        }

    # A visible quest giver has priority over red NPC diamonds.  The user's
    # live run showed red diamonds above non-quest Corrupt Marines before the
    # Bandit quest was correctly established.
    if yellow:
        if (yellow_proximity is not None and yellow_proximity >= 0.68
                and yellow_direction is not None
                and abs(yellow_direction) <= 0.50):
            return {
                "scene": "quest_giver",
                "objective": "accept or confirm the visible quest",
                "target": "yellow quest giver",
                "skill": "TAKE_QUEST",
                "confidence": 0.99,
                "explanation": "The yellow quest cue is close and centered.",
                "next_after_success": "verify the quest/objective changed",
                "skill_card": "quest_accept",
            }
        return {
            "scene": "approach_quest_giver",
            "objective": "move closer to the yellow quest giver",
            "target": "yellow quest giver",
            "skill": "NAVIGATE_OBJECTIVE",
            "confidence": 0.97,
            "explanation": "A quest giver is visible but is not yet in safe interaction position.",
            "next_after_success": "interact only when close and centered",
            "skill_card": "quest_accept",
        }

    if target_type == "recommended_quest_waypoint":
        return {
            "scene": "quest_travel",
            "objective": "follow the active green quest objective",
            "target": "green recommended quest waypoint",
            "skill": "NAVIGATE_OBJECTIVE",
            "confidence": 0.99,
            "explanation": "The green recommended quest waypoint is the current travel objective.",
            "next_after_success": "reobserve when the marker changes or becomes close",
            "skill_card": "quest_travel",
        }

    if target_type == "quest_enemy_marker":
        if proximity is not None and proximity >= 0.66:
            return {
                "scene": "quest_combat",
                "objective": "defeat the confirmed active objective enemy",
                "target": "red objective NPC",
                "skill": "FIGHT_QUEST_TARGET",
                "confidence": 0.92,
                "explanation": "A red objective marker is close enough for combat.",
                "next_after_success": "reobserve quest progress",
                "skill_card": "quest_combat",
            }
        return {
            "scene": "hunt_quest_enemy",
            "objective": "approach the active objective NPC",
            "target": "red objective NPC",
            "skill": "NAVIGATE_OBJECTIVE",
            "confidence": 0.90,
            "explanation": "A red objective marker is visible but not yet close enough.",
            "next_after_success": "fight only at useful range",
            "skill_card": "quest_combat",
        }

    phase = str(quest.get("phase") or "")
    if phase in {"obstacle_recovery", "stuck"}:
        return {
            "scene": "movement_blocked",
            "objective": "recover from the obstacle without repeating the same failed move",
            "target": target_type,
            "skill": "GO_AROUND",
            "confidence": 0.90,
            "explanation": "Progress stalled; prefer lateral recovery over blind climbing.",
            "next_after_success": "resume navigation after position changes",
            "skill_card": "obstacle_recovery",
        }

    return None
