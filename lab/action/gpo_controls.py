"""GPO control catalog used by the engineered action layer.

The fly brain chooses behavioral intentions. This module describes the game
controls that may implement those intentions. It deliberately separates:
  * verified/common controls with stable bindings;
  * contextual controls whose effect depends on player state;
  * equipped abilities, whose names/hotkeys vary by current loadout and are
    therefore learned from the live HUD instead of hard-coded.

Nothing in this module enables autonomy by itself. MotorExecutor/backend gates
still decide which subset is legal in the current experimental stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class GPOControl:
    control_id: str
    name: str
    category: str
    bindings: tuple[str, ...]
    pattern: str = "tap"           # tap | hold | chord | double_tap | repeat
    context: str = ""
    provenance: str = "community_verified"
    confidence: float = 1.0
    autonomous_ready: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bindings"] = list(self.bindings)
        return d


# Controls supported by the Windows backend. Some keys have multiple meanings
# depending on the equipped fighting style/item; the catalog never assumes a
# move name from a key alone.
SUPPORTED_GAME_KEYS = frozenset({
    *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    *"0123456789",
    "SPACE", "CTRL", "SHIFT", "ALT", "TAB", "ESC",
})
SUPPORTED_MOUSE_BUTTONS = frozenset({"left", "right", "middle"})


CORE_CONTROLS: tuple[GPOControl, ...] = (
    # locomotion
    GPOControl("move_forward", "Move forward", "movement", ("key:W",),
               pattern="hold", autonomous_ready=True),
    GPOControl("move_backward", "Move backward", "movement", ("key:S",),
               pattern="hold"),
    GPOControl("move_left", "Move/strafe left", "movement", ("key:A",),
               pattern="hold", autonomous_ready=True),
    GPOControl("move_right", "Move/strafe right", "movement", ("key:D",),
               pattern="hold", autonomous_ready=True),
    GPOControl("jump", "Jump", "movement", ("key:SPACE",), pattern="tap"),
    GPOControl("sprint", "Sprint", "movement", ("key:W",),
               pattern="double_tap",
               notes="GPO control reference: double-tap W."),
    GPOControl("dash_forward", "Dash/roll forward", "movement",
               ("key:W", "key:Q"), pattern="chord"),
    GPOControl("dash_backward", "Dash/roll backward", "movement",
               ("key:S", "key:Q"), pattern="chord"),
    GPOControl("dash_left", "Dash/roll left", "movement",
               ("key:A", "key:Q"), pattern="chord"),
    GPOControl("dash_right", "Dash/roll right", "movement",
               ("key:D", "key:Q"), pattern="chord"),
    GPOControl("climb_or_dive", "Climb / dive", "movement",
               ("key:CTRL",), pattern="hold",
               context="wall contact or water"),
    GPOControl("geppo", "Geppo / Sky Walk", "movement",
               ("key:SPACE",), pattern="repeat",
               context="airborne and ability unlocked",
               notes="Repeated airborne Space; availability is character-dependent."),

    # combat fundamentals
    GPOControl("basic_attack", "Basic attack / M1", "combat",
               ("mouse:left",), pattern="tap",
               notes="Common GPO basic attack; combo timing is contextual."),
    GPOControl("m1_string", "M1 combo string", "combat",
               ("mouse:left",), pattern="repeat",
               context="target in melee range",
               notes="Repeated M1 clicks; exact string length depends on timing/context."),
    GPOControl("uptilt_air_combo", "Uptilt / air-combo launcher", "combat",
               ("mouse:left", "key:SPACE"), pattern="chord",
               context="during the M1 string",
               notes="Hold Space during the M1 sequence to launch into the air combo."),
    GPOControl("block", "Block", "combat", ("key:F",), pattern="hold",
               notes="Holding F blocks while block is available."),
    GPOControl("perfect_block", "Perfect block / parry", "combat",
               ("key:F",), pattern="tap",
               context="precise timing immediately before an incoming hit",
               notes="Same F binding as block; timing changes the outcome."),
    GPOControl("reload", "Reload firearm", "combat", ("key:R",),
               pattern="tap", context="gun equipped"),

    # world / utility
    GPOControl("interact", "Interact / talk", "world", ("key:T",),
               pattern="tap", provenance="community_verified",
               confidence=1.0, autonomous_ready=True,
               notes="Current GPO community wiki: T interacts with NPCs/quests."),
    GPOControl("carry_downed", "Carry downed player", "world", ("key:V",),
               pattern="tap", context="downed player nearby"),
    GPOControl("grip_downed", "Grip/execute downed player", "combat",
               ("key:B",), pattern="tap", context="downed target nearby"),
    GPOControl("sit_ship", "Sit / attach to ship", "world", ("key:P",),
               pattern="tap", context="ship/seat nearby"),
    GPOControl("menu", "Open menu", "ui", ("key:M",), pattern="tap"),
    GPOControl("buso_haki", "Toggle Busoshoku Haki", "combat",
               ("key:J",), pattern="tap", context="Haki acquired"),
    GPOControl("observation_haki", "Toggle Observation Haki", "combat",
               ("key:G",), pattern="tap", context="Haki acquired"),

    GPOControl("evasive_contextual", "Evasive while stunned", "combat",
               ("key:CTRL",), pattern="tap",
               context="stunned and evasive ready",
               provenance="community_verified", confidence=1.0,
               notes="Current GPO controls reference: Left CTRL while stunned."),

    # camera / equipment
    GPOControl("camera_look", "Rotate camera", "camera",
               ("mouse:right",), pattern="hold",
               context="hold RMB while moving mouse",
               provenance="community_verified", confidence=0.9,
               autonomous_ready=True),
    GPOControl("equip_slot_1", "Equip hotbar slot 1", "equipment",
               ("key:1",), pattern="tap"),
    GPOControl("equip_slot_2", "Equip hotbar slot 2", "equipment",
               ("key:2",), pattern="tap"),
    GPOControl("equip_slot_3", "Equip hotbar slot 3", "equipment",
               ("key:3",), pattern="tap"),
    GPOControl("equip_slot_4", "Equip hotbar slot 4", "equipment",
               ("key:4",), pattern="tap"),
    GPOControl("equip_slot_5", "Equip hotbar slot 5", "equipment",
               ("key:5",), pattern="tap"),
    GPOControl("equip_slot_6", "Equip hotbar slot 6", "equipment",
               ("key:6",), pattern="tap"),
    GPOControl("equip_slot_7", "Equip hotbar slot 7", "equipment",
               ("key:7",), pattern="tap"),
    GPOControl("equip_slot_8", "Equip hotbar slot 8", "equipment",
               ("key:8",), pattern="tap"),
    GPOControl("equip_slot_9", "Equip hotbar slot 9", "equipment",
               ("key:9",), pattern="tap"),
    GPOControl("equip_slot_0", "Equip hotbar slot 0", "equipment",
               ("key:0",), pattern="tap"),
)


# Examples from current/legacy fighting styles show ability hotkeys such as
# E/R/Z/X/C/N/Q. Rather than hard-code every fruit/style/sword move (which
# changes with loadout and updates), the action layer accepts any observed HUD
# binding that the backend can physically emit.
def binding_supported(binding: str) -> bool:
    if not isinstance(binding, str):
        return False
    if binding.startswith("key:"):
        return binding[4:].upper() in SUPPORTED_GAME_KEYS
    if binding.startswith("mouse:"):
        return binding[6:].lower() in SUPPORTED_MOUSE_BUTTONS
    return False


def validate_observed_ability_binding(binding: str) -> str:
    """Validate and normalize a move binding learned from the live HUD."""
    if not binding_supported(binding):
        raise ValueError(f"unsupported GPO ability binding {binding!r}")
    if binding.startswith("key:"):
        return "key:" + binding[4:].upper()
    return "mouse:" + binding[6:].lower()


def control_map() -> dict[str, GPOControl]:
    return {c.control_id: c for c in CORE_CONTROLS}


def controls_by_category(category: str) -> list[GPOControl]:
    return [c for c in CORE_CONTROLS if c.category == category]


# Current starter/default Melee moves confirmed both by the live HUD supplied
# in this project and the current GPO wiki. These are a LOADOUT PROFILE, not a
# claim that E/R always mean the same move after the user changes style/item.
CURRENT_DEFAULT_MELEE: tuple[GPOControl, ...] = (
    GPOControl(
        "melee_gut_punch", "Gut Punch", "ability", ("key:E",),
        pattern="tap", context="default Melee equipped; close range",
        provenance="live_plus_community_verified", confidence=1.0,
        notes="10 mastery; 20 stamina; ~15 s cooldown; block-break."),
    GPOControl(
        "melee_ground_smash", "Ground Smash", "ability", ("key:R",),
        pattern="tap", context="default Melee equipped; close range",
        provenance="live_plus_community_verified", confidence=1.0,
        notes="25 mastery; 25 stamina; ~20 s cooldown; blockable."),
)

# Equipped fruits/styles/swords use loadout-specific skill keys. The action
# backend supports these common observed skill keys, but autonomy must only
# bind a named move after the current HUD/loadout confirms it.
OBSERVED_ABILITY_KEYS = frozenset({
    "E", "R", "Z", "X", "C", "V", "B", "N", "Q", "F", "G", "J",
})


def loadout_controls(loadout: str) -> list[GPOControl]:
    loadout = str(loadout or "").strip().lower()
    if loadout in {"default_melee", "melee", "combat"}:
        return list(CURRENT_DEFAULT_MELEE)
    return []
