"""AI-only Ollama Cloud controller.

This branch deliberately removes the fruit-fly brain from the control path.
The cloud model is the sole gameplay decision-maker. Local CV, the semantic
adapter, and the input backend are execution/safety machinery only.
"""

from __future__ import annotations

import base64
import json
import time

from .ollama_cloud import OllamaCloudCoachWorker
from .semantic_coach import (
    ALLOWED_SKILLS, SemanticCoachWorker, _compact,
)


class AIOnlyOllamaCoachWorker(OllamaCloudCoachWorker):
    """Cloud gameplay controller with the full GPO playbook and live state."""

    name = "ai_only_ollama_coach"

    def __init__(self, bus, target_hz: float = 1.5, model: str | None = None,
                 base_url: str | None = None, timeout_s: float = 30.0,
                 api_key: str | None = None):
        super().__init__(
            bus, target_hz=target_hz, model=model, base_url=base_url,
            timeout_s=timeout_s, api_key=api_key)
        # Keep cloud decisions event-driven. The fast local actuator servo
        # continues an already-selected goal; asking the model every few
        # frames only produced duplicate plans and extra cost.
        # The cloud call itself averaged ~2.8 s on the 2026-10-02 run.
        # Keep decisions event-driven and let the local skill executor react
        # at 8-20 Hz. 2 s does not cause a 2 s cadence because unchanged
        # scenes are still held for the longer refresh interval.
        self.min_call_interval_s = 1.0
        self.unchanged_refresh_s = 18.0
        self.provider = "ollama_cloud_ai_only"
        self.skill_state = bus.state("training.skills")
        self.teacher_prior_state = bus.state("training.teacher_priors")
        self.drop_stale_responses = True
        self.suppress_duplicate_plan_logs = True
        # One frame + compact JSON is enough for this controller. Sending the
        # previous frame doubled vision work while local CV already measures
        # progress/stuck state.
        # 420 tokens truncated 13/63 calls in lab_20261004_005047.
        # Give the compact structured controller enough room to finish once
        # instead of triggering a slower retry that can fail at the transport.
        self.max_output_tokens = 640
        self.stats.update({
            "provider": self.provider,
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "ai_only": True,
        })

    def _call_gemini(self, prompt: str, image_b64: str | None,
                     previous_image_b64: str | None = None) -> dict:
        # Latency path: one CURRENT frame only. Temporal progress is supplied
        # by quest.state (stuck/circling/action epochs), so a second image is
        # redundant and expensive.
        return super()._call_gemini(
            prompt, image_b64, previous_image_b64=None)

    def _signature(self, obs: dict, quest: dict, brain: dict) -> tuple:
        """Semantic replan signature, not a motor-feedback signature.

        The 2026-10-03 run spent 38/66 cloud calls on responses that became
        "stale" merely because local execution changed action_epoch or a
        marker flickered while Gemini was thinking. Motor geometry is handled
        locally at 8-20 Hz. Wake the cloud for genuine state transitions.
        """
        player = obs.get("player") or {}
        try:
            health_bucket = int(max(
                0.0, min(1.0, float(player.get("health")))) * 4)
        except (TypeError, ValueError):
            health_bucket = -1
        actor_recent = bool(
            quest.get("quest_enemy_actor_recent",
                      quest.get("quest_enemy_actor_visible")))
        return (
            str(quest.get("quest_status") or quest.get("phase") or "unknown"),
            actor_recent,
            bool(quest.get("awaiting_quest_confirmation")),
            bool(quest.get("stuck")),
            bool(quest.get("circling")),
            int(quest.get("ui_epoch") or 0),
            int(quest.get("visual_nav_epoch") or 0),
            health_bucket,
        )

    def _should_drop_stale_response(
            self, plan: dict, start_sig: tuple, current_sig: tuple) -> bool:
        """Keep persistent goal plans despite cloud latency.

        Navigation/fight/recovery skills are revalidated against fresh local
        perception before every key or mouse event, so they remain safe/useful
        even if the scene changed during a 1-3 s cloud call. UI/equipment/world
        interactions can become dangerous when stale and are still dropped.
        """
        if current_sig == start_sig:
            return False
        skill = str(plan.get("skill") or "").upper()
        persistent = {
            "WAIT", "REOBSERVE", "NAVIGATE_OBJECTIVE", "TRAVEL",
            "FIGHT_QUEST_TARGET", "BLOCK", "EVADE", "JUMP", "CLIMB",
            "GO_AROUND", "BACKTRACK", "SPRINT", "GEPPO",
            "LOOK_LEFT", "LOOK_RIGHT",
        }
        return skill not in persistent

    def _plan_response_schema(self) -> dict:
        """Require minimal grounding fields in AI-only mode.

        The prior run had FIGHT plans that claimed an enemy was visible while
        visual_target was "none", leaving the fast tracker with zero
        initializations. Requiring these small objects makes the plan
        executable between cloud replies without adding a second model call.
        """
        schema = super()._plan_response_schema()
        required = list(schema.get("required") or [])
        for key in ("perception", "visual_target"):
            if key not in required:
                required.append(key)
        schema["required"] = required

        perception = schema["properties"]["perception"]
        perception["required"] = [
            "quest_state", "quest_hud_visible", "dialogue_visible",
            "interaction_prompt_visible", "enemy_actor_visible",
            "player_dead", "safezone_visible",
            "equipped_slot_visible", "melee_ready_visible",
        ]
        perception["properties"]["quest_hud_visible"] = {
            "type": "boolean"
        }
        perception["properties"]["quest_progress_text"] = {
            "type": "string"
        }
        perception["properties"]["equipped_style_visible"] = {
            "type": "string"
        }
        visual = schema["properties"]["visual_target"]
        visual["required"] = [
            "kind", "x_norm", "y_norm", "bbox_norm",
            "confidence", "melee_ready",
        ]
        schema["properties"]["action_channels"] = {
            "type": "object",
            "properties": {
                "locomotion": {
                    "type": "string",
                    "enum": [
                        "none", "approach", "orbit_left", "orbit_right",
                        "retreat", "hold", "sprint", "jump", "climb",
                    ],
                },
                "offense": {
                    "type": "string",
                    "enum": ["none", "m1", "observed_ability"],
                },
                "defense": {
                    "type": "string",
                    "enum": ["none", "guard_between_attacks", "evade"],
                },
                "camera": {
                    "type": "string",
                    "enum": ["none", "track_target", "look_left", "look_right"],
                },
                "equip_control_id": {"type": "string"},
            },
            "required": [
                "locomotion", "offense", "defense",
                "camera", "equip_control_id",
            ],
        }
        if "action_channels" not in required:
            required.append("action_channels")
        schema["required"] = required
        return schema

    def _jpeg_b64(self, frame) -> tuple[str | None, float]:
        """Adaptive cloud frame.

        Navigation/combat need fast scene understanding more than tiny text, so
        use a smaller frame there. Quest dialogue/shop/hotbar interaction needs
        fine UI text, so temporarily send the 960 px frame. This directly cuts
        cloud vision latency without making button coordinates less accurate
        when accuracy matters.
        """
        if frame is None:
            return None, 0.0
        t0 = time.perf_counter()
        try:
            import cv2
            quest = self._state(self.quest_state)
            prev = self._state(self.plan_state)
            prev_skill = str(prev.get("skill") or "").upper()
            detail_mode = bool(
                quest.get("awaiting_quest_confirmation")
                or str(quest.get("quest_status") or "") == "pending_accept"
                or prev_skill in {
                    "TAKE_QUEST", "INTERACT", "UI_CLICK",
                    "BUY_ITEM", "EQUIP_SLOT", "USE_OBSERVED_ABILITY",
                })
            # Navigation needs scene layout, not tiny UI text. Dialogue/shop
            # pages get a larger frame only while they are actually active.
            max_w = 800 if detail_mode else 576
            quality = 70 if detail_mode else 60
            img = frame
            h, w = img.shape[:2]
            if w > max_w:
                scale = max_w / float(w)
                img = cv2.resize(
                    img, (max_w, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return None, 0.0
            data = base64.b64encode(enc.tobytes()).decode("ascii")
            return data, (time.perf_counter() - t0) * 1000.0
        except Exception:
            return None, (time.perf_counter() - t0) * 1000.0

    def _prompt(self, obs: dict, quest: dict, brain: dict, catalog: dict,
                action_meta: dict, previous_plan: dict,
                retrieved_context: str = "") -> str:
        controls = [{
            "control_id": c.get("control_id"),
            "name": c.get("name"),
            "category": c.get("category"),
            "pattern": c.get("pattern"),
            "context": c.get("context"),
            "notes": c.get("notes"),
        } for c in (catalog.get("controls") or []) if c.get("control_id")]

        target = obs.get("target") or {}
        player = obs.get("player") or {}
        notes = obs.get("notes") or {}
        # Avoid the misleading legacy field name "distance": in this CV
        # contract the value is actually PROXIMITY (0=far, 1=close).
        world_for_ai = {
            "player": {
                "health_fraction": player.get("health"),
                "stamina_fraction": player.get("stamina"),
            },
            "target": {
                "type": target.get("type"),
                "direction_radians": target.get("direction"),
                "proximity_0_far_1_close": target.get("distance"),
                "confidence": target.get("confidence"),
            },
            "cues": {
                "yellow_quest_giver": bool(notes.get("quest_marker_detected")),
                "green_recommended_waypoint": bool(
                    notes.get("recommended_waypoint_detected")),
                "red_quest_objective": bool(
                    notes.get("quest_enemy_marker_detected")),
                "quest_enemy_actor_visible": bool(
                    notes.get("quest_enemy_actor_visible")),
                "quest_enemy_actor": (
                    {
                        "track_id": (
                            (notes.get("quest_enemy_actor") or {})
                            .get("track_id")),
                        "direction_radians": (
                            (notes.get("quest_enemy_actor") or {})
                            .get("direction")),
                        "proximity_0_far_1_close": (
                            (notes.get("quest_enemy_actor") or {})
                            .get("distance")),
                        "confidence": (
                            (notes.get("quest_enemy_actor") or {})
                            .get("confidence")),
                    }
                    if notes.get("quest_enemy_actor") else {}
                ),
            },
        }
        skill_env = self.skill_state.read()
        learned_skills = (
            (skill_env.payload or {}).get("skills", {})
            if skill_env is not None else {})
        compact_skills = {}
        for sid, rec in learned_skills.items():
            if not isinstance(rec, dict):
                continue
            compact_skills[str(sid)] = {
                "category": rec.get("category"),
                "attempts": rec.get("attempts"),
                "positive": rec.get("positive_outcomes"),
                "negative": rec.get("negative_outcomes"),
                "neutral": rec.get("neutral_outcomes"),
                "last_reward": rec.get("last_reward"),
                "preconditions": rec.get("preconditions"),
                "success_conditions": rec.get("success_conditions"),
                "failure_conditions": rec.get("failure_conditions"),
            }

        teacher_env = self.teacher_prior_state.read()
        teacher_priors = (
            teacher_env.payload if teacher_env is not None else {})

        situation = {
            "world": _compact(world_for_ai, 2600),
            "quest": _compact(quest, 2200),
            "agent": {
                "enabled": bool(action_meta.get("autonomy")),
                "mode": "AI_ONLY",
                "loadout": action_meta.get("gpo_loadout"),
            },
            "verified_controls": controls,
            "learned_skill_library": _compact(compact_skills, 2600),
            "human_teacher_priors": _compact(teacher_priors, 1800),
            "previous_plan": _compact(previous_plan, 1300),
            "persistent_character_profile": _compact(self.profile, 2400),
            "state_contract": {
                "proximity": "0 means far; 1 means close/melee",
                "quest_status": (
                    "sticky perception state; active beats visible quest giver"),
                "quest_enemy_marker": (
                    "objective/location cue; NOT proof an enemy body is in melee"),
                "quest_enemy_actor_visible": (
                    "tracked hostile NPC body associated with active quest"),
                "stuck_or_circling": (
                    "closed-loop progress monitor; recover before repeating navigation"),
                "last_equipped_slot": (
                    "last hotbar slot the agent physically selected"),
                "action_epoch": (
                    "increments after one-shot physical actions so you can "
                    "verify the result on the next frame"),
                "learned_skill_library": (
                    "persistent measured outcomes from previous sessions; "
                    "use it as experience, not as current visual truth"),
            },
        }
        situation_text = json.dumps(situation, default=str)
        # Keep the full playbook on disk, but send only relevant sections on
        # each live frame. This cuts thousands of repeated prompt tokens and
        # lowers cloud latency without deleting any available knowledge.
        playbook = self._relevant_playbook(situation_text)
        if not playbook:
            playbook = self.knowledge[:12000]
        if retrieved_context:
            playbook += "\n\nCURRENT GPO WIKI CONTEXT:\n" + retrieved_context[:8000]
        skills = ", ".join(sorted(ALLOWED_SKILLS))

        return f"""You are the SOLE GAMEPLAY CONTROLLER for an autonomous
Grand Piece Online research run. There is NO fruit-fly controller in this
branch. Every gameplay decision comes from you. Local computer vision only
measures the screen, the adapter converts your chosen skill into verified
controls, and F12 remains a human emergency stop.

GOAL:
Play normally and make useful progression. Take appropriate quests, navigate
to objectives, fight quest NPCs, equip the needed hotbar item/style, defend,
use observed abilities, buy ordinary in-game progression items when sensible,
travel by ship when needed, and update the persistent character profile.

CONTROL DISCIPLINE:
- Choose exactly ONE skill per response. Treat that skill as a PERSISTENT
  behavior/macro, not a one-frame keypress. Do not restate the same objective
  because its coordinates moved. The local servo keeps executing the selected
  behavior against fresh CV while this cloud call is sleeping. Replan only
  after a semantic state change, failure/stuck signal, UI page change, or goal
  completion.
- Never output raw keys. EXEC_CONTROL must use a control_id from
  verified_controls.
- Read quest state in this priority order:
    1) A visible top-left quest HUD such as "Defeat X 1/8", its progress bar,
       Rewards and QUIT button is authoritative evidence that a quest is ACTIVE.
       Set perception.quest_state="active" even if the NPC body is behind a wall.
    2) quest_status=active -> DO NOT take another quest. Follow/fight its target.
    3) quest_status=pending_accept -> wait for confirmation/dialogue change.
    4) quest_status=completed -> locate/choose the next appropriate quest.
    5) quest_status=available -> approach the yellow quest giver and interact.
    6) travel_to_quest_giver -> follow the green recommended waypoint.
    "Recommended Quest" alone is navigation guidance, not proof of acceptance.
- The red quest marker/dot above a quest enemy is persistent target identity
  and stays useful when the NPC body is occluded by a wall. Use it to PURSUE
  the correct quest enemy through occlusion. Do not confuse it with red UI.
  Start M1 when the body is grounded/melee-ready OR the active-quest red target
  is very close and centered. If the red target is close but blocked, keep the
  fight goal and choose jump/climb/orbit instead of claiming no enemy exists.
- SAFEZONE / PROTECTED text is PvP protection in this experiment. It is NOT a
  reason to avoid or postpone fighting quest NPCs. Never invent a need to leave
  the safe zone before PvE.
- TAKE_QUEST: move to the yellow QUEST/! giver. Whenever the giver is visible,
  visual_target.kind MUST be "quest_giver" with a tight bbox_norm around the
  NPC/interaction target. If the CURRENT screenshot visibly shows the
  T/Interact prompt, TAKE_QUEST itself is enough: the actuator presses T
  immediately even if geometric proximity is noisy.
  Otherwise it approaches until ready, presses T once, then waits. If a quest
  dialogue/Accept/Yes button is visible, fill ui_click AND preferably ground
  that same button as visual_target kind="ui". Do not repeatedly press T while
  awaiting_quest_confirmation=true.
- FIGHT_QUEST_TARGET is a MULTI-CHANNEL combat policy. Fill action_channels:
    locomotion=approach while far, orbit_left/orbit_right when close;
    offense=m1 for the normal melee combo;
    defense=guard_between_attacks for ordinary melee pressure or evade if needed;
    camera=track_target during combat;
    equip_control_id=the verified combat hotbar slot when needed, else "".
  The executor runs movement + attacks + guarding concurrently/interleaved while
  this cloud plan stays current. Do not attack ordinary players.
- Whenever you can visually identify the object/person you are acting on, fill
  visual_target with its CURRENT normalized screen center, confidence, AND a
  tight normalized bounding box [x1,y1,x2,y2]. For a quest NPC enemy use
  kind="quest_enemy_actor". For a giver use kind="quest_giver". Set
  melee_ready=true only when the actual NPC body is visibly close enough to
  hit now. The local visual tracker follows YOUR selected box between cloud
  replies; it does not choose a target by itself.
- visual_target kind="waypoint" is SHORT-HORIZON visual grounding, not an
  imaginary objective marker. Box a real visible navigable cue/opening/path
  you intend to walk toward. Never place a waypoint box on a wall merely
  because the quest objective is somewhere behind that wall. If no safe
  visible path point is grounded, choose LOOK_LEFT/LOOK_RIGHT, GO_AROUND,
  BACKTRACK or REOBSERVE instead of fabricating a waypoint.
- EQUIPMENT FOR COMBAT: inspect the CURRENT hotbar/hand/ability HUD. Report
  the visibly selected hotbar slot in perception.equipped_slot_visible. A
  selected fist/melee slot plus visible melee move HUD means the style is ready;
  do not toggle the same slot repeatedly. Put a needed equip_slot_N in
  action_channels.equip_control_id so it can equip inside the same fight plan.
  Set perception.melee_ready_visible=true only when visibly ready.
  * If you only need to equip, choose EQUIP_SLOT with verified equip_slot_N.
  * If the quest enemy is already present and you know the needed slot, you
    may choose FIGHT_QUEST_TARGET AND set control_id="equip_slot_N". The local
    macro will equip that exact AI-selected slot once, then continue YOUR
    fight plan immediately without waiting another cloud round-trip.
  state.last_equipped_slot records the last slot physically selected.
- USE_OBSERVED_ABILITY requires BOTH a move label and binding visible on the
  CURRENT HUD. Never guess fruit/style/sword ability keys.
- Verified movement mechanics: SPACE jumps; CTRL climbs while contacting a
  wall/object (the actuator combines forward+CTRL for CLIMB); double-W sprints;
  Q with a direction rolls/dashes; repeated airborne SPACE is GEPPO if unlocked.
- If stuck=true or circling=true during NAVIGATION, do NOT repeat
  NAVIGATE_OBJECTIVE unchanged. Choose a recovery that matches visible geometry.
  JUMP requires a visible low obstacle; CLIMB requires a visible wall/climb
  surface. Otherwise prefer LOOK_LEFT/RIGHT, GO_AROUND or BACKTRACK.
- During FIGHT_QUEST_TARGET, constant distance while orbiting a visible enemy
  is normal combat, not "stuck". Do NOT emit generic JUMP/CLIMB recovery in
  melee unless the CURRENT screenshot clearly shows the corresponding obstacle.
  If the fight target is temporarily lost, prioritize camera reacquisition,
  target-centering and bounded repositioning, then resume offense only after
  the target is grounded again.
- Treat learned_skill_library as measured past experience. When current visual
  preconditions match, prefer skills with repeated positive outcomes and avoid
  repeatedly retrying skills with negative outcomes. Current screenshot/state
  always overrides old experience.
- human_teacher_priors are descriptive imitation statistics from recorded
  human play, not reward. Use timing/coverage only when its readiness flag is
  true. Missing BLOCK/EVADE coverage means "unknown", never "defense is bad".
- BLOCK/EVADE are your decisions; no hidden combat policy chooses them.
- LOOK_LEFT/LOOK_RIGHT are explicit camera actions for searching/recentering.
- UI_CLICK/BUY_ITEM may click clearly visible ordinary in-game quest/menu/shop
  buttons at confidence >= 0.80. For any click, provide BOTH ui_click normalized
  center and visual_target kind="ui" with a tight bbox_norm around that exact
  button. If you cannot ground the button, do not click it. Never confirm Robux, premium/gamepass,
  account/security, external-link, or trade UI.
- Use CURRENT screenshot/HUD + structured state as truth over static knowledge.
- If the screenshot does not support a claim, do not invent it. Choose
  REOBSERVE/WAIT instead.
- When useful, request a short GPO wiki query; after retrieval make the final
  decision instead of requesting another lookup.

GPO PLAYBOOK / STRATEGY KNOWLEDGE:
{playbook}

CURRENT STATE:
{situation_text}

Return one compact JSON decision matching the API schema.
REQUIRED: scene, objective, target, skill, confidence, explanation.
Keep scene/objective/target/explanation short.

REQUIRED in AI-only mode:
- action_channels with locomotion/offense/defense/camera/equip_control_id.
  Use several non-"none" channels when simultaneous behavior is useful.

OPTIONAL — include only when useful for THIS action:
- control_id for EXEC_CONTROL/EQUIP_SLOT/Haki.
- observed_ability only for a move visibly read from the current HUD.
- visual_target when you can ground the chosen quest giver/enemy/UI target.
- ui_click only for a clearly visible in-game button.
- perception only for facts you can actually see now. In particular,
  quest_hud_visible must mean the CURRENT frame actually shows the accepted
  quest/progress HUD, and quest_progress_text should copy only the short
  visible objective/progress phrase (for example "Defeat Corrupt Marines 1/8").
- knowledge_query only when current GPO detail is genuinely missing.
- memory_updates only for high-confidence persistent facts.

skill must be one of: {skills}
Do not pad the response with empty/default objects. Do not repeat the rules.
"""


    def _validate_plan(self, raw: dict, catalog: dict) -> dict:
        plan = super()._validate_plan(raw, catalog)
        p = raw.get("perception") or {}
        allowed_quest = {
            "active", "available", "pending_accept", "completed", "unknown"
        }
        qstate = str(p.get("quest_state") or "unknown").strip().lower()
        if qstate not in allowed_quest:
            qstate = "unknown"
        plan["perception"] = {
            "quest_state": qstate,
            "quest_hud_visible": bool(p.get("quest_hud_visible")),
            "quest_progress_text": str(
                p.get("quest_progress_text") or "")[:96],
            "dialogue_visible": bool(p.get("dialogue_visible")),
            "interaction_prompt_visible": bool(
                p.get("interaction_prompt_visible")),
            "enemy_actor_visible": bool(p.get("enemy_actor_visible")),
            "player_dead": bool(p.get("player_dead")),
            "safezone_visible": bool(p.get("safezone_visible")),
            "equipped_slot_visible": (
                str(p.get("equipped_slot_visible") or "unknown").strip()),
            "equipped_style_visible": str(
                p.get("equipped_style_visible") or "unknown")[:96],
            "melee_ready_visible": bool(p.get("melee_ready_visible")),
        }

        vt = raw.get("visual_target") or {}
        allowed_kinds = {
            "none", "quest_giver", "quest_enemy_actor",
            "quest_objective", "waypoint", "ui",
        }
        kind = str(vt.get("kind") or "none").strip().lower()
        if kind not in allowed_kinds:
            kind = "none"
        try:
            vx = max(0.0, min(1.0, float(vt.get("x_norm", 0.5))))
            vy = max(0.0, min(1.0, float(vt.get("y_norm", 0.5))))
            vc = max(0.0, min(1.0, float(vt.get("confidence", 0.0))))
        except (TypeError, ValueError):
            vx, vy, vc = 0.5, 0.5, 0.0

        bbox = vt.get("bbox_norm")
        clean_bbox = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [
                    max(0.0, min(1.0, float(v))) for v in bbox
                ]
                if x2 - x1 >= 0.025 and y2 - y1 >= 0.035:
                    clean_bbox = [
                        round(x1, 4), round(y1, 4),
                        round(x2, 4), round(y2, 4),
                    ]
            except (TypeError, ValueError):
                clean_bbox = None
        if vc < 0.65:
            kind = "none"
        plan["visual_target"] = {
            "kind": kind,
            "x_norm": round(vx, 4),
            "y_norm": round(vy, 4),
            "bbox_norm": clean_bbox,
            "confidence": round(vc, 4),
            "melee_ready": bool(vt.get("melee_ready")) and vc >= 0.82,
        }

        channels = raw.get("action_channels") or {}
        locomotion = str(
            channels.get("locomotion") or "none").strip().lower()
        offense = str(
            channels.get("offense") or "none").strip().lower()
        defense = str(
            channels.get("defense") or "none").strip().lower()
        camera = str(
            channels.get("camera") or "none").strip().lower()
        equip_id = str(channels.get("equip_control_id") or "").strip()
        if locomotion not in {
                "none", "approach", "orbit_left", "orbit_right",
                "retreat", "hold", "sprint", "jump", "climb"}:
            locomotion = "none"
        if offense not in {"none", "m1", "observed_ability"}:
            offense = "none"
        if defense not in {
                "none", "guard_between_attacks", "evade"}:
            defense = "none"
        if camera not in {
                "none", "track_target", "look_left", "look_right"}:
            camera = "none"
        valid_ids = {
            str(x.get("control_id"))
            for x in (catalog.get("controls") or [])
            if x.get("control_id")
        }
        if equip_id not in valid_ids or not equip_id.startswith("equip_slot_"):
            equip_id = ""
        plan["action_channels"] = {
            "locomotion": locomotion,
            "offense": offense,
            "defense": defense,
            "camera": camera,
            "equip_control_id": equip_id,
        }
        return plan


class AIOnlyGeminiCoachWorker(AIOnlyOllamaCoachWorker):
    """AI-only controller using Gemini Flash-Lite multimodal transport.

    Reuses the exact AI-only prompt/validation/event cadence above, but sends
    frames to Google's low-latency multimodal endpoint instead of Ollama
    Cloud. This is the preferred path on the target laptop because inference
    stays off-device and the model is substantially smaller/faster than the
    31B Ollama fallback that caused multi-second control lag.
    """

    name = "ai_only_gemini_coach"

    def __init__(self, bus, target_hz: float = 1.5,
                 model: str | None = None,
                 base_url: str | None = None,
                 timeout_s: float = 16.0,
                 api_key: str | None = None):
        # Call the Gemini base initializer directly so Ollama-specific auth,
        # model defaults, and transport are never activated.
        SemanticCoachWorker.__init__(
            self, bus,
            target_hz=target_hz,
            model=model,
            min_call_interval_s=1.0,
            unchanged_refresh_s=18.0,
            timeout_s=timeout_s,
            api_key=api_key,
        )
        # AIOnlyOllamaCoachWorker._prompt() uses the compact relevant-section
        # selector provided by OllamaCloudCoachWorker; initialize the same
        # cached section index without using its network transport.
        self._sections = self._split_sections(self.knowledge)
        self.min_call_interval_s = 1.0
        self.unchanged_refresh_s = 18.0
        self.provider = "gemini_ai_only"
        self.skill_state = bus.state("training.skills")
        self.teacher_prior_state = bus.state("training.teacher_priors")
        self.drop_stale_responses = True
        self.suppress_duplicate_plan_logs = True
        # Live run lab_20261004_005047 truncated 13/63 responses at 420.
        # Match the AI-only controller budget used by the Ollama-derived path.
        self.max_output_tokens = 640
        self.stats.update({
            "provider": self.provider,
            "model": self.model,
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "ai_only": True,
            "supports_vision": True,
            "cloud_ai_used": True,
        })

    def _call_gemini(self, prompt: str, image_b64: str | None,
                     previous_image_b64: str | None = None) -> dict:
        # One CURRENT frame only; local CV and quest.state provide temporal
        # feedback between cloud calls.
        return SemanticCoachWorker._call_gemini(
            self, prompt, image_b64, previous_image_b64=None)
