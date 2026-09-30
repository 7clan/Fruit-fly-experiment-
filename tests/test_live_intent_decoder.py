from lab.brain.intent import IntentionDecoder


def test_lateral_mdn_activity_does_not_override_valid_p9_turn():
    sizes = {
        "P9_left": 1, "P9_right": 1,
        "BPN_bilateral": 1, "RRN_bilateral": 1,
        "MDN_bilateral": 2, "MDN_left": 1, "MDN_right": 1,
        "GF_left": 1, "GF_right": 1,
        "FG_bilateral": 1, "BB_bilateral": 1,
    }
    dec = IntentionDecoder(sizes, params={"min_action_chunks": 0})
    out = dec.decode({
        "chunk_id": 1,
        "chunk_ms": 50.0,
        "ts_ns": 1,
        "pop_counts": {
            # 60 Hz P9-right -> valid TURN_RIGHT.
            "P9_right": 3,
            # Diagnostic lateral MDN activity must NOT create RETREAT when
            # the validated MDN_bilateral slot is silent.
            "MDN_left": 1,
            "MDN_bilateral": 0,
        },
    })
    assert out.name == "TURN_RIGHT"
    assert out.dn_rates_hz["MDN_bilateral"] == 0.0
