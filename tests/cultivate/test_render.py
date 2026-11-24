from orchardlogic.cultivate.render import render_text, tool_lines


def test_shared_descriptions_are_written_once_with_full_ids():
    lines = tool_lines(
        [
            {"tool_id": "amazon.show_profile", "description": "Show a profile."},
            {"tool_id": "venmo.login", "description": "Log in."},
            {"tool_id": "gmail.show_profile", "description": "Show a profile."},
        ]
    )
    assert lines == [
        "- Show a profile.",
        "    amazon.show_profile",
        "    gmail.show_profile",
        "- venmo.login: Log in.",
    ]


def test_explore_drops_rarely_used_fields_and_keeps_cultivar():
    text = render_text(
        "explore",
        {
            "session_id": "S-1",
            "catalog_id": "cat",
            "cohort_id": "C1",
            "results": [
                {
                    "ref": "D3",
                    "label": "Payments",
                    "summary": "Send money.",
                    "tools": 2,
                    "retained_pct": 100.0,
                    "restricted": False,
                    "cultivar": "domain",
                    "groups": [
                        {"ref": "D9", "label": "Requests", "summary": "", "tools": 1,
                         "retained_pct": 100.0, "restricted": False}
                    ],
                    "tool_cards": [{"tool_id": "venmo.create_transaction", "description": "Send."}],
                }
            ],
        },
    )
    assert "retained_pct" not in text and "restricted" not in text and "S-1" not in text
    assert "## D3 Payments (2 tools, domain)" in text
    assert "- D9 Requests (1)" in text
    assert "- venmo.create_transaction: Send." in text


def test_errors_and_unknown_commands():
    assert render_text("explore", {"error": {"code": "unknown_ref", "message": "no"}}) == (
        "Error (unknown_ref): no"
    )
    assert render_text("schemas", {"schemas": [{"a": 1}]}) == '{"schemas":[{"a":1}]}'

