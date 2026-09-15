"""Tests for the template-based reply generator."""

from utils.reply_generator import extract_issue_keywords, generate_reply, REPLY_TEMPLATES


def test_amount_is_extracted_into_reply():
    detail = extract_issue_keywords("I was charged $49.99 twice")
    assert "regarding the charge of $49.99" in detail


def test_error_code_and_order_id_extracted():
    detail = extract_issue_keywords("I get E-500 when uploading. Order #789012")
    assert "related to error E-500" in detail
    assert "for reference #789012" in detail


def test_no_keywords_yields_empty_detail():
    assert extract_issue_keywords("how do your plans work") == ""


def test_urgent_priority_adds_prefix_and_suffix():
    reply = generate_reply("Someone hacked my account", "Account Access", "Urgent")
    assert reply.startswith("[!!] PRIORITY")
    assert "escalated to our senior support team" in reply


def test_low_priority_has_no_escalation_markers():
    reply = generate_reply("What are your plans?", "General Inquiry", "Low")
    assert not reply.startswith("[!!]")
    assert ">> " not in reply


def test_category_templates_are_used():
    reply = generate_reply("please refund $9.99", "Refund Request", "Medium")
    assert "refund" in reply.lower()


def test_unknown_category_falls_back_to_general_inquiry():
    text = "hello there"
    reply = generate_reply(text, "Not A Real Category", "Medium")
    templates = REPLY_TEMPLATES["General Inquiry"]
    expected = templates[len(text) % len(templates)].format(issue_detail="")
    assert reply == expected
