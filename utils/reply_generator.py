"""
Auto-Reply Generator (Template + Rule-Based)
==============================================
Generates customer support replies without any external API calls.

Design approach:
- Category-specific reply templates with keyword-aware customization
- Priority-aware tone adjustment (Urgent -> more empathetic, Low -> standard)
- Keyword extraction from ticket text for personalized responses
- No LLM, no API calls -- purely deterministic template logic
"""

import re
import random

random.seed(42)

REPLY_TEMPLATES = {
    "Billing Issue": [
        (
            "Dear Customer,\n\n"
            "Thank you for reaching out regarding your billing concern. "
            "We understand how important accurate billing is to you.\n\n"
            "We have reviewed the details of your case{issue_detail}. "
            "Our billing team is investigating this matter and will ensure "
            "any incorrect charges are resolved promptly.\n\n"
            "You can expect a resolution within 2-3 business days. If the charge is "
            "confirmed to be an error, a full reversal will be processed.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        (
            "Hello,\n\n"
            "We appreciate you bringing this billing issue to our attention{issue_detail}. "
            "We take billing accuracy very seriously.\n\n"
            "Our finance team has been notified and will review your account within "
            "24-48 hours. If an overcharge or duplicate payment is confirmed, we will "
            "issue a correction immediately.\n\n"
            "Warm regards,\nBilling Support Team"
        ),
    ],
    "Technical Problem": [
        (
            "Dear Customer,\n\n"
            "Thank you for reporting this technical issue{issue_detail}. We understand how "
            "frustrating it can be when things don't work as expected.\n\n"
            "Please try the following:\n"
            "1. Clear your browser cache and cookies\n"
            "2. Try using a different browser or device\n"
            "3. Ensure your app is updated to the latest version\n\n"
            "If the issue persists, our engineering team will provide a fix.\n\n"
            "Thank you for your patience,\nTechnical Support Team"
        ),
        (
            "Hello,\n\n"
            "We're sorry to hear you're experiencing a technical difficulty{issue_detail}.\n\n"
            "As an immediate workaround:\n"
            "- Restart the application\n"
            "- Check your internet connection stability\n"
            "- Disable any browser extensions that might interfere\n\n"
            "We've escalated this to our engineering team.\n\n"
            "Kind regards,\nTechnical Support Team"
        ),
    ],
    "Account Access": [
        (
            "Dear Customer,\n\n"
            "We understand the urgency of regaining access to your account{issue_detail}.\n\n"
            "We recommend the following steps:\n"
            "1. Visit our password reset page at the login screen\n"
            "2. Check your spam/junk folder for the reset email\n"
            "3. If using two-factor authentication, try backup codes\n\n"
            "If you're still unable to access your account, our security team can "
            "verify your identity and manually restore access within 24 hours.\n\n"
            "Best regards,\nAccount Security Team"
        ),
        (
            "Hello,\n\n"
            "We're sorry you're having trouble accessing your account{issue_detail}.\n\n"
            "Our team has flagged your account for priority review. "
            "A security specialist will reach out to verify your identity.\n\n"
            "Warm regards,\nAccount Recovery Team"
        ),
    ],
    "Refund Request": [
        (
            "Dear Customer,\n\n"
            "Thank you for contacting us regarding your refund request{issue_detail}.\n\n"
            "Refunds typically follow this timeline:\n"
            "- Review: 1-2 business days\n"
            "- Processing: 3-5 business days\n"
            "- Bank reflection: 5-10 business days\n\n"
            "You will receive a confirmation email once approved.\n\n"
            "We appreciate your patience,\nRefund Processing Team"
        ),
        (
            "Hello,\n\n"
            "We acknowledge your refund request{issue_detail} and apologize "
            "for any inconvenience.\n\n"
            "Your case has been assigned for expedited review. "
            "We aim to process all eligible refunds within 5 business days.\n\n"
            "Thank you for your understanding,\nCustomer Care Team"
        ),
    ],
    "Feature Request": [
        (
            "Dear Customer,\n\n"
            "Thank you for your valuable suggestion{issue_detail}!\n\n"
            "Your feature request has been logged and shared with our "
            "product development team. We prioritize features based on "
            "user demand and feasibility.\n\n"
            "Thank you for helping us build a better product!\n\n"
            "Best regards,\nProduct Team"
        ),
        (
            "Hello,\n\n"
            "We appreciate you sharing your idea{issue_detail}!\n\n"
            "Your suggestion has been added to our feature backlog "
            "for consideration in future releases.\n\n"
            "Kind regards,\nProduct Development Team"
        ),
    ],
    "General Inquiry": [
        (
            "Dear Customer,\n\n"
            "Thank you for reaching out{issue_detail}! We're happy to help.\n\n"
            "Our support team is reviewing your inquiry and will provide a detailed "
            "response shortly. For quick answers, you can also check our Help Center.\n\n"
            "We aim to respond within 24 hours.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        (
            "Hello,\n\n"
            "Thanks for getting in touch{issue_detail}!\n\n"
            "A member of our support team will review your inquiry and get back "
            "to you within one business day.\n\n"
            "Warm regards,\nSupport Team"
        ),
    ],
}

PRIORITY_PREFIXES = {
    "Urgent": "[!!] PRIORITY: This ticket has been flagged as URGENT. ",
    "High": "This has been marked as a high-priority issue. ",
    "Medium": "",
    "Low": "",
}

PRIORITY_SUFFIXES = {
    "Urgent": "\n\n>> This ticket has been escalated to our senior support team.",
    "High": "\n\n>> Your ticket has been prioritized for faster resolution.",
    "Medium": "",
    "Low": "",
}


def extract_issue_keywords(text):
    details = []
    amounts = re.findall(r"\$[\d,.]+", text)
    if amounts:
        details.append(f"regarding the charge of {amounts[0]}")
    errors = re.findall(r"E-\d{3}|ERR_\w+", text, re.IGNORECASE)
    if errors:
        details.append(f"related to error {errors[0]}")
    orders = re.findall(r"(?:order|ticket|ref)\s*#?\s*(\w+)", text, re.IGNORECASE)
    if orders:
        details.append(f"for reference #{orders[0]}")
    if details:
        return " " + ", ".join(details)
    return ""


def generate_reply(text, category, priority="Medium"):
    templates = REPLY_TEMPLATES.get(category, REPLY_TEMPLATES["General Inquiry"])
    template_idx = len(text) % len(templates)
    template = templates[template_idx]
    issue_detail = extract_issue_keywords(text)
    reply = template.format(issue_detail=issue_detail)

    prefix = PRIORITY_PREFIXES.get(priority, "")
    suffix = PRIORITY_SUFFIXES.get(priority, "")
    if prefix:
        reply = prefix + "\n\n" + reply
    if suffix:
        reply = reply + suffix
    return reply


if __name__ == "__main__":
    test_cases = [
        {"text": "I was charged $49.99 twice for my subscription", "category": "Billing Issue", "priority": "High"},
        {"text": "The app keeps crashing when I upload files", "category": "Technical Problem", "priority": "Medium"},
        {"text": "I want a full refund for my Pro plan. Order #789012", "category": "Refund Request", "priority": "Urgent"},
    ]
    for case in test_cases:
        print("=" * 70)
        print(f"TICKET: {case['text']}")
        print(f"CATEGORY: {case['category']} | PRIORITY: {case['priority']}")
        print("-" * 70)
        print(generate_reply(case["text"], case["category"], case["priority"]))
        print()
