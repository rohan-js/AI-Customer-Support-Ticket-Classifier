"""
Synthetic Customer Support Ticket Dataset Generator
====================================================
Generates 5,000 realistic support tickets across 6 categories and 4 priority levels.
"""

import random
import csv
import os
from datetime import datetime, timedelta

random.seed(42)

CATEGORIES = {
    "Billing Issue": [
        "I was charged {amount} twice for my subscription this month. Please fix this.",
        "My credit card was billed {amount} but I already cancelled my plan last week.",
        "There is an unexpected charge of {amount} on my account. I did not authorize this.",
        "I see a duplicate charge on my statement for {amount}. Can you reverse it?",
        "My invoice shows {amount} but my plan should only cost half that.",
        "I upgraded my plan but was charged the old rate plus {amount} extra.",
        "Payment of {amount} was deducted even though I switched to the free tier.",
        "Why was I charged {amount}? I thought I had a promotional discount applied.",
        "I received a billing notification for {amount} but I never made this purchase.",
        "The auto-renewal charged me {amount} and I want to dispute this transaction.",
    ],
    "Technical Problem": [
        "The app keeps crashing whenever I try to {action}. I'm using {device}.",
        "I'm getting a {error_code} error when I try to {action}. Please help.",
        "The website is extremely slow and takes over 30 seconds to {action}.",
        "I cannot {action} because the page keeps showing a blank screen on {device}.",
        "After the latest update, the {feature} feature is completely broken.",
        "My data is not syncing properly between my {device} and the web version.",
        "The {feature} button does nothing when I click it. Tried multiple browsers.",
        "I'm experiencing frequent disconnections while trying to {action}.",
        "The search function returns no results even when I search for existing {feature}.",
        "Videos won't load and I keep getting buffering issues on {device}.",
    ],
    "Account Access": [
        "I forgot my password and the reset email never arrives. My email is {email}.",
        "My account has been locked after too many login attempts. Please unlock it.",
        "I cannot log in with my credentials. I keep getting 'invalid password' error.",
        "Someone may have hacked my account. I see login activity from {location}.",
        "I need to change my email from {email} to a new one but the option is greyed out.",
        "Two-factor authentication is not sending codes to my phone. I'm locked out.",
        "I merged my accounts but now I cannot access either of them.",
        "My account was deactivated without any notice. I need it restored immediately.",
        "I'm trying to log in but it says my account does not exist.",
        "SSO login through Google is not working. It redirects me back to login.",
    ],
    "Refund Request": [
        "I want a full refund for my {product} purchase. It did not meet expectations.",
        "Please refund my last payment of {amount}. The service was down for 3 days.",
        "I accidentally purchased {product} and want my money back immediately.",
        "I cancelled within the trial period but was still charged {amount}. I want a refund.",
        "The {product} I received was defective. I need a refund, not a replacement.",
        "I was promised a refund two weeks ago but haven't received it. Order #{order_id}.",
        "I'm requesting a refund because the {product} features advertised are not available.",
        "My subscription renewal should not have gone through. Please refund {amount}.",
        "I returned the {product} 5 days ago but my refund has not been processed.",
        "I want to cancel and get a prorated refund for the remaining {amount} on my plan.",
    ],
    "Feature Request": [
        "It would be great if you could add {feature} to the platform.",
        "I really wish the app had a {feature} option. Competitors already offer this.",
        "Can you please implement {feature}? It would make the workflow much smoother.",
        "I'd like to suggest adding {feature} -- it would save me hours every week.",
        "Is there any plan to introduce {feature}? I've seen other users request it too.",
        "The product would be perfect if it had {feature}. Please consider adding it.",
        "I need {feature} to integrate with my existing tools. Is this on the roadmap?",
        "Your app is missing {feature} which is essential for professional use.",
        "Could you add bulk {action} capability? Currently I have to do everything one by one.",
        "Please consider adding keyboard shortcuts for {action}.",
    ],
    "General Inquiry": [
        "Can you tell me more about your {product} pricing plans?",
        "What are the differences between the Basic and Premium plans?",
        "How do I {action} in my account settings? I can't find the option.",
        "Is there a student discount available for {product}?",
        "What is your data privacy policy? I want to know how my data is handled.",
        "Do you offer enterprise plans for teams larger than 50 people?",
        "How long does it typically take for support to respond to tickets?",
        "Can I export my data in CSV format? I need it for reporting.",
        "What integrations do you support? Specifically looking for {feature} integration.",
        "Is there an API available for {product}? I'd like to automate some tasks.",
    ],
}

AMOUNTS = ["$9.99", "$14.99", "$19.99", "$24.99", "$29.99", "$49.99", "$79.99", "$99.99"]
ACTIONS = ["upload files", "download reports", "open the dashboard", "save my settings",
           "access my profile", "send a message", "export data", "generate invoices"]
DEVICES = ["iPhone 15", "Samsung Galaxy S24", "MacBook Pro", "Windows laptop", "iPad"]
ERROR_CODES = ["E-404", "E-500", "E-503", "ERR_TIMEOUT", "ERR_AUTH_FAILED", "E-403"]
FEATURES = ["dark mode", "export to PDF", "calendar view", "offline mode", "bulk editing",
            "multi-language support", "custom dashboards", "real-time notifications"]
EMAILS = ["user123@gmail.com", "john.doe@email.com", "jane@company.org", "alex@outlook.com"]
LOCATIONS = ["Russia", "China", "Brazil", "an unknown IP address", "a different country"]
PRODUCTS = ["annual subscription", "Pro plan", "Enterprise license", "add-on package"]
ORDER_IDS = [f"{random.randint(100000, 999999)}" for _ in range(20)]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]
PRIORITY_WEIGHTS = [0.25, 0.35, 0.25, 0.15]


def fill_template(template):
    result = template
    result = result.replace("{amount}", random.choice(AMOUNTS))
    result = result.replace("{action}", random.choice(ACTIONS))
    result = result.replace("{device}", random.choice(DEVICES))
    result = result.replace("{error_code}", random.choice(ERROR_CODES))
    result = result.replace("{feature}", random.choice(FEATURES))
    result = result.replace("{email}", random.choice(EMAILS))
    result = result.replace("{location}", random.choice(LOCATIONS))
    result = result.replace("{product}", random.choice(PRODUCTS))
    result = result.replace("{order_id}", random.choice(ORDER_IDS))
    return result


def generate_dataset(num_tickets=5000, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tickets.csv")

    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    categories = list(CATEGORIES.keys())
    tickets = []

    for i in range(num_tickets):
        category = random.choice(categories)
        template = random.choice(CATEGORIES[category])
        text = fill_template(template)
        priority = random.choices(PRIORITIES, weights=PRIORITY_WEIGHTS, k=1)[0]
        random_days = random.randint(0, date_range_days)
        created_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

        tickets.append({
            "ticket_id": f"TKT-{i + 1:05d}",
            "text": text,
            "category": category,
            "priority": priority,
            "created_date": created_date,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "text", "category", "priority", "created_date"])
        writer.writeheader()
        writer.writerows(tickets)

    print(f"[OK] Generated {num_tickets} tickets -> {output_path}")
    print(f"    Categories: {len(categories)}")
    print(f"    Date range: {start_date.date()} to {end_date.date()}")
    return output_path


if __name__ == "__main__":
    generate_dataset()
