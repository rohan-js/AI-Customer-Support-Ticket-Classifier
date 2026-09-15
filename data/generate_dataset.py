"""
Synthetic Customer Support Ticket Dataset Generator
====================================================
Generates 5,000 realistic support tickets across 6 categories and 4 priority levels.

Priorities are derived from ticket content, not random: each template carries a
canonical priority (e.g. a hacked account is Urgent, a feature suggestion is Low)
so that the priority prediction task is actually learnable from text. A small
labeled-noise step shifts a fraction of labels one adjacent level to simulate
human labeling variance and keep the task realistic (~85% ceiling, not 100%).
"""

import random
import csv
import os
from datetime import datetime, timedelta

random.seed(42)

PRIORITIES = ["Low", "Medium", "High", "Urgent"]

# Fraction of tickets whose priority is shifted one adjacent level (up or down)
# relative to the canonical priority signaled by the text.
NOISE_PROB = 0.15

# Each category maps to a list of (template, canonical_priority) tuples.
CATEGORIES = {
    "Billing Issue": [
        ("I was charged {amount} twice for my subscription this month. Please fix this.", "High"),
        ("My credit card was billed {amount} but I already cancelled my plan last week.", "Medium"),
        ("There is an unexpected charge of {amount} on my account. I did not authorize this.", "High"),
        ("I see a duplicate charge on my statement for {amount}. Can you reverse it?", "High"),
        ("My invoice shows {amount} but my plan should only cost half that.", "Medium"),
        ("I upgraded my plan but was charged the old rate plus {amount} extra.", "Medium"),
        ("Payment of {amount} was deducted even though I switched to the free tier.", "Medium"),
        ("Why was I charged {amount}? I thought I had a promotional discount applied.", "Low"),
        ("I received a billing notification for {amount} but I never made this purchase.", "High"),
        ("The auto-renewal charged me {amount} and I want to dispute this transaction.", "Medium"),
    ],
    "Technical Problem": [
        ("The app keeps crashing whenever I try to {action}. I'm using {device}.", "High"),
        ("I'm getting a {error_code} error when I try to {action}. Please help.", "Medium"),
        ("The website is extremely slow and takes over 30 seconds to {action}.", "Medium"),
        ("I cannot {action} because the page keeps showing a blank screen on {device}.", "High"),
        ("After the latest update, the {feature} feature is completely broken.", "High"),
        ("My data is not syncing properly between my {device} and the web version.", "Medium"),
        ("The {feature} button does nothing when I click it. Tried multiple browsers.", "Medium"),
        ("I'm experiencing frequent disconnections while trying to {action}.", "Medium"),
        ("The search function returns no results even when I search for existing {feature}.", "Low"),
        ("Videos won't load and I keep getting buffering issues on {device}.", "Medium"),
    ],
    "Account Access": [
        ("I forgot my password and the reset email never arrives. My email is {email}.", "Medium"),
        ("My account has been locked after too many login attempts. Please unlock it.", "High"),
        ("I cannot log in with my credentials. I keep getting 'invalid password' error.", "Medium"),
        ("Someone may have hacked my account. I see login activity from {location}.", "Urgent"),
        ("I need to change my email from {email} to a new one but the option is greyed out.", "Low"),
        ("Two-factor authentication is not sending codes to my phone. I'm locked out.", "High"),
        ("I merged my accounts but now I cannot access either of them.", "High"),
        ("My account was deactivated without any notice. I need it restored immediately.", "High"),
        ("I'm trying to log in but it says my account does not exist.", "Medium"),
        ("SSO login through Google is not working. It redirects me back to login.", "Medium"),
    ],
    "Refund Request": [
        ("I want a full refund for my {product} purchase. It did not meet expectations.", "Medium"),
        ("Please refund my last payment of {amount}. The service was down for 3 days.", "High"),
        ("I accidentally purchased {product} and want my money back immediately.", "Medium"),
        ("I cancelled within the trial period but was still charged {amount}. I want a refund.", "High"),
        ("The {product} I received was defective. I need a refund, not a replacement.", "Medium"),
        ("I was promised a refund two weeks ago but haven't received it. Order #{order_id}.", "High"),
        ("I'm requesting a refund because the {product} features advertised are not available.", "Medium"),
        ("My subscription renewal should not have gone through. Please refund {amount}.", "Medium"),
        ("I returned the {product} 5 days ago but my refund has not been processed.", "High"),
        ("I want to cancel and get a prorated refund for the remaining {amount} on my plan.", "Low"),
    ],
    "Feature Request": [
        ("It would be great if you could add {feature} to the platform.", "Low"),
        ("I really wish the app had a {feature} option. Competitors already offer this.", "Low"),
        ("Can you please implement {feature}? It would make the workflow much smoother.", "Low"),
        ("I'd like to suggest adding {feature} -- it would save me hours every week.", "Low"),
        ("Is there any plan to introduce {feature}? I've seen other users request it too.", "Low"),
        ("The product would be perfect if it had {feature}. Please consider adding it.", "Low"),
        ("I need {feature} to integrate with my existing tools. Is this on the roadmap?", "Medium"),
        ("Your app is missing {feature} which is essential for professional use.", "Medium"),
        ("Could you add bulk {action} capability? Currently I have to do everything one by one.", "Low"),
        ("Please consider adding keyboard shortcuts for {action}.", "Low"),
    ],
    "General Inquiry": [
        ("Can you tell me more about your {product} pricing plans?", "Low"),
        ("What are the differences between the Basic and Premium plans?", "Low"),
        ("How do I {action} in my account settings? I can't find the option.", "Low"),
        ("Is there a student discount available for {product}?", "Low"),
        ("What is your data privacy policy? I want to know how my data is handled.", "Low"),
        ("Do you offer enterprise plans for teams larger than 50 people?", "Low"),
        ("How long does it typically take for support to respond to tickets?", "Low"),
        ("Can I export my data in CSV format? I need it for reporting.", "Low"),
        ("What integrations do you support? Specifically looking for {feature} integration.", "Low"),
        ("Is there an API available for {product}? I'd like to automate some tasks.", "Low"),
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


def noisy_priority(canonical):
    """Return the canonical priority, shifted one adjacent level with NOISE_PROB probability."""
    if random.random() < NOISE_PROB:
        idx = PRIORITIES.index(canonical)
        step = random.choice([-1, 1])
        idx = min(max(idx + step, 0), len(PRIORITIES) - 1)
        return PRIORITIES[idx]
    return canonical


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
        template, canonical_priority = random.choice(CATEGORIES[category])
        text = fill_template(template)
        priority = noisy_priority(canonical_priority)
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
    print(f"    Priority noise: {NOISE_PROB:.0%} shifted one adjacent level")
    return output_path


if __name__ == "__main__":
    generate_dataset()
