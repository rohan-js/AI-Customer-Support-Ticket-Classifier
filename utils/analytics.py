"""
Analytics Module
=================
Generates analytics data from the ticket dataset for business intelligence.

Produces:
    1. Tickets per category (bar chart + data)
    2. Average priority distribution
    3. Daily ticket volume (time series)
    4. Priority by category heatmap
    5. CSV export for Power BI dashboard

All visualizations use matplotlib/seaborn with a clean, professional style.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set plot style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150


def load_ticket_data(data_path=None):
    """Load ticket dataset."""
    if data_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, "data", "tickets.csv")
    return pd.read_csv(data_path, parse_dates=["created_date"])


def generate_analytics(df=None, output_dir=None):
    """
    Generate comprehensive analytics from ticket data.
    Returns dictionary with all analytics dataframes.
    """
    if df is None:
        df = load_ticket_data()

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
    os.makedirs(output_dir, exist_ok=True)

    analytics = {}

    # 1. Tickets per Category
    print("\n[...] Generating analytics...")
    category_counts = df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "ticket_count"]
    category_counts["percentage"] = (category_counts["ticket_count"] / len(df) * 100).round(2)
    analytics["tickets_per_category"] = category_counts

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(category_counts))
    bars = ax.bar(category_counts["category"], category_counts["ticket_count"], color=colors, edgecolor="white")
    for bar, count in zip(bars, category_counts["ticket_count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(count), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Tickets per Category", fontsize=16, fontweight="bold")
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Number of Tickets", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tickets_per_category.png"))
    plt.close()

    # 2. Priority Distribution
    priority_counts = df["priority"].value_counts().reset_index()
    priority_counts.columns = ["priority", "ticket_count"]
    priority_counts["percentage"] = (priority_counts["ticket_count"] / len(df) * 100).round(2)
    analytics["priority_distribution"] = priority_counts

    fig, ax = plt.subplots(figsize=(8, 8))
    priority_order = ["Urgent", "High", "Medium", "Low"]
    ordered_counts = []
    for p in priority_order:
        match = priority_counts[priority_counts["priority"] == p]
        ordered_counts.append(match["ticket_count"].values[0] if len(match) > 0 else 0)
    colors_pie = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    ax.pie(ordered_counts, labels=priority_order, colors=colors_pie, autopct="%1.1f%%", startangle=90)
    ax.set_title("Priority Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "priority_distribution.png"))
    plt.close()

    # 3. Daily Ticket Volume
    daily_volume = df.groupby("created_date").size().reset_index(name="ticket_count").sort_values("created_date")
    analytics["daily_volume"] = daily_volume

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_volume["created_date"], daily_volume["ticket_count"], color="#3498db", linewidth=1, alpha=0.7)
    if len(daily_volume) > 7:
        rolling_avg = daily_volume["ticket_count"].rolling(window=7, center=True).mean()
        ax.plot(daily_volume["created_date"], rolling_avg, color="#e74c3c", linewidth=2, label="7-day moving average")
        ax.legend()
    ax.set_title("Daily Ticket Volume", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Tickets", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "daily_ticket_volume.png"))
    plt.close()

    # 4. Priority by Category Heatmap
    pivot = pd.crosstab(df["category"], df["priority"])
    for col in ["Low", "Medium", "High", "Urgent"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Low", "Medium", "High", "Urgent"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Priority Distribution by Category", fontsize=16, fontweight="bold")
    ax.set_xlabel("Priority Level", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "priority_by_category_heatmap.png"))
    plt.close()

    # 5. Export Combined Analytics CSV for Power BI
    analytics_export = df.copy()
    priority_map = {"Low": 1, "Medium": 2, "High": 3, "Urgent": 4}
    analytics_export["priority_numeric"] = analytics_export["priority"].map(priority_map)
    analytics_export["day_of_week"] = pd.to_datetime(analytics_export["created_date"]).dt.day_name()
    analytics_export["month"] = pd.to_datetime(analytics_export["created_date"]).dt.month_name()

    csv_path = os.path.join(output_dir, "analytics_export.csv")
    analytics_export.to_csv(csv_path, index=False)
    analytics["export_path"] = csv_path

    # Summary
    print("\n" + "=" * 60)
    print("ANALYTICS SUMMARY")
    print("=" * 60)
    print(f"\nTotal tickets: {len(df)}")
    print(f"\nTickets per Category:")
    for _, row in category_counts.iterrows():
        print(f"  {row['category']:25s} {row['ticket_count']:5d}  ({row['percentage']}%)")
    print(f"\nPriority Distribution:")
    for _, row in priority_counts.iterrows():
        print(f"  {row['priority']:10s} {row['ticket_count']:5d}  ({row['percentage']}%)")
    print(f"\nDate range: {df['created_date'].min()} to {df['created_date'].max()}")
    print(f"\nAvg daily volume: {daily_volume['ticket_count'].mean():.1f} tickets/day")
    print(f"\n[OK] Charts saved to: {output_dir}/")
    print(f"[OK] Power BI CSV exported -> {csv_path}")

    return analytics


if __name__ == "__main__":
    generate_analytics()
