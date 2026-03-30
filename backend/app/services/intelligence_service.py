def get_shelf_life(ripeness):
    if ripeness < 30:
        return "6-8 days"
    elif ripeness < 60:
        return "4-6 days"
    elif ripeness < 80:
        return "2-4 days"
    else:
        return "1-2 days"


def get_quality_score(ripeness):
    # Ideal ripeness = 80–90
    ideal = 85
    score = 100 - abs(ripeness - ideal)
    return round(max(0, score), 2)


def get_grade_and_market(score):
    if score >= 90:
        return "A+", "International Export", "Premium"
    elif score >= 80:
        return "A", "Export", "High"
    elif score >= 65:
        return "B+", "Domestic (Tier-1)", "Medium-High"
    elif score >= 50:
        return "B", "Domestic", "Medium"
    elif score >= 30:
        return "C", "Processing Industry", "Low"
    else:
        return "Reject", "Waste", "None"


def get_recommendation(ripeness):
    if ripeness < 50:
        return "Not ready for harvest"
    elif ripeness < 70:
        return "Prepare for harvest in few days"
    elif ripeness < 85:
        return "Optimal harvest window approaching"
    elif ripeness <= 95:
        return "Harvest now (ideal)"
    else:
        return "Immediate sale required"


def generate_intelligence(ripeness):
    shelf_life = get_shelf_life(ripeness)
    score = get_quality_score(ripeness)
    grade, market, price = get_grade_and_market(score)
    recommendation = get_recommendation(ripeness)

    return {
        "shelf_life": shelf_life,
        "quality_score": score,
        "grade": grade,
        "market": market,
        "price_category": price,
        "recommendation": recommendation
    }