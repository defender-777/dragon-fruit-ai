import torch

def interpret_ripeness(probs):
    immature_prob = probs[0][0].item()
    mature_prob = probs[0][1].item()

    ripeness_percent = mature_prob * 100
    confidence = max(immature_prob, mature_prob)

    #  OOD FILTER
    if confidence < 0.7:
        return {
            "status": "invalid",
            "message": "Input not recognized as dragon fruit",
            "confidence": round(confidence * 100, 2)
        }

    #  IMPROVED STAGE INTERPRETATION
    if ripeness_percent < 30:
        stage = "Immature"
    elif 30 <= ripeness_percent <= 70:
        stage = "Transitional"
    else:
        stage = "Mature"

    return {
        "status": "success",
        "ripeness_percent": round(ripeness_percent, 2),
        "stage": stage,
        "confidence": round(confidence * 100, 2)
    }