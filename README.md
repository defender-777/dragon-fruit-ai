#  DeepVision AI – Dragon Fruit Intelligence Engine

![Dockerized](https://img.shields.io/badge/Docker-Containerized-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)

---

##  Overview

**DeepVision AI** is a **production-grade ML-as-a-Service system** that performs **intelligent dragon fruit analysis** using a multi-stage AI pipeline.

Unlike basic ML projects, this system combines:

*  Deep Learning (EfficientNet)
*  Object Detection (DETR)
*  FastAPI inference engine
*  Docker containerization
*  Cloud deployment (Render)
*  Dynamic model loading (Hugging Face)
*  Intelligence layer for decision-making

---

##  Key Features

###  Core AI Capabilities

* Dragon fruit ripeness prediction (image-based)
* Multi-stage inference pipeline (Detection → Classification → Validation)
* Strict out-of-distribution rejection (non-dragon fruits filtered)
* Dynamic model loading from Hugging Face

---

###  Intelligent Output Layer

* Ripeness percentage (0–100)
* Quality score
* Grade classification (A+, A, B+, B, C)
* Shelf-life estimation
* Market category (Export / Local)
* Price category
* Harvest recommendation

---

###  System Features

* Production-ready REST API
* In-memory inference (optimized performance)
* DETR-based object localization
* Smart fallback mechanism (robust inference)
* Strict validation pipeline (reduces false positives)
* Dockerized backend for portability
* Cloud deployment with auto-rebuild (CI/CD style)
* Mobile-ready API design

---

##  System Architecture

```text
Client (Mobile / Web)
        ↓
POST /predict  (Public API)
        ↓
DETR Object Detection
        ↓
Object Cropping (ROI Extraction)
        ↓
Deep Learning Model (EfficientNet)
        ↓
Validation Layer (OOD + Confidence Filtering)
        ↓
Intelligence Layer (Decision Engine)
        ↓
JSON Response
```

---

##  Tech Stack

* **ML Framework:** PyTorch, timm
* **Detection:** DETR (Transformers)
* **Backend:** FastAPI
* **Deployment:** Docker
* **Model Hosting:** Hugging Face
* **Cloud Platform:** Render
* **Image Processing:** PIL
* **Numerical Ops:** NumPy

---

##  Getting Started

### 1️ Clone Repository

```bash
git clone https://github.com/defender-777/dragon-fruit-ai.git
cd dragon-fruit-ai/backend
```

---

###  Run with Docker

```bash
docker build -t deepvision-ai .
docker run -p 8000:8000 deepvision-ai
```

---

###  Access API

```text
http://localhost:8000/docs
```

---

##  API Usage

###  Public Endpoint

```text
POST /predict
```

> This is the **main production endpoint** (Smart Pipeline)

---

###  Request

* Type: `multipart/form-data`
* Field: `file` (image)

---

###  Example Response

```json
{
  "status": "success",
  "ripeness_percent": 89.98,
  "stage": "Mature",
  "confidence": 89.98,
  "shelf_life": "1-2 days",
  "quality_score": 95.02,
  "grade": "A+",
  "market": "Export",
  "price_category": "Premium",
  "recommendation": "Harvest now (ideal)"
}
```

---

##  Error Handling

| Scenario           | Response               |
| ------------------ | ---------------------- |
| Invalid image      | 400                    |
| Non-dragon fruit   | Rejected with guidance |
| No object detected | Guidance message       |
| Internal error     | 500                    |

---

##  Testing

* Swagger UI: `/docs`
* Supports real-time image uploads
* Tested on multiple fruit classes
* Validated with edge-case scenarios (OOD inputs)

---

##  Project Structure

```text
dragon-fruit-ai/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── detr.py
│   │   ├── utils.py
│   │   ├── services/
│   │   │   ├── prediction_service.py
│   │   │   └── intelligence_service.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
├── ai-engine/
├── mobile-app/ (future)
```

---

##  Deployment

Designed for **cloud-native deployment**:

* Docker-based builds
* Dynamic port handling
* Auto-deploy via GitHub integration
* Stateless architecture (model loaded at runtime)

---

##  Live API

🔗 https://dragon-fruit-ai.onrender.com
📚 Docs: https://dragon-fruit-ai.onrender.com/docs

---

##  Contributing

We welcome contributors across domains:

### Steps:

1. Fork repository
2. Create branch (`feature/...`)
3. Commit changes
4. Open Pull Request

---

##  Open Issues

* `good-first-issue`
* `help wanted`
* `ml`, `backend`, `devops`, `frontend`

---

##  Future Improvements

*  API authentication & rate limiting
*  MongoDB analytics dashboard
*  Mobile app integration
*  Batch inference API
*  Model explainability (Grad-CAM)
*  Multi-fruit classification system
*  CI/CD automation

---

##  Learnings & Engineering Challenges

* Handling model–framework mismatches (timm vs torchvision)
* Optimizing inference (GPU → CPU transition)
* Docker layer caching & build optimization
* Dynamic model loading (reducing image size by >80%)
* Handling OOD (Out-of-Distribution) inputs
* Designing multi-stage inference pipelines
* Building production-ready ML APIs

---

##  Highlights

* End-to-end ML system (training → deployment → scaling)
* Multi-stage AI pipeline (Detection + Classification)
* Production-ready backend architecture
* Cloud-deployed ML service
* Real-world problem solving (agriculture AI)

---

##  Contact

GitHub: https://github.com/defender-777

---

## ⭐ Support

If you find this project useful:

* Star ⭐ the repository
* Share 📢
* Contribute 

---

##  Final Note

This is not just a model.

This is a **production-grade AI system combining ML, backend engineering, and cloud deployment - ML-as-a-Service**.

---
