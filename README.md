#  Dragon Fruit Ripeness Detection System

![Dockerized](https://img.shields.io/badge/Docker-Containerized-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)

---

##  Overview

The **Dragon Fruit Ripeness Detection System** is a production-ready AI system that predicts the ripeness level of dragon fruits using deep learning.

It combines:

*  Deep Learning (EfficientNet)
*  FastAPI backend for inference
*  Docker containerization for deployment
*  Intelligent decision layer for actionable insights

---

##  Features

*  Image-based ripeness detection
*  Ripeness percentage prediction
*  Intelligent output layer:

  * Quality scoring (0–100)
  * Market grading (A+, A, B+, B, C)
  * Shelf-life estimation
  * Price categorization
  * Harvest recommendation
*  Out-of-Distribution (OOD) detection
*  Optimized CPU inference
*  REST API for mobile/web integration
*  Fully Dockerized for portability

---

##  System Architecture

ML Model (EfficientNet)
↓
FastAPI Backend
↓
Inference + Intelligence Layer
↓
Docker Container
↓
Cloud Deployment (Render)
↓
Mobile / Client Integration

---

##  Tech Stack

* **ML Framework:** PyTorch, timm
* **Backend:** FastAPI
* **Deployment:** Docker
* **Image Processing:** PIL
* **Numerical Ops:** NumPy

---

##  Getting Started

###  1. Clone Repository

```bash
git clone https://github.com/your-username/dragon-fruit-ai.git
cd dragon-fruit-ai/backend
```

---

###  2. Run with Docker

```bash
docker build -t dragon-fruit-api .
docker run -p 8000:8000 dragon-fruit-api
```

---

###  3. Access API

```text
http://localhost:8000/docs
```

---

##  API Usage

### Endpoint

```text
POST /predict
```

---

### Request

* Type: `multipart/form-data`
* Field: `file` (image)

---

### Example Response

```json
{
  "ripeness_percent": 78,
  "grade": "A",
  "quality_score": 82,
  "price_category": "Premium",
  "shelf_life_days": 3,
  "harvest_recommendation": "Ready"
}
```

---

##  Error Handling

| Scenario       | Response                  |
| -------------- | ------------------------- |
| Invalid image  | 400                       |
| Internal error | 500                       |
| OOD input      | `{ "status": "invalid" }` |

---

##  Testing

* Swagger UI available at `/docs`
* Supports real-time image upload
* Validated with real dataset samples

---

##  Project Structure

```text
dragon-fruit-ai/
├── backend/
│   ├── app/
│   ├── models/
│   │   └── ripeness_model.pth
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
├── ai-engine/
```

---

##  Deployment

The system is designed for deployment on cloud platforms like Render.

Key Features:

* Dynamic port handling
* Docker-based deployment
* Production-ready architecture

---

##  Contributing

We welcome contributions from developers, ML engineers, and students!

### Steps:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a Pull Request

---

##  Open Issues

Check out:

* `good-first-issue` for beginners
* `help wanted` for contributors
* `ml`, `backend`, `devops` for domain-specific tasks

---

##  Future Improvements

*  API authentication & rate limiting
*  Mobile app integration
*  Batch inference support
*  Model explainability (Grad-CAM)
*  Multi-fruit detection system
*  CI/CD pipeline automation

---

##  Learnings & Challenges

This project involved solving real-world engineering problems:

* Model architecture mismatch (timm vs torchvision)
* Docker dependency conflicts
* Large image optimization (CUDA → CPU)
* Network timeout handling
* Cloud deployment constraints

---

##  Highlights

* End-to-end ML system (training → deployment)
* Production-ready backend
* Dockerized and optimized
* Real-world problem solving

---

##  Contact

For collaboration or queries:

* GitHub: https://github.com/defender-777


---

##  Support

If you find this project useful:

 Star ⭐ the repository
 Share with others
 Contribute 🚀

---
