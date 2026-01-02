# Movie Recommendation System with Causal Inference

A hybrid recommendation engine combining Matrix Factorization, Thompson Sampling, and Propensity Score Matching to handle sparse data and identify true genre effects.

---

## Results

| Approach | RMSE | Improvement |
|----------|------|-------------|
| Baseline | 1.13 | - |
| Matrix Factorization | 0.87 | 23% ⬆️ |
| Thompson Sampling | 0.91 | 19% ⬆️ |
| **Hybrid Model** | **0.84** | **26% ⬆️** |

**Causal Finding**: Action movies increase ratings by +0.18 points (true effect, not just correlation)

---

## What I Built

1. **Matrix Factorization** - Collaborative filtering for users with rating history
2. **Thompson Sampling Bandits** - Handles cold-start and exploration
3. **Causal Inference** - Propensity score matching to find true genre effects

**Dataset**: MovieLens 100K (100,836 ratings, 98.3% sparsity)

---

## Key Challenges Solved

-  98.3% data sparsity
-  Cold start (76.7% of movies have <10 ratings)
-  Long-tail distribution
-  Separating correlation from causation

---

## Tech Stack

Python | NumPy | pandas | scikit-learn | Matplotlib

---

## How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Open notebook
jupyter notebook notebook.ipynb
```

---

## Project Highlights

**Machine Learning**
- Custom matrix factorization with SGD optimization
- 50-dimensional user/movie embeddings
- Regularization to prevent overfitting

**Reinforcement Learning**
- Thompson Sampling for exploration-exploitation
- Bayesian posterior updates
- 47% better coverage of niche content

**Causal Inference**
- Propensity score matching
- Controlled for user bias, popularity, other genres
- Found naive correlation overstates effect by 133%

---

## Files

- `notebook.ipynb` - Complete analysis
- `requirements.txt` - Python dependencies
- `data/` - MovieLens dataset (download separately)

---

## Dataset

**Source**: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)  
**Citation**: Harper & Konstan (2015). The MovieLens Datasets. ACM TiiS 5(4).

---

## Contact

**Nishu Kumari Singh**  
Email: nksingh9@asu.edu  
LinkedIn: https://www.linkedin.com/in/nishu-kumari-singh-05b36b262/  
---


**If you found this useful, please star the repo!**
