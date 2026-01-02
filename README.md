# Movie Recommendation System with Causal Inference

A hybrid recommendation engine combining Matrix Factorization and Propensity Score Matching to handle sparse data and identify true genre effects on the MovieLens 100K dataset.

---

## Executive Summary

This project demonstrates advanced machine learning and causal inference techniques applied to the movie recommendation problem. The system achieves 15.9% error reduction on held-out test data despite 98.3% data sparsity. More importantly, the causal analysis reveals that action movies decrease ratings by -0.15 points, an effect that is 79% larger than what naive correlation analysis suggests.

**Key Achievement**: Successfully identified and quantified selection bias, demonstrating why correlation-based analysis can miss 44% of the true causal effect.

---

## Results

### Model Performance

| Approach | RMSE | Improvement |
|----------|------|-------------|
| Baseline (Global Mean) | 1.04 | - |
| Matrix Factorization (Train) | 0.82 | 20.8% |
| **Matrix Factorization (Test)** | **0.88** | **15.9%** |

The test RMSE of 0.88 represents a 15.9% improvement over the baseline while maintaining only a 0.05 gap between training and test performance, indicating minimal overfitting.

### Causal Analysis Results

**Research Question**: Do action movies causally increase or decrease ratings?

**Finding**: Action movies decrease ratings by -0.15 points (p < 0.001)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Naive difference | -0.09 | Simple correlation (biased) |
| **True causal effect (ATE)** | **-0.15** | IPW-adjusted (unbiased) |
| 95% Confidence Interval | [-0.17, -0.14] | Statistical uncertainty |
| P-value | < 0.001 | Highly significant |
| Z-score | -20.50 | Extremely confident |

**Key Insight**: The true causal effect is 79% larger than the naive estimate. Simple correlation analysis captures only 56% of the true effect, missing 44% of the real magnitude due to selection bias.

---

## Technical Approach

### Dataset

**MovieLens 100K**
- 100,836 ratings from 610 users on 9,724 movies
- 98.3% sparsity (extreme missing data)
- 76.7% of movies have fewer than 10 ratings
- 30.4% of ratings are for action movies (30,635 ratings)

### Matrix Factorization

**Architecture**:
- 50-dimensional user and movie embeddings
- User and movie bias terms
- L2 regularization (lambda = 0.05)
- Stochastic Gradient Descent optimization

**Preprocessing**:
- User-mean normalization to handle rating scale differences
- Train/test split (80/20)

**Performance**:
- Training RMSE: 0.82
- Test RMSE: 0.88
- Improvement: 15.9% over baseline
- Overfitting gap: 0.05 (well-controlled)

### Causal Inference

**Methodology**: Inverse Propensity Weighting

**Treatment**: Binary indicator for action genre

**Confounders Controlled**:
- User average rating (generous vs harsh raters)
- User activity level (number of ratings)
- Movie popularity (blockbuster effect)

**Propensity Score Model**:
- Logistic regression on movie popularity
- Propensity range: [0.244, 0.618]
- Good common support for matching

**Balance Achieved**:
- User average rating SMD: 0.003
- User activity SMD: 0.001
- Movie popularity SMD: 0.012
- Maximum SMD: 0.012 (excellent, well below 0.1 threshold)

**Statistical Inference**:
- Bootstrap with 1,000 iterations
- 95% Confidence Interval: [-0.17, -0.14]
- P-value < 0.001 (highly significant)
- Z-score: -20.50

---

## Why This Matters

### The Selection Bias Problem

**What Simple Correlation Shows**: Action movies decrease ratings by -0.09 points

**What Causal Analysis Reveals**: True effect is -0.15 points (79% larger)

**The Mechanism**:
1. Action fans preferentially select action movies (self-selection)
2. These fans rate action movies higher than average viewers would
3. This fan boost (+0.06) partially masks the quality gap
4. Naive analysis sees the net effect (-0.09)
5. Causal analysis removes selection bias to reveal true quality (-0.15)

**Practical Impact**: Using naive correlation for content acquisition decisions would underestimate the quality disadvantage of action movies by 44%, potentially leading to suboptimal budget allocation.

### Business Implications

**Content Strategy**:
- Do not prioritize action content based on raw ratings alone
- Account for the -0.15 point quality disadvantage in valuation
- Use causal estimates for data-driven acquisition decisions

**Recommendation Systems**:
- Implement genre-aware prediction adjustments
- Personalize based on true preferences, not selection artifacts
- Account for systematic genre biases in scoring algorithms

**Strategic Value**:
- Demonstrates importance of causal thinking in data-driven decisions
- Shows how correlation can mislead even with large datasets
- Provides methodology for offline causal evaluation

---

## Technical Details

### Challenges Addressed

**Extreme Sparsity (98.3%)**:
- Only 1.7% of possible user-movie pairs have ratings
- Solution: Matrix Factorization with strong regularization
- Successfully learned meaningful patterns despite sparsity

**Long-tail Distribution**:
- Median movie has only 3 ratings
- 76.7% of movies have fewer than 10 ratings
- Solution: Regularization and careful validation

**Selection Bias**:
- Users choose which movies to watch and rate
- This creates systematic confounding
- Solution: Propensity score matching to remove bias

**Statistical Validation**:
- Bootstrap confidence intervals for uncertainty quantification
- Covariate balance checks to validate matching quality
- Proper train/test split for honest performance reporting

### Dataset Statistics

**User Behavior**:
- Average: 165 ratings per user
- Median: 70 ratings (50% of users rated fewer than 70 movies)
- Range: 20 to 2,698 ratings
- Mean user rating: 3.66 (positive bias)

**Movie Characteristics**:
- Average: 10 ratings per movie
- Median: 3 ratings
- Mean movie rating: 3.26
- Maximum: 329 ratings (blockbuster)

**Rating Distribution**:
- 81% of ratings are 3 stars or higher
- Demonstrates positive selection bias (users rate what they like)

---

## Methodology Validation

### Propensity Score Quality

The propensity score model successfully balanced all covariates:

**Before Matching** (confounded):
- User average rating SMD: 0.046
- User activity SMD: 0.081
- Movie popularity SMD: 0.320 (large imbalance)

**After Propensity Weighting** (balanced):
- User average rating SMD: 0.003
- User activity SMD: 0.001
- Movie popularity SMD: 0.012
- Maximum SMD: 0.012 (excellent balance)

**Interpretation**: The propensity score weighting successfully removed confounding, enabling unbiased causal effect estimation.

### Statistical Rigor

**Confidence Intervals**: Bootstrap with 1,000 iterations provides robust uncertainty quantification. The 95% CI of [-0.17, -0.14] does not include zero, confirming statistical significance.

**Common Support**: Propensity scores range from 0.244 to 0.618, indicating good overlap between treatment and control groups. No extreme propensity scores near 0 or 1.

**Significance Testing**: P-value < 0.001 and Z-score of -20.50 indicate extremely high confidence in the causal effect.

---

## Implementation

### Tech Stack

**Core Libraries**:
- Python 3.x
- NumPy (numerical computing)
- pandas (data manipulation)
- scikit-learn (machine learning, propensity scores)
- Matplotlib (visualization)
- Seaborn (statistical plots)

**Algorithms**:
- Matrix Factorization with Stochastic Gradient Descent
- Propensity Score estimation via Logistic Regression
- Inverse Propensity Weighting
- Bootstrap resampling

**Environment**:
- Google Colab with GPU acceleration
- Jupyter Notebook for interactive analysis

### How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Open notebook
jupyter notebook notebook.ipynb
```

### Files

- `notebook.ipynb` - Complete analysis with code and visualizations
- `requirements.txt` - Python dependencies
- `data/` - MovieLens dataset (download separately)

---

## Key Contributions

### Methodological

1. **Demonstrates causal inference on observational data**: Successfully applied propensity score methods to identify true causal effects without experimental data

2. **Quantifies selection bias**: Showed that naive correlation captures only 56% of true effect, missing 44% due to selection

3. **Validates assumptions**: Verified common support, covariate balance, and statistical significance through rigorous checks

### Technical

1. **Handles extreme sparsity**: Achieved meaningful predictions with 98.3% missing data

2. **Robust evaluation**: Used honest test metrics, bootstrap confidence intervals, and proper validation

3. **Production-ready methodology**: Demonstrated techniques applicable to real-world recommendation systems

### Business Value

1. **Data-driven decision making**: Provides framework for causal evaluation of content strategies

2. **Bias awareness**: Highlights importance of accounting for selection effects in analytics

3. **Strategic guidance**: Enables better content acquisition and recommendation algorithm design

---

## Limitations and Future Work

### Current Limitations

**Observational Data**:
- Cannot completely rule out hidden confounders
- Would benefit from experimental validation (A/B test)
- Assumes no unmeasured confounding

**Sample Size**:
- 100K ratings is moderate, not massive scale
- Limited to MovieLens users (generalizability question)

**Treatment Definition**:
- Binary action/non-action (most movies have multiple genres)
- Does not capture genre interactions
- Could miss heterogeneous effects

### Future Directions

**Model Improvements**:
1. Deep learning embeddings (autoencoders, neural collaborative filtering)
2. Temporal dynamics (time-varying preferences)
3. Context-aware features (device, time of day, viewing sequence)

**Causal Extensions**:
1. Heterogeneous treatment effects (effect variation by user type)
2. Multiple simultaneous treatments (all genres)
3. Instrumental variables (address hidden confounding)
4. Mediation analysis (understand causal mechanisms)

**Production Deployment**:
1. REST API for real-time recommendations
2. A/B testing to validate causal findings
3. Monitoring for concept drift
4. Scalable inference for large user bases

---

## Dataset Source

**Source**: MovieLens 100K  
**Link**: https://grouplens.org/datasets/movielens/100k/  
**Citation**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.  
**License**: Free to use for research purposes

---

