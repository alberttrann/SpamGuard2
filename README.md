
The following table summarizes the performance of all key architectures on a consistent hold-out test set of 92 messages.

| Model ID | Classifier Architecture | Training Data | Overall Accuracy | Total Time (s) | Avg. Time / Msg (ms) | Spam Recall | Spam Precision | Spam F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `GaussianNB` (Hybrid) | Original (Biased) | 59.78% | N/A | N/A | 0.61 | 0.60 | 0.60 |
| **2** | `MultinomialNB` Only | Original (Biased) | 81.52% | **0.380 s** | **4.13 ms** | 0.67 | **0.94** | 0.78 |
| **3** | `k-NN Vector Search` Only | Original (Biased) | 88.04% | 1.983 s | 21.56 ms | 0.78 | 0.97 | 0.87 |
| **4** | **`Hybrid System`** | Original (Biased) | 86.96% | 0.703 s | 7.64 ms | 0.74 | **1.00** | 0.85 |
| | | | | | | | | |
| **5** | `MultinomialNB` Only | **Augmented** | 88.04% | **0.362 s** | **3.93 ms** | **0.91** | 0.86 | 0.88 |
| **6** | `k-NN Vector Search` Only | **Augmented** | **96.74%** | 1.550 s | 16.85 ms | **1.00** | 0.94 | **0.97** |
| **7**| **`Hybrid System`** | **Augmented** | 95.65% | 0.695 s | 7.56 ms | **1.00** | 0.92 | 0.96 |

*(Total and Average times are for classifying all 92 messages in the test set.)*

---

### **In-Depth Analysis and Conclusions**

#### 1. Architectural Choice is the Primary Driver of Performance

The most significant performance leap came from abandoning the mathematically inappropriate `GaussianNB` model (Model 1, 59.78% Acc) in favor of `MultinomialNB` (Model 2, 81.52% Acc) on the same biased dataset. This **+21.74%** gain underscores that selecting the correct model for the data's characteristics is the most critical decision.

#### 2. Data Augmentation Unlocks the Model's Full Potential

While architecture provided the biggest initial jump, data quality was key to achieving excellence. Training the `Hybrid System` on the augmented data (Model 7) instead of the biased data (Model 4) pushed its accuracy from 86.96% to **95.65%** and, most importantly, improved its **Spam Recall from a mediocre 74% to a perfect 100%**. Better data allowed the model to be both more accurate and more effective at its core task.

#### 3. Quantifying the Speed vs. Accuracy Trade-Off

The final comparison between the three architectures on the augmented data provides the clearest picture:

*   **The Speed King (`MultinomialNB` Only):** At a total time of **0.36 seconds** (~4ms/msg), this model is the undisputed champion of efficiency. However, its 88.04% accuracy leaves a significant performance gap.

*   **The Accuracy King (`k-NN` Only):** At **96.74% accuracy**, this model is the most intelligent. But this intelligence comes at a steep price: **1.55 seconds** of total prediction time (~17ms/msg), making it over **4 times slower** than the Naive Bayes model.

*   **The Optimal Engine (`Hybrid System`):** This is where the engineering brilliance lies.
    *   **Performance:** It achieves **95.65% accuracy**, sacrificing only a single percentage point compared to the Accuracy King. Crucially, it matches the k-NN model's **perfect 1.00 Spam Recall**.
    *   **Efficiency:** It completes the task in just **0.70 seconds** (~7.6ms/msg). This makes it **2.2 times faster** than the pure k-NN approach.

    The detailed usage statistics reveal the secret to its success:
    *   `MultinomialNB used: 63 times (68.5%)`
    *   `Vector Search (k-NN) used: 29 times (31.5%)`

    The fast, lightweight `MultinomialNB` model handled over two-thirds of the workload instantly and with a **96.83% accuracy rate** on those triage decisions. The expensive Vector Search was reserved for only the most difficult ~30% of cases, where its semantic power was most needed.

**Final Conclusion:** The `Hybrid System` is demonstrably the superior architecture for a production environment. It achieves the accuracy of a state-of-the-art semantic search model while maintaining an operational speed closer to that of a simple classical model. It successfully proves the value of a tiered, confidence-based approach, leveraging the strengths of each component to create a system that is both exceptionally intelligent and highly efficient.
