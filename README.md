Raw benchmark results from evaluate_all_architecture.py:
```
(.venv) PS E:\AIO\SpamGuard> python e:\AIO\SpamGuard\evaluate_all_architecture.py
Loading sentence-transformer model (intfloat/multilingual-e5-base)...
Model loaded on cuda.

================================================================================
  STARTING FULL EVALUATION SUITE ON: ORIGINAL BIASED DATASET
================================================================================

[Step 1/4] Loading and preparing data from '2cls_spam_text_cls_original.csv'...

[Step 2/4] Building all classifier architectures...
E:\AIO\SpamGuard\.venv\lib\site-packages\sklearn\feature_extraction\text.py:517: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
E:\AIO\SpamGuard\.venv\lib\site-packages\torch\nn\modules\module.py:1762: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `XLMRobertaSdpaSelfAttention.forward`.
  return forward_call(*args, **kwargs)
✅ All models and indexes are built.

[Step 3/4] Running evaluations for each architecture...

--- Testing: MultinomialNB Only ---
Classifying with MultinomialNB Only: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [00:00<00:00, 238.34it/s]

--- Testing: k-NN Only ---
Classifying with k-NN Only:   0%|                                                                                                                  | 0/92 [00:00<?, ?it/s]E:\AIO\SpamGuard\.venv\lib\site-packages\torch\nn\modules\module.py:1762: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `XLMRobertaSdpaSelfAttention.forward`.
  return forward_call(*args, **kwargs)
Classifying with k-NN Only: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [00:01<00:00, 46.13it/s]

--- Testing: Hybrid System ---
Classifying with Hybrid System: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [00:00<00:00, 130.27it/s]

[Step 4/4] Generating detailed performance and timing reports...

------------------------------------------------------------
  DETAILED REPORT FOR: MultinomialNB Only on Original Biased Dataset
------------------------------------------------------------
Overall Accuracy: 81.52%
Total Prediction Time: 0.3802 seconds
Average Prediction Time: 4.1323 ms/message

Classification Report:
              precision    recall  f1-score   support

         ham       0.75      0.96      0.84        46
        spam       0.94      0.67      0.78        46

    accuracy                           0.82        92
   macro avg       0.84      0.82      0.81        92
weighted avg       0.84      0.82      0.81        92


------------------------------------------------------------
  DETAILED REPORT FOR: k-NN Only on Original Biased Dataset
------------------------------------------------------------
Overall Accuracy: 88.04%
Total Prediction Time: 1.9834 seconds
Average Prediction Time: 21.5587 ms/message

Classification Report:
              precision    recall  f1-score   support

         ham       0.82      0.98      0.89        46
        spam       0.97      0.78      0.87        46

    accuracy                           0.88        92
   macro avg       0.90      0.88      0.88        92
weighted avg       0.90      0.88      0.88        92


------------------------------------------------------------
  DETAILED REPORT FOR: Hybrid System on Original Biased Dataset
------------------------------------------------------------
Overall Accuracy: 86.96%
Total Prediction Time: 0.7025 seconds
Average Prediction Time: 7.6357 ms/message

Hybrid Model Usage:
  - MultinomialNB used: 66 times (71.7%)
  - Vector Search (k-NN) used: 26 times (28.3%)
  - Accuracy of NB Triage: 87.88%
  - Accuracy of k-NN Escalation: 84.62%

Classification Report:
              precision    recall  f1-score   support

         ham       0.79      1.00      0.88        46
        spam       1.00      0.74      0.85        46

    accuracy                           0.87        92
   macro avg       0.90      0.87      0.87        92
weighted avg       0.90      0.87      0.87        92


================================================================================
  STARTING FULL EVALUATION SUITE ON: CURRENT AUGMENTED DATASET
================================================================================

[Step 1/4] Loading and preparing data from '2cls_spam_text_cls.csv'...

[Step 2/4] Building all classifier architectures...
E:\AIO\SpamGuard\.venv\lib\site-packages\sklearn\feature_extraction\text.py:517: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
E:\AIO\SpamGuard\.venv\lib\site-packages\torch\nn\modules\module.py:1762: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `XLMRobertaSdpaSelfAttention.forward`.
  return forward_call(*args, **kwargs)
✅ All models and indexes are built.

[Step 3/4] Running evaluations for each architecture...

--- Testing: MultinomialNB Only ---
Classifying with MultinomialNB Only: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [00:00<00:00, 252.74it/s] 

--- Testing: k-NN Only ---
Classifying with k-NN Only:   0%|                                                                                                                  | 0/92 [00:00<?, ?it/s]E:\AIO\SpamGuard\.venv\lib\site-packages\torch\nn\modules\module.py:1762: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `XLMRobertaSdpaSelfAttention.forward`.
  return forward_call(*args, **kwargs)
Classifying with k-NN Only: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [00:01<00:00, 59.07it/s]

--- Testing: Hybrid System ---
Classifying with Hybrid System: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [00:00<00:00, 131.72it/s] 

[Step 4/4] Generating detailed performance and timing reports...

------------------------------------------------------------
  DETAILED REPORT FOR: MultinomialNB Only on Current Augmented Dataset
------------------------------------------------------------
Overall Accuracy: 88.04%
Total Prediction Time: 0.3617 seconds
Average Prediction Time: 3.9311 ms/message

Classification Report:
              precision    recall  f1-score   support

         ham       0.91      0.85      0.88        46
        spam       0.86      0.91      0.88        46

    accuracy                           0.88        92
   macro avg       0.88      0.88      0.88        92
weighted avg       0.88      0.88      0.88        92


------------------------------------------------------------
  DETAILED REPORT FOR: k-NN Only on Current Augmented Dataset
------------------------------------------------------------
Overall Accuracy: 96.74%
Total Prediction Time: 1.5502 seconds
Average Prediction Time: 16.8503 ms/message

Classification Report:
              precision    recall  f1-score   support

         ham       1.00      0.93      0.97        46
        spam       0.94      1.00      0.97        46

    accuracy                           0.97        92
   macro avg       0.97      0.97      0.97        92
weighted avg       0.97      0.97      0.97        92


------------------------------------------------------------
  DETAILED REPORT FOR: Hybrid System on Current Augmented Dataset
------------------------------------------------------------
Overall Accuracy: 95.65%
Total Prediction Time: 0.6954 seconds
Average Prediction Time: 7.5587 ms/message

Hybrid Model Usage:
  - MultinomialNB used: 63 times (68.5%)
  - Vector Search (k-NN) used: 29 times (31.5%)
  - Accuracy of NB Triage: 96.83%
  - Accuracy of k-NN Escalation: 93.10%

Classification Report:
              precision    recall  f1-score   support

         ham       1.00      0.91      0.95        46
        spam       0.92      1.00      0.96        46

    accuracy                           0.96        92
   macro avg       0.96      0.96      0.96        92
weighted avg       0.96      0.96      0.96        92


================================================================================
  ALL EXPERIMENTS COMPLETE
================================================================================
```
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
