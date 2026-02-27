# Confidence Cartography: Teacher-Forced Probability as a False-Belief Sensor in Language Models

**Bryan Sanchez**

Independent Researcher

---

## Abstract

We show that the token-level probabilities a causal language model assigns to its own training text carry interpretable structure about encoded false beliefs. Across a scaling study of seven Pythia models (160M to 12B parameters), model confidence ratios on Mandela Effect items correlate significantly with human false-belief prevalence (Spearman rho = 0.718, p = 0.006 at 1B; rho = 0.652, p = 0.016 at 6.9B; n = 13 items). The signal generalizes to medical misconceptions (88% accuracy at 6.9B, p = 0.01), scales log-linearly with parameter count (71% to 92% on a truth-detection benchmark, r^2 = 0.97), and stabilizes by training step 256. Token-level analysis reveals that true/false sentence pairs diverge at specific "answer" positions, with the win rate of correctly localizing the divergence point rising from 57.5% at 160M to 82.5% at 12B. We identify two distinct confidence regimes: items the model gets right (Regime 1) show early, sharp divergence that is stable across checkpoints, while items the model gets wrong (Regime 2) show late, shallow divergence that oscillates during training. Comparing base and instruction-tuned models, we find that RLHF preserves these regime assignments almost perfectly (Fisher's exact p = 2.58 x 10^{-21}) without repairing Regime 2 errors. We interpret these results as evidence that teacher-forced confidence tracks the transmissibility of beliefs in training corpora rather than their factual truth. As a practical application, we show that oracle-targeted resampling at low-confidence token positions can recover correct answers on items where greedy decoding fails.

---

## 1. Introduction

Language models are trained to predict the next token in a sequence. The probability they assign to each token under this teacher-forcing objective is rarely examined as a signal in its own right: it is the loss to be minimized, not a quantity to be reported at inference. Yet these probabilities encode something real. They reflect the degree to which each token was predictable given the preceding context and everything the model absorbed during training.

This paper asks what teacher-forced confidence reveals about false beliefs. When a model has absorbed a false claim (that the Monopoly Man wears a monocle, say, or that "Berenstain Bears" is spelled with an *e*), does it assign that claim higher or lower probability than the corrected version? And does the degree of confidence track how widely the false belief is held among humans?

The Mandela Effect provides a natural testbed. These are claims that are false, well-documented as false, and that vary in prevalence across the human population. If model confidence tracks the cultural footprint of beliefs rather than their truth value, then items with high human false-belief rates should show relatively higher model confidence on the wrong version, and vice versa.

We find that this is what happens. The rank correlation between model confidence ratios and human false-belief prevalence is statistically significant at six of seven model sizes tested (peak rho = 0.718, p = 0.006 at 1B; rho = 0.652, p = 0.016 at 6.9B). The signal generalizes to medical misconceptions, where it achieves 88% binary accuracy at 6.9B without any fine-tuning or domain adaptation.

Beyond the correlation result, we provide a mechanistic account of the signal. Token-level divergence analysis reveals that true/false sentence pairs diverge at specific positions in the sequence, and that this divergence concentrates at "answer" tokens rather than spreading diffusely across the sentence. The model's item-level behavior splits into two regimes: items where the model assigns higher confidence to the true version (Regime 1) and items where it assigns higher confidence to the false version (Regime 2). These regimes differ in their divergence dynamics, their stability across training checkpoints, and their response to RLHF. Regime membership turns out to be predictable from surface-level properties of the items (whether they reference fiction, whether they are cultural rather than factual), consistent with the transmissibility interpretation.

**Contributions:**

1. We introduce confidence cartography, the systematic mapping of teacher-forced token probabilities across knowledge domains, as a method for characterizing what language models have absorbed from training data.
2. We provide the first direct calibration of model confidence ratios against human false-belief prevalence, demonstrating significant correlation (rho = 0.652, p = 0.016, n = 13).
3. We identify two mechanistic regimes in model confidence (early-diverging/stable vs. late-diverging/oscillatory) and show they are preserved under instruction tuning.
4. We show the signal generalizes to medical claims (88% accuracy) and is significant at six of seven model sizes tested.

---

## 2. Background

### 2.1 Teacher-Forced Probability

In autoregressive language model training, the model receives the full target sequence and predicts each token given all preceding tokens. The probability assigned to the actual next token, P(t_i | t_1, ..., t_{i-1}), is the quantity whose negative log is minimized during training. At inference, this quantity can be computed for any fixed text by a single forward pass, making it cheap to extract.

Prior work has used this quantity primarily as a perplexity measure for model evaluation. We treat it instead as a signal with interpretable structure across different types of claims.

### 2.2 Probing and Interpretability

Work on locating factual knowledge in language models has focused on probing internal representations. Meng et al. (2022) use causal tracing to identify specific MLP layers that store factual associations. Burns et al. (2023) discover latent knowledge directions in activation space using unsupervised contrast-consistent search. Marks and Tegmark (2024) find linear "truth directions" in residual streams.

Our approach is complementary: we require no learned probe, no labeled training data, and no access to internal activations beyond the output logits. Teacher-forced confidence is a single scalar per token, directly interpretable, and applicable to any autoregressive model without modification.

### 2.3 Calibration and Uncertainty

Calibration research asks whether a model's stated probability matches the empirical frequency of correctness (Guo et al., 2017). Work on verbal uncertainty probes whether models can express calibrated uncertainty in natural language (Lin et al., 2022; Kadavath et al., 2022). Kuhn et al. (2023) introduce semantic entropy as an uncertainty estimator that accounts for meaning-preserving paraphrases.

Our focus is different. We ask whether confidence *differences* between paired true and false claims carry signal, and whether those differences correlate with external measures of human belief prevalence.

### 2.4 The Mandela Effect

The Mandela Effect was named after the widespread false memory that Nelson Mandela died in prison in the 1980s, popularized by Fiona Broome around 2009-2010. YouGov (2022) conducted nationally representative polling on Mandela Effect items with a sample of 1,000 US adults, providing empirical prevalence estimates that serve as ground truth for human false-belief rates in this study.

---

## 3. Method

### 3.1 Teacher-Forced Confidence Extraction

Given a fixed text T = (t_1, t_2, ..., t_n), we perform a single forward pass through a causal language model and extract, for each position i, the probability the model assigns to the actual token t_i given all preceding tokens:

    c_i = P_model(t_i | t_1, ..., t_{i-1}) = softmax(logits_i)[token_id(t_i)]

We compute summary statistics over the sequence: mean confidence (mu_c), standard deviation (sigma_c), and per-token entropy H_i = -sum_v P(v | t_{<i}) log P(v | t_{<i}).

For a pair of claims (true version T+, false version T-), the confidence ratio is:

    R = mu_c(T-) / (mu_c(T+) + mu_c(T-))

This ratio lies in [0, 1]. Values above 0.5 indicate higher mean confidence on the false version; values below 0.5 indicate higher confidence on the true version. A "win" is when R < 0.5 (the model prefers the true version).

### 3.2 Models

We use the Pythia model suite (Biderman et al., 2023): seven sizes spanning 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B parameters. Pythia provides consistent architecture and training data (the Pile; Gao et al., 2020) across all sizes, enabling clean scaling analysis. All models are evaluated in their base, non-instruction-tuned form.

For the RLHF comparison experiment, we use Qwen 2.5-7B and Qwen 2.5-7B-Instruct (Qwen Team, 2024), which share the same base architecture but differ in whether instruction tuning and RLHF were applied. We additionally validate on Qwen 2.5-32B to confirm generalization beyond the Pythia family.

### 3.3 Mandela Effect Items

We use 13 Mandela Effect items with human prevalence estimates. The items span misquoted film lines (6 items: Star Wars, Snow White, Jaws, Forrest Gump, The Silence of the Lambs, Casablanca), misremembered cultural artifacts (4 items: Berenstain Bears, Curious George, Monopoly Man, Fruit of the Loom), misattributed proverbs (2 items), and a color-name confusion (chartreuse). Four items have YouGov (2022) nationally representative polling data (n = 1,000 US adults); the remaining nine use proxy prevalence estimates derived from web-hit ratios calibrated against the YouGov items.

Selected items with confirmed prevalence:

| Item | Correct version | Common false version | Human false-belief rate |
|------|----------------|---------------------|------------------------|
| Star Wars quote | "No, I am your father" | "Luke, I am your father" | 62% |
| Berenstain Bears | Berenstain | Berenstein | 61% |
| Monopoly Man | No monocle | Wears a monocle | 58% |
| Fruit of the Loom | No cornucopia | Cornucopia present | 55% |

For each item, we construct matched sentence pairs expressing the correct and false versions, controlling for sentence length and syntactic structure.

### 3.4 Medical Validation

We constructed 25 true/false pairs across seven medical domains: anatomy (6 pairs), disease and pathology (6), public health (4), pharmacology (3), nutrition (2), neuroscience (2), and genetics (2). Pairs were selected to include widely circulated misconceptions alongside correct factual alternatives. A prediction is counted as correct if mu_c(T+) > mu_c(T-).

### 3.5 Token-Level Divergence Analysis

For matched true/false sentence pairs that share a common prefix and diverge at some position d, we compute the KL divergence between the model's output distributions at position d conditioned on the true vs. false continuations. Define:

    div_delta(d) = KL(P_model(. | T+[:d]) || P_model(. | T-[:d]))

We say the divergence is "correctly localized" if the maximum div_delta occurs at or near the first semantically meaningful difference between the two versions. The win rate is the fraction of items where this localization is correct.

### 3.6 Targeted Resampling

For a given generation, we identify the k tokens with lowest top-1 confidence. We resample only those positions, drawing from the model's conditional distribution at each low-confidence point, and select the completion with the highest global mean confidence. We compare this "oracle" variant (where error positions are known) and a "blind" variant (where only confidence ranks are available) against uniform best-of-N, which regenerates the full sequence N times.

---

## 4. Results

### 4.1 Baseline Confidence Fingerprints

Teacher-forced confidence distinguishes knowledge categories. Simple factual claims such as geographic facts and unit conversions show high mean confidence (mu_c = 0.71) and low entropy. False statements show systematically lower mean confidence (mu_c = 0.48) than matched true alternatives. Contested claims on policy or values produce wider entropy distributions than settled empirical questions. These patterns hold across all seven model sizes, with separation increasing at scale.

### 4.2 Truth Detection Scales with Model Size

On a 40-item true/false benchmark spanning geography, science, history, and policy, binary classification by confidence ratio achieves:

| Model size | Accuracy | Cohen's d |
|-----------|---------|-----------|
| 160M | 71% | 0.31 |
| 410M | 76% | 0.48 |
| 1B | 81% | 0.62 |
| 1.4B | 83% | 0.69 |
| 2.8B | 87% | 0.81 |
| 6.9B | 90% | 0.92 |
| 12B | 92% | 0.98 |

Accuracy scales log-linearly with parameter count (r^2 = 0.97). The relationship holds within each sub-domain (geography, science, history, policy) tested separately.

### 4.3 Mandela Effect Calibration

The primary result. Model confidence ratios on 13 Mandela Effect items correlate significantly with human false-belief prevalence:

| Model size | Spearman rho | p-value |
|-----------|-------------|---------|
| 160M | 0.561 | 0.046* |
| 410M | 0.652 | 0.016* |
| 1B | **0.718** | **0.006** |
| 1.4B | 0.578 | 0.039* |
| 2.8B | 0.473 | 0.102 |
| 6.9B | 0.652 | 0.016* |
| 12B | 0.619 | 0.024* |

Six of seven model sizes reach p < 0.05, with the peak at 1B (rho = 0.718, p = 0.006). Items where more humans hold the false belief correspond to items where the model assigns higher relative confidence to the false version.

The correlation holds in both raw and context-embedded prompt framings (Pearson r = 0.92 between framing variants, p < 0.001), confirming that the result is not an artifact of surface-level phrasing.

The 2.8B model is a notable exception (p = 0.10). We have no clear explanation for this dip; it may reflect idiosyncratic features of that checkpoint's training dynamics.

### 4.4 Medical Domain Generalization

Without fine-tuning or domain adaptation, teacher-forced confidence achieves 88% binary classification accuracy on 25 medical true/false pairs at Pythia-6.9B (p = 0.01 by binomial test). The signal is present across all seven medical sub-domains, with anatomy showing the strongest separation (91%) and public health the weakest (75%). The weaker public health result is consistent with a greater volume of conflicting guidance in general web text, which would produce less consistent confidence signals.

Medical accuracy scales with model size, rising from near-chance at 160M to the 88% figure at 6.9B. This out-of-domain generalization is the strongest evidence that the confidence signal reflects something general about how models encode frequency of exposure, not something specific to the Mandela Effect items.

### 4.5 Early Emergence in Training

The Mandela Effect confidence pattern stabilizes early. Across 13 checkpoints of Pythia-1.4B (training steps 1 through 143,000), the confidence ratio pattern at step 256 already correlates with the final checkpoint at Pearson r > 0.9.

This is a striking result. By step 256, the model has seen only a tiny fraction of its total training data, yet the relative confidence ordering across Mandela Effect items is already set. The implication is that the signal reflects low-level statistical properties of the training corpus (roughly, token co-occurrence frequencies) rather than abstract semantic representations that build up slowly during training.

### 4.6 Token-Level Divergence Localization

Where in a sentence do models distinguish true from false? We analyzed 40 matched true/false pairs where the sentences share an identical prefix and diverge at a specific position. The "win rate" is the fraction of items where the model's maximum confidence divergence falls at or near the first semantically meaningful difference.

| Model size | Win rate | Mean divergence delta |
|-----------|----------|----------------------|
| 160M | 57.5% | 0.012 |
| 410M | 75.0% | 0.098 |
| 1B | 70.0% | 0.122 |
| 1.4B | 77.5% | 0.196 |
| 2.8B | 82.5% | 0.208 |
| 6.9B | 80.0% | 0.230 |
| 12B | 82.5% | 0.232 |

At 160M, the model barely localizes the divergence (57.5%, close to chance if the divergence point occurs in the second half of the sentence). By 410M and above, localization is substantially above chance. The divergence magnitude (measured as the KL between output distributions at the divergence point) increases roughly log-linearly with model size.

This finding adds mechanistic detail to the confidence-ratio result. The model's preference for true vs. false versions is not a diffuse property of the whole sentence; it concentrates at specific tokens, the ones that actually differ between the two versions.

### 4.7 Two Regimes of Model Confidence

When we split the 40 truth-detection items by whether the model assigns higher confidence to the true version (Regime 1, "model gets it right") or the false version (Regime 2, "model gets it wrong"), two qualitatively different patterns emerge.

At Pythia-6.9B with n = 99 items (truth + Mandela + medical):

**Regime 1 (n = 65 items, model correct):**
- Win rate: 80.0%
- Mean divergence delta: +0.193
- Prefix delta: +5.87 x 10^{-5} (nearly zero before the divergence point)
- Suffix delta: +0.080 (large, concentrated at answer tokens)

**Regime 2 (n = 34 items, model wrong):**
- Win rate: 32.4%
- Mean divergence delta: -0.073
- Prefix delta: -1.68 x 10^{-4}
- Suffix delta: -0.065

In Regime 1, the model's divergence is sharp, early, and concentrated at the answer position. In Regime 2, the divergence is smaller in magnitude, and what divergence exists goes in the wrong direction (higher confidence on the false version).

Checkpoint stability analysis reinforces the distinction. Regime 1 items converge to their final confidence ratio early in training and stay there. Regime 2 items show higher variance across checkpoints, with more reversals (flipping between correct and incorrect preference) during training. At Pythia-1B, Regime 1 items show variance of 0.0076 across 13 checkpoints, while Regime 2 items show variance of 0.0089 (16% higher, though the difference does not reach significance at any individual model size; Levene's test p > 0.2 for all sizes).

### 4.8 RLHF Preserves Confidence Structure

One natural question is whether instruction tuning and RLHF repair Regime 2 errors by teaching the model to prefer true versions of claims. We compared Qwen 2.5-7B (base) against Qwen 2.5-7B-Instruct (RLHF) on 98 items.

The short answer is no. Regime assignments are almost perfectly preserved:

| Transition | Count |
|------------|-------|
| R1 → R1 (stable correct) | 71 |
| R2 → R2 (stable wrong) | 25 |
| R2 → R1 (fixed by RLHF) | 2 |
| R1 → R2 (broken by RLHF) | 0 |

Fisher's exact test: p = 2.58 x 10^{-21}. Regime membership is almost entirely determined by the base model; RLHF barely touches it.

The quantitative shifts are small and non-significant:

| Regime | Base win rate | Instruct win rate | Confidence shift | Wilcoxon p |
|--------|-------------|-------------------|------------------|-----------|
| R1 (n=64) | 89.1% | 85.9% | +0.002 | 0.556 |
| R2 (n=34) | 41.2% | 41.2% | -0.001 | 0.209 |

RLHF does not significantly change the confidence gap in either regime. The instruction-tuned model is about as confidently wrong on Regime 2 items as the base model. This result is consistent with the view that RLHF primarily shapes the model's generation style and refusal behavior rather than correcting the frequency-driven confidence patterns inherited from pretraining.

### 4.9 What Predicts Regime Membership?

If Regime 2 items are the ones where the model is confidently wrong, what makes them different? We tested whether surface-level properties of the items predict regime membership using logistic regression with domain and lexical features.

The top predictors of Regime 2 membership:

| Feature | Coefficient | Direction |
|---------|------------|-----------|
| references_fiction | +0.538 | R2 |
| is_cultural | +0.428 | R2 |
| has_person | +0.172 | R2 |
| is_medical | -0.182 | R1 |
| references_brand | +0.075 | R2 |

Items that reference fictional characters or cultural artifacts (movie quotes, brand logos, fictional details) are more likely to be in Regime 2. Medical and factual items are more likely to be in Regime 1. The domain features alone yield AUC = 0.72 for regime classification; adding lexical features raises this to AUC = 0.82.

This makes sense under the transmissibility interpretation. Fictional and cultural items generate enormous volumes of informal, often inaccurate text online (fan discussions, memes, social media posts). The false version of "Luke, I am your father" appears far more often in casual text than the correct "No, I am your father" because the misquote is more culturally salient. Medical facts, by contrast, appear disproportionately in edited sources (textbooks, medical databases, Wikipedia articles with citation requirements), so the true version dominates in the training corpus.

Conversely, the model's own confidence features (mean confidence ratio, scaling slope) perform poorly at predicting regime membership (AUC = 0.59 and 0.65 respectively). The model knows which version it prefers, but its preference magnitude is not a reliable indicator of whether that preference is correct. This is the same observation that makes calibration hard in general: a confidently held belief can be either right or wrong, and confidence magnitude alone does not distinguish the two.

### 4.10 Targeted Resampling

As a practical application, we test whether confidence can guide more efficient generation. The idea is to identify tokens where the model is least confident and resample only those positions rather than regenerating the entire sequence.

We tested three strategies on items where greedy decoding produced an incorrect answer:

- **Oracle-targeted (k=5, 10):** Resample only at positions known (from the paired true version) to be error positions. This is cheating, but establishes an upper bound.
- **Blind-targeted (k=5, 10):** Resample at the k positions with lowest confidence, without access to the true version.
- **Best-of-N (N=5, 10):** Regenerate the full sequence N times and pick the one with highest mean confidence.

Results on 89 items at Pythia-6.9B (number of items flipped from incorrect to correct):

| Strategy | Items fixed |
|----------|-----------|
| Greedy (baseline) | 0 |
| Oracle-5 | 12 |
| Oracle-10 | 11 |
| Best-of-5 | 8 |
| Best-of-10 | 7 |
| Blind-5 | 2 |
| Blind-10 | 2 |

Oracle targeting outperforms best-of-N: it fixes 50% more items while resampling only a fraction of the tokens. The practical blind variant, however, underperforms best-of-N. The gap between oracle and blind performance indicates that confidence rank alone is an imperfect proxy for locating error-causing tokens. Low-confidence positions are often function words or subword boundaries rather than the semantically critical tokens where errors originate.

This suggests that combining confidence with other signals (entropy, divergence from a reference model, position relative to the claim's key entity) could close the gap between oracle and blind targeting. We leave this to future work.

---

## 5. Discussion

### 5.1 Confidence Tracks Transmissibility, Not Truth

The central claim of this paper is that teacher-forced confidence reflects how commonly a particular formulation appeared in training data (its transmissibility) rather than whether it is factually correct. Several lines of evidence support this interpretation:

1. The Mandela Effect correlation (rho = 0.652) directly links model confidence to human false-belief prevalence, which in turn reflects how widely the false version circulates in culture and, by extension, in web-scraped training data.

2. The signal emerges by training step 256, long before the model has developed sophisticated representations. This is consistent with the model picking up on token co-occurrence frequencies rather than building an internal fact-checking mechanism.

3. Regime 2 items are disproportionately fictional and cultural, exactly the domains where informal, inaccurate text is most abundant online.

4. RLHF does not repair Regime 2 errors, consistent with RLHF operating on generation style rather than on the frequency-driven confidence patterns set during pretraining.

### 5.2 Relationship to Probing Methods

Burns et al. (2023) show that unsupervised probes can discover "truth directions" in model activations. Marks and Tegmark (2024) find similar linear structure using supervised probes. Our results suggest that a weaker version of this signal is available at the output level without any probing: the model's own token probabilities carry crude but consistent information about belief status.

The key difference is that probing methods can separate truth from frequency because they access internal representations that may disentangle the two. Our output-level signal cannot make that separation. A claim can be highly confident and wrong (Regime 2) or uncertain and right. This is a real limitation: confidence cartography maps what the model absorbed, not what is true.

### 5.3 Implications for Alignment

The RLHF result (Section 4.8) has a concrete takeaway for alignment work. Instruction tuning and RLHF reshape how models generate text, making them more helpful, less toxic, and more likely to refuse harmful requests. But they do not reach into the base model's confidence structure to correct false beliefs. The 25 items that Regime 2 errors persisted on after RLHF are items where the model will, under the right prompting, confidently produce the wrong answer, because the wrong version is baked into the pretraining statistics.

This is consistent with observations from Ouyang et al. (2022) that RLHF improves helpfulness and reduces toxicity without necessarily improving factual accuracy. Our contribution is to give this observation a specific, measurable form: the confidence ratio on paired claims before and after RLHF.

### 5.4 Limitations

**Sample size.** The Mandela Effect analysis uses n = 13 items, of which only 4 have nationally representative prevalence data (YouGov, 2022). The remaining 9 use proxy estimates from web-hit ratios, which introduce noise. The resulting confidence intervals on the correlation estimates are wide.

**Prevalence estimation.** The proxy prevalence measures are a known weakness. Web-hit ratios conflate search interest with belief prevalence, and different estimation procedures could yield different values. The correlation result rests on the assumption that these proxies are at least ordinally correct.

**Single training corpus.** All Pythia models were trained on the Pile. The relationship between confidence and transmissibility may differ for models trained on different data mixtures. The Qwen validation provides some reassurance, but a systematic test across training corpora would strengthen the claim.

**Medical validation.** The 25 medical pairs were hand-curated, and selection effects cannot be ruled out. A blind selection procedure (e.g., drawing from an existing medical misconception database) would be more convincing.

**Targeted resampling.** The blind variant of targeted resampling underperforms best-of-N, limiting its practical utility in its current form. The oracle variant demonstrates the ceiling but requires knowledge of error positions that is not available in practice.

### 5.5 Future Directions

Several extensions follow naturally from these results. First, the Mandela Effect correlation should be tested with a larger item set using rigorously estimated prevalence (a dedicated survey rather than proxy measures). Second, the regime analysis could be extended to track how fine-tuning on curated factual data shifts Regime 2 items toward Regime 1. Third, the combination of confidence with other uncertainty signals (semantic entropy from Kuhn et al. 2023, probe-based truth directions from Burns et al. 2023) might close the oracle/blind gap in targeted resampling. Fourth, confidence cartography applied at scale to a model's full training corpus could produce a map of which knowledge domains the model has absorbed accurately vs. which domains contain high rates of false confidence.

---

## 6. Conclusion

Teacher-forced confidence, the probability a language model assigns to its own training text, is a cheap and model-agnostic signal that carries interpretable structure about encoded beliefs. It correlates with human false-belief prevalence across Mandela Effect items (rho up to 0.718 at 1B, significant at 6 of 7 model sizes), generalizes to medical misconceptions without domain adaptation, and scales consistently from 160M to 12B parameters.

The signal is best understood as measuring the transmissibility of claims in training data rather than their factual accuracy. Items the model gets wrong (Regime 2) are disproportionately drawn from fictional and cultural domains where informal, inaccurate text dominates the training corpus. These errors are not fixed by RLHF.

Token-level analysis shows that confidence differences concentrate at specific "answer" positions rather than spreading across sentences, and that the divergence pattern differs qualitatively between items the model gets right (sharp, stable) and items it gets wrong (shallow, oscillatory). These mechanistic details may prove useful for developing more targeted interventions than the blunt approaches (retraining, filtering) currently available.

The practical ceiling of this signal is set by a fundamental limit: output probabilities cannot distinguish high frequency from high truth. Closing that gap requires either access to internal representations (as in probing methods) or external grounding. Within that limit, confidence cartography provides a fast, zero-shot diagnostic that requires nothing beyond a forward pass.

---

## References

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U., Raff, E., Skowron, A., Sutawika, L., and van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*.

Burns, C., Ye, H., Klein, D., and Steinhardt, J. (2023). Discovering latent knowledge in language models without supervision. In *Proceedings of the 11th International Conference on Learning Representations (ICLR 2023)*.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. (2020). The Pile: An 800GB dataset of diverse text for language modeling. arXiv:2101.00027.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*, PMLR 70, 1321-1330.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., Johnston, S., El-Showk, S., Jones, A., Elhage, N., Hume, T., Chen, A., Bai, Y., Bowman, S., Fort, S., Ganguli, D., Hernandez, D., Jacobson, J., Kernion, J., Kravec, S., Lovitt, L., Ndousse, K., Olsson, C., Ringer, S., Amodei, D., Brown, T., Clark, J., Joseph, N., Mann, B., McCandlish, S., Olah, C., and Kaplan, J. (2022). Language models (mostly) know what they know. arXiv:2207.05221.

Kuhn, L., Gal, Y., and Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In *Proceedings of the 11th International Conference on Learning Representations (ICLR 2023)*.

Lin, S., Hilton, J., and Evans, O. (2022). Teaching models to express their uncertainty in words. *Transactions on Machine Learning Research (TMLR)*.

Marks, S. and Tegmark, M. (2024). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In *Proceedings of the 1st Conference on Language Modeling (COLM 2024)*.

Meng, K., Bau, D., Andonian, A., and Belinkov, Y. (2022). Locating and editing factual associations in GPT. In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., and Lowe, R. (2022). Training language models to follow instructions with human feedback. In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*.

Qwen Team. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.

YouGov. (2022). The Mandela Effect: Survey of US adults on popular false memories. Conducted August 23-26, 2022. Sample: 1,000 US adult citizens. Available at: https://today.yougov.com/entertainment/articles/43634-measuring-mandela-effect-false-memory-yougov-poll

---

## Appendix A: Mandela Effect Item Details

Full text of all paired prompts for the 13 Mandela Effect items, in both raw and context-embedded framings, is available in the project repository alongside the confidence extraction code.

**Item categories:**

- **Misquoted films (6):** Star Wars (Darth Vader line), Snow White (mirror line), Jaws ("bigger boat"), Forrest Gump ("box of chocolates"), The Silence of the Lambs ("Hello, Clarice"), Casablanca ("Play it again, Sam")
- **Cultural artifacts (4):** Berenstain Bears spelling, Curious George tail, Monopoly Man monocle, Fruit of the Loom cornucopia
- **Proverbs (2):** "Money is the root of all evil" (vs. "love of money"), "Curiosity killed the cat" (original form includes "but satisfaction brought it back")
- **Color names (1):** Chartreuse (actually yellow-green, commonly misremembered as pink/magenta)

## Appendix B: Medical Claim Pairs

The 25 true/false medical claim pairs cover anatomy (6), disease and pathology (6), public health (4), pharmacology (3), nutrition (2), neuroscience (2), and genetics (2). Per-model classification results are available in the project repository.

## Appendix C: Targeted Resampling Algorithm

    Input: prompt P, model M, k (fraction of tokens to resample), N (candidates)

    1. Generate base completion C from M given P
    2. Extract per-token confidences c_1, ..., c_T for C
    3. Identify low-confidence positions L = {i : c_i < quantile(c, k)}
    4. For n = 1 to N:
         Copy C to candidate C_n
         For each position i in L (left to right):
             Sample t_i from P_M( . | P, C_n[:i] )
             Set C_n[i] = t_i
         Compute score(C_n) = mean(c_i for all i in C_n)
    5. Return the candidate with highest score

## Appendix D: Regime Classification Features

Logistic regression for regime membership was trained on 99 items using the following feature groups:

- **Domain features:** is_cultural, is_medical, is_factual, is_proverb, is_language
- **Lexical features:** references_fiction, references_brand, has_person, has_quote, text_length, mean_word_frequency
- **Transmissibility features:** web_hit_ratio (false/true search results ratio)

The domain + lexical model (AUC = 0.82) outperforms the model-based signal (6.9B confidence ratio AUC = 0.59) for regime prediction, consistent with the interpretation that regime membership is determined by properties of the training data distribution rather than by properties the model can self-diagnose.
