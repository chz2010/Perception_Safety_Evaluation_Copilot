# YOLO11s Fine-Tuning Summary

## Dataset Overview

Training was performed on a driving-scene dataset with the following class distribution:

| Class | Instances |
|---|---:|
| Car | 714,120 |
| Traffic Sign | 239,961 |
| Traffic Light | 186,301 |
| Person | 91,435 |
| Truck | 30,012 |
| Bike | 7,227 |
| Rider | 4,522 |
| Motor | 3,002 |
| Bus | 1,168 |
| Train | 136 |

Key observation: the dataset is highly imbalanced. Vehicle and infrastructure classes dominate, while vulnerable road user (VRU) classes such as riders, bicycles, and motorcycles have substantially fewer examples. This imbalance is reflected in the final model performance.

## Training Convergence

Across 20 epochs:

- box loss decreased steadily
- classification loss decreased steadily
- distribution focal loss (DFL) decreased steadily
- validation losses also decreased and stabilized

Assessment: the model converged normally, with no strong evidence of overfitting during the training window.

## Overall Detection Performance

| Metric | Value |
|---|---:|
| Precision | ~0.72 |
| Recall | ~0.46 |
| mAP50 | ~0.51 |
| mAP50-95 | ~0.28 |

Interpretation: the model achieves reasonable precision but only moderate recall. In practice, this means detections are often correct when produced, but a meaningful portion of real objects remain undetected.

## Confidence Threshold Analysis

The F1-confidence analysis shows:

- optimal confidence threshold: ~0.256
- maximum F1: ~0.52

Interpretation: the default YOLO operating threshold of 0.25 is already close to the best overall tradeoff for this model. Raising the threshold increases precision but reduces recall quickly. For safety-oriented perception assessment, thresholds materially above 0.25 should be treated cautiously because they can suppress potentially safety-relevant detections.

## Class-Level Performance

### Strongest Classes

| Class | AP50 |
|---|---:|
| Car | 0.769 |
| Traffic Sign | 0.638 |
| Truck | 0.615 |
| Traffic Light | 0.601 |

These classes benefit from strong representation in the training set and show comparatively reliable detection performance.

### Moderate Classes

| Class | AP50 |
|---|---:|
| Bus | 0.594 |
| Person | 0.589 |

Pedestrian detection is usable but still leaves substantial room for improvement.

### Weakest Classes

| Class | AP50 |
|---|---:|
| Rider | 0.422 |
| Bike | 0.421 |
| Motor | 0.407 |

These weaker results are especially important because they affect vulnerable road users and likely reflect both class imbalance and higher visual variability.

## Confusion Matrix Findings

### Best Detected Classes

- Car
- Traffic Light
- Traffic Sign

These classes show the strongest diagonal dominance in the confusion matrix.

### Primary Failure Mode: False Negatives

The dominant failure pattern is missed detections rather than incorrect class assignment.

Examples of missed objects include:

| Class | Missed Objects |
|---|---:|
| Person | 6,165 |
| Bike | 562 |
| Motor | 240 |
| Traffic Light | 9,206 |
| Traffic Sign | 13,677 |

Interpretation: the model is more likely to fail by not detecting an object than by assigning it the wrong class. For automated-driving perception, this is a more serious safety concern than ordinary class confusion.

### Observed Class Confusions

- Car ↔ Truck
- Bus ↔ Truck
- Traffic Sign ↔ Traffic Light

These confusions are plausible given visual similarity, scale variation, and scene complexity.

## Safety-Relevant Findings

The most important safety observation is that vulnerable road users, including pedestrians, cyclists, riders, and motorcyclists, show lower recall than dominant vehicle classes.

This implies:

- higher missed-detection risk
- greater perception uncertainty
- stronger safety relevance in scenarios involving vulnerable road users

From a system-safety perspective, the main perception risk is object not detected at all, rather than object detected but classified incorrectly.

## Implications for Project 3: Perception Safety Evaluation Copilot

These results support the design direction of Project 3. The evaluation framework should emphasize:

### Detection Coverage

- recall analysis
- missed-object statistics
- false-negative trends

### Vulnerable Road Users

- pedestrian performance
- rider performance
- bicycle performance
- motorcycle performance

### Environmental Disturbance Analysis

- rain
- fog
- night
- glare
- occlusion

### Safety Lens Reporting

- missed-object summaries
- vulnerable-road-user impact assessment
- SOTIF-oriented perception limitations
- ISO 26262 downstream safety implications
- ISO/PAS 8800 AI-performance concerns

## Final Engineering Conclusion

The fine-tuned YOLO11s model shows stable convergence and reasonable overall detection performance, with mAP50 around 0.51. It performs well on vehicle and infrastructure classes but shows weaker recall for vulnerable road users. Confusion-matrix analysis indicates that missed detections are a more significant failure mode than class misclassification. From a safety-engineering perspective, future improvement should focus on recall, especially for pedestrians, cyclists, riders, and motorcycles, because these classes represent the most safety-critical perception challenges in automated-driving scenarios.
