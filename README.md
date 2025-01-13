# **Psychological State Classification**

[**Kedro Visualization**](https://anfrejter.github.io/)

## **Project Overview**

 This project focuses on a comprehensive dataset capturing physiological, behavioral, and environmental data from biosensors to study students' psychological states during educational activities. It lays the foundation for advanced machine learning research, aiming to develop models capable of real-time analysis of stress, emotional engagement, and focus. Key features include heart rate variability, EEG power bands, environmental factors like noise and light, and behavioral metrics such as focus duration. By offering insights into well-being and engagement, the dataset aims to advance mental health support and optimize learning experiences through innovative applications of biosensor technology.

 **Dataset**: [Psychological State Identification Dataset on Kaggle](https://www.kaggle.com/datasets/ziya07/psychological-state-identification-dataset)

---

## Column description

| No.| Column      | Description | Type |
| -- | ----------- | ----------- | -- |
| 1. | ID | Unique identifier for each participant. | Integer|
| 2. | Time | Timestamp indicating when the data was recorded.| Datetime|
| 3. | HRV (ms) | Heart Rate Variability-Indicates stress and relaxation states. | Float
| 4. | Gen(GSR) (μS) | Galvanic Skin Response-Reflects stress through changes in skin conductivity. |Float |
| 5. | EEG Power Bands | Captures brain activity in Delta, Alpha, and Beta bands. | String|
| 6. | Blood Pressure (mmHg) | Measures cardiovascular response. |String |
| 7. | Oxygen Saturation (%) | Indicates oxygen levels in the blood. |Float |
| 8. | Heart Rate (BPM) | Shows physical or emotional excitement. |Integer |
| 9. | Ambient Noise (dB) | Noise intensity during educational activities. |Float |
| 10. | Cognitive Load | Low, Moderate, High- reflects mental effort. |Categorical |
| 11. | Mood State | Happy, Neutral, Sad, Anxious- represents emotional conditions. |Categorical | 
| 12. | Psychological State | Stressed, Relaxed, Focused, Anxious- inferred from biosensor data.|Categorical | 
| 13. | Respiration Rate (BPM) | Measures breathing activity |Integer | 
| 14. | Skin Temp (°C) | Indicates stress or comfort levels. |Float | 
| 15. | Focus Duration (s) | Time spent in sustained attention on a task. |Integer |
| 16. | Task Type | Lecture, Group Discussion, Assignment, Exam. |Categorical |
| 17. | Age |Participant age. |Integer |
| 18. | Gender | Male, Female, Other |Categorical |
| 19. | Educational Level | High School, Undergraduate, Postgraduate |Categorical |
| 20. | Study Major |Science, Arts, Engineering |Categorical |

---

## **Project Tools**

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

This project is built using **Kedro** and **Kedro-Viz**, generated with `kedro 0.19.10`.

---
