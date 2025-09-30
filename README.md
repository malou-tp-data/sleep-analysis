# Sleep Hours and Cognitive Performance Analysis  

## Dataset  
- File: `sleep_deprivation_dataset_detailed.csv`  
- 60 participants, 14 variables (sleep, cognitive tasks, lifestyle measures).  

## Methods  
- Data exploration with **pandas**  
- Visualizations with **seaborn** and **matplotlib**  
- Descriptive statistics  
- Correlation analysis (Pearson)  

## Results  

### Sleep Hours distribution
![Sleep Distribution](figures/distribution_sleeping_hours.png)

### Sleep Hours vs N-Back Accuracy (scatter)
![Scatter Sleep vs Performance](figures/sleep_vs_performance.png)

### Linear Trend (regression)
![Linear Trend](figures/sleep_vs_n_back_accuracy_trend.png)

**Correlation Sleep_Hours vs N_Back_Accuracy = -0.118**  

## Interpretation

There is a very weak negative correlation between sleep hours and N-Back accuracy (r = -0.118).  
This indicates that participants who slept slightly less did not systematically perform worse on the working memory task.  
The effect size is so small that it is likely explained by random variation rather than a meaningful trend.  

This suggests that, in this dataset, sleep duration alone does not have a strong impact on working memory performance.  
It is possible that other variables — such as sleep quality, stress levels, or individual differences in cognitive capacity — play a more important role.  
Therefore, while sleep is crucial for overall cognitive health, this analysis shows that its direct effect on short-term working memory (as measured by the N-Back task) may be limited in this sample.

## How to Run  
```bash
pip install pandas seaborn matplotlib scipy  
python3 analyse.py  

## Next Steps
- Test other cognitive measures (`Stroop_Task_Reaction_Time`, `PVT_Reaction_Time`)
- Extend analysis with regression models including multiple predictors
- Compare across different age groups or genders