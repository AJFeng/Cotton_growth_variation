# Cotton_growth_variation
Quantifcation of interactions of soil conditions, plant available water and weather conditions on crop development and production is the key for optimizing feld management to achieve optimal production. The goal of this study was to quantify the efects of soil and weather conditions on cotton development and production using temporal aerial imagery data, weather and soil apparent electrical conductivity (ECa) of the feld.

Feng, A., Zhou, J., Vories, E.D. et al. Quantifying the effects of soil texture and weather on cotton development and yield using UAV imagery. Precision Agric (2022). https://doi.org/10.1007/s11119-022-09883-6

https://trebuchet.public.springernature.app/get_content/dad7191f-3d41-444f-a986-cc6efba45207

## Data processing and results
The overall procedure of this study is given in Fig. 1 that shows the flowchart of the data and analysis used in this study. Firstly, high spatial resolution ECa data, high temporal resolution weather data and monthly UAV image data were collected. Features that reflected soil water holding capacity, crop water requirement and crop growing status were extracted. A data fusion method was developed to integrate different spatio-temporal features representing soil, water stress and crop growth. Pearson correlation (r), analysis of variance (ANOVA), eXtreme Gradient Boosting (XGBoost) models and feature contribution analysis were used to quantify the relationships between crop response derived from UAV images and environments (soil texture and weather).
![alt text](https://github.com/AJFeng/Cotton_growth_variation/blob/main/figures/Flowchart.png)

The Ks maps and the data fusion method of connecting soil, image features and Ks maps. Ks was calculated for each position in the 38 (or 37 in 2018) × 63 spatial raster with each Ks map representing a day.
![alt text](https://github.com/AJFeng/Cotton_growth_variation/blob/main/figures/data_fusion.png)

Ks maps on (a) August 8, (b) August 10 and (c) August 13. The legend in (c) is also used in (a)-(b). The white circle in (c) marks a plot that had a 22 mm irrigation applied on August 6. (d) NDVI map collected on Aug 14. (e) relationship between NDVI and the Aug 13 Ks map. Different lower-case letters indicate a significant difference at the 5% level of Tukey’s honest significant difference test. The Ks groups were split based on quartiles of the Aug 13 Ks map
![alt text](https://github.com/AJFeng/Cotton_growth_variation/blob/main/figures/Ks_NDVI.png)

NDVI maps collected on the east field in 2019. (a)-(c) NDVI of July, August and September in 2019. The legend in (c) is also used in (a) and (b). (d)-(e) show the NDVI changes from July to August and August to September in 2019. (f) ECa-based sand % map of the east field
![alt text](https://github.com/AJFeng/Cotton_growth_variation/blob/main/figures/2019NDVI.png)

NDVI maps collected on west field in 2018. (a)-(d) NDVI of June, July, August and Sep in 2018. (e)-(g) NDVI changes from June to July, July to August and August to September in 2018. (h) ECa-based sand% map of the west field
![alt text](https://github.com/AJFeng/Cotton_growth_variation/blob/main/figures/2018NDVI.png)

Heat map for the feature contribution (%) of soil features and Ks features to estimates of each image feature and yield. I_ Ks: the Ks on the imaging date; T_1: total days having Ks <1; T_0.9_1: total days having Ks in [0.9,1]; L_1: the largest number of continuous days having Ks <1; B_1: number of days after planting to first instance of Ks <1. Yield_2019_08_2019: the Ks features for the 2019 yield estimation were calculated from the date of planting to the imaging date in August. The pink and blue boxes mark the relationships described in the text
![alt text](https://github.com/AJFeng/Cotton_growth_variation/blob/main/figures/feature_contribution.png)

## Conclusion
Results showed that differences in NDVI were found in both 2018 and 2019 under varying ECa-based soil texture. Soil features and Ks features did not model crop growth variation well in the early growth stages when crops do not require a large amount of water and soil water storage is sufficient.  Soil features and Ks features had a higher correlation with image features in the middle growth stages. Soil features had a stronger correlation with crop development when crops suffered from water stress. Clay content in shallow layers affected crop development in early growth stages, while clay content in the deeper layers affected the middle growth stages. The Ks features were important indicators of crop growth variation if irrigations were applied. The results showed that the combination of soil and weather data with UAV image data makes it feasible to examine the effects of soil and weather on crop growth variation.
