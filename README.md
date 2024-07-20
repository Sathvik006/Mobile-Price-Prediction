# Mobile Price Prediction

## Project Description

### Problem Statement
Create a classification model to predict the price range of a mobile based on certain specifications.

### Context
An entrepreneur has started his own mobile company and wants to compete with big companies like Apple and Samsung. To price his mobiles competitively, he needs to find relationships between mobile features (e.g., RAM, Internal Memory, etc.) and their selling price.

### Dataset
You can download the dataset [here](https://drive.google.com/file/d/1pjDDlI4kJ75GLOOj_HkFqqeV8Of_c6H1/view?usp=share_link).

### Features
The dataset contains 21 features and 2000 entries. The features are as follows:
1. `battery_power`: Total energy a battery can store in one time measured in mAh
2. `blue`: Has Bluetooth or not
3. `clock_speed`: Speed at which microprocessor executes instructions
4. `dual_sim`: Has dual SIM support or not
5. `fc`: Front Camera megapixels
6. `four_g`: Has 4G or not
7. `int_memory`: Internal Memory in Gigabytes
8. `m_dep`: Mobile Depth in cm
9. `mobile_wt`: Weight of mobile phone
10. `n_cores`: Number of cores of the processor
11. `pc`: Primary Camera megapixels
12. `px_height`: Pixel Resolution Height
13. `px_width`: Pixel Resolution Width
14. `ram`: Random Access Memory in MegaBytes
15. `sc_h`: Screen Height of mobile in cm
16. `sc_w`: Screen Width of mobile in cm
17. `talk_time`: Longest time that a single battery charge will last
18. `three_g`: Has 3G or not
19. `touch_screen`: Has touch screen or not
20. `wifi`: Has WiFi or not
21. `price_range`: Target variable with values of 0 (low cost), 1 (medium cost), 2 (high cost), and 3 (very high cost).

## Steps
1. Handle null values (if any).
2. Split data into training and test sets.
3. Apply the following models on the training dataset and generate predictions for the test dataset:
   - Logistic Regression
   - KNN Classification
   - SVM Classifier (linear and RBF kernel)
   - Decision Tree Classifier
   - Random Forest Classifier
4. Predict the price range for test data.
5. Compute the Confusion matrix and classification report for each model.
6. Report the model with the best accuracy.


