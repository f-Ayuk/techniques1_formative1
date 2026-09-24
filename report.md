# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## Abstract

This study investigates one-step-ahead mobile Internet traffic forecasting using activity data collected across Milan at 10-minute intervals. The study compares three substantially different sequential forecasting approaches: Seasonal Autoregressive Integrated Moving Average (SARIMA), Long Short-Term Memory (LSTM), and one-dimensional Convolutional Neural Network (1D CNN).

## 1. Introduction

Mobile network traffic forecasting supports proactive resource allocation,
energy saving, and quality-of-service management. Operators need reliable
short-term predictions of traffic demand to decide when to activate
additional capacity, schedule maintenance, or trigger beam-level
reconfiguration. This study investigates one-step-ahead forecasting of
Internet traffic across the Milan metropolitan area, using the Telecom
Italia Big Data Challenge dataset published by Barlacchi et al. (2015).

**Research question.** How do different sequential models compare for
one-step-ahead mobile network traffic forecasting, and how does their
performance vary across geographical areas with different traffic
characteristics?

We compare three models representing three distinct paradigms — classical
statistical (SARIMA), deep recurrent (LSTM), and gradient-boosted trees
(XGBoost) — and evaluate them on the highest-traffic areas identified in
the exploratory analysis.

### 2.1 Dataset Description

The dataset used in this study contains telecommunications activity collected across the city of Milan over approximately two months. Milan is divided into 10,000 geographical areas, represented by unique square IDs, and observations are recorded at 10-minute intervals. Each record contains activity information associated with a geographical square and time interval, including SMS, call, and Internet traffic measurements.

For this study, the analysis is restricted to Internet traffic. The Internet traffic variable represents the amount of Internet activity recorded within a geographical square during a given 10-minute interval. Since the forecasting task concerns the total Internet traffic generated within each geographical area, records associated with different country codes are aggregated for the same square and time interval.

The resulting forecasting dataset can therefore be represented as a temporal sequence for each geographical area:

𝑥𝑎,𝑡 where 𝑎 denotes the geographical square and 𝑡 denotes a 10-minute time interval. The forecasting objective is to use the historical sequence of observations for an area to predict its Internet traffic at the next time interval, 𝑥𝑎,𝑡+1.

The original dataset is distributed across multiple daily text files. Consequently, the dataset is sufficiently large that loading all raw files simultaneously is unnecessary and may result in substantial memory consumption. This makes memory-efficient data processing an important consideration before conducting the exploratory analysis and forecasting experiments.


The analysis focuses specifically on Internet traffic.

### 2.2 Data Processing Strategy
Because the raw dataset contains many daily files and a large number of observations, loading all raw files simultaneously is unnecessary and can lead to excessive memory consumption.

The adopted strategy processes the files incrementally:

identify the directory containing the raw .txt files;

process one file at a time;

read only the columns required for Internet traffic forecasting;

use reduced numerical data types where appropriate;

aggregate Internet traffic across country codes for each geographical square and time interval;

release intermediate objects from memory;

combine the compact results;

save the resulting dataset in Parquet format for subsequent analysis.

This approach avoids retaining the full raw dataset in memory and also avoids repeatedly parsing the original text files during later stages of the experiment.

1. Introduction
This should establish the research problem before getting into the technical work.

1.1 Background
Explain:

Growth and variability of mobile network traffic.

Why traffic forecasting matters.

Why short-term/one-step-ahead forecasting is useful.

Why geographical traffic patterns may differ.

1.2 Problem Statement
Explain that the dataset contains mobile Internet activity across Milan at 10-minute intervals and that traffic is likely to contain temporal dependencies, periodic behaviour, and geographical heterogeneity.

1.3 Research Question
Use the assignment's question directly:

How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

1.4 Objectives
Your objectives can be:

Develop an efficient strategy for processing the large Milan traffic dataset.

Characterise the temporal and geographical properties of Internet traffic.

Review existing forecasting approaches.

Compare SARIMA, LSTM and CNN.

Evaluate one-step-ahead forecasting performance.

Investigate differences between geographical areas.

Compare predictive accuracy and computational cost.

Analyse cases where models produce large forecasting errors.

1.5 Contributions
Briefly state what your experiment contributes.

For example:

This study provides an empirical comparison of a classical statistical model, a recurrent neural model and a convolutional neural model under the same one-step-ahead forecasting setting, with particular attention to differences across high-traffic geographical areas and computational requirements.

1.6 Report Structure
One short paragraph explaining what each subsequent section contains.

2. Data Handling and Memory Management
This is the assignment's Task 1.

2.1 Dataset Description
Describe:

Milan dataset

geographical squares

10-minute intervals

observation period

Internet traffic variable

raw file organisation

2.2 Initial Data Inspection
Show:

file count

sample records

columns

data types

number of observations


### 2.3 Memory Optimisation
Memory usage was measured before and after the optimisation.

[INSERT FIGURE/TABLE: Memory usage before and after optimisation]

Figure/Table X. Memory consumption before and after the data-processing optimisation.

The following values were obtained during execution:

Measure	Result
Number of raw files	[INSERT RESULT]
Original memory usage	[INSERT RESULT] MB
Optimised memory usage	[INSERT RESULT] MB
Memory reduction	[INSERT RESULT] %
Raw processing time	[INSERT RESULT] seconds
Processed dataset size	[INSERT RESULT] MB

The optimisation reduces memory requirements by avoiding unnecessary columns, using lower-precision numerical representations where appropriate, and processing the raw files incrementally.

An important limitation is that process-level memory usage and the memory reported by a pandas DataFrame are not identical measures. Consequently, the experiment distinguishes between DataFrame memory consumption and overall process memory where appropriate.

The conversion from text files to a compact columnar representation also introduces an additional preprocessing step, but this cost is incurred once and reduces the amount of work required for subsequent analysis.

## 3. Exploratory Data Analysis
### 3.1 Distribution of Total Internet Traffic
For each geographical square, the total Internet traffic over the complete observation period was calculated.

[INSERT FIGURE: Distribution of total Internet traffic across geographical areas]

Figure X. Distribution of total Internet traffic across the geographical areas.

The resulting distribution has the following characteristics:

Number of geographical areas: [INSERT RESULT]

Minimum total traffic: [INSERT RESULT]

Median total traffic: [INSERT RESULT]

Mean total traffic: [INSERT RESULT]

Maximum total traffic: [INSERT RESULT]

The distribution should be interpreted in terms of the degree of heterogeneity between geographical areas. In particular, the comparison between the mean and median, the histogram, and the presence of extreme observations indicate whether traffic is concentrated in a relatively small number of locations or distributed more uniformly.

[WRITE FINAL DATA-BASED INTERPRETATION HERE AFTER KAGGLE EXECUTION.]

This geographical heterogeneity is relevant to forecasting because models may encounter substantially different signal characteristics depending on the area being considered.

### 3.2 Highest-Traffic Geographical Areas
The three geographical areas with the highest total Internet traffic over the complete observation period were identified automatically.

Rank	Square ID	Total Internet Traffic
1	[INSERT RESULT]	[INSERT RESULT]
2	[INSERT RESULT]	[INSERT RESULT]
3	[INSERT RESULT]	[INSERT RESULT]

The three highest-traffic areas are subsequently used as the principal geographical areas for the forecasting experiments.

The analysis additionally includes Square IDs 4159 and 4556, as required by the assignment.

### 3.3 Traffic During the First Two Weeks
The Internet traffic time series during the first two weeks was examined for:

the three highest-traffic areas;

Square ID 4159;

Square ID 4556.

Highest-Traffic Area
[INSERT FIGURE: First-two-week time series for highest-traffic area]

Figure X. Internet traffic during the first two weeks for Square [TOP_AREA_1].

Second-Highest-Traffic Area
[INSERT FIGURE: First-two-week time series for second-highest area]

Figure X. Internet traffic during the first two weeks for Square [TOP_AREA_2].

Third-Highest-Traffic Area
[INSERT FIGURE: First-two-week time series for third-highest area]

Figure X. Internet traffic during the first two weeks for Square [TOP_AREA_3].

Square 4159
[INSERT FIGURE: First-two-week time series for Square 4159]

Figure X. Internet traffic during the first two weeks for Square 4159.

Square 4556
[INSERT FIGURE: First-two-week time series for Square 4556]

Figure X. Internet traffic during the first two weeks for Square 4556.

Temporal Comparison
The five areas were compared in terms of:

daily periodicity;

weekday/weekend differences;

peak traffic periods;

low-traffic periods;

amplitude of fluctuations;

volatility;

unusual spikes or drops.

[WRITE DATA-BASED COMPARISON HERE AFTER VIEWING THE FIVE FIGURES.]

The comparison is important because the forecasting models do not operate on identical statistical signals. Areas with stronger and more regular periodicity may be easier to forecast than areas characterised by irregular or abrupt changes.

## 4. Temporal Characteristics of the Highest-Traffic Area
Two principal analyses were conducted for the geographical area with the highest total Internet traffic.

### 4.1 Autocorrelation Analysis
Autocorrelation was examined using the ACF and PACF.

Because observations occur every 10 minutes:

lag 1 represents 10 minutes;

lag 6 represents one hour;

lag 144 represents one day;

lag 1008 represents one week.

[INSERT FIGURE: ACF and PACF]

Figure X. ACF and PACF of the highest-traffic geographical area.

The analysis was used to identify temporal dependencies that may be useful for forecasting and to inform the seasonal structure of the SARIMA model and the sequence length of the neural models.

[WRITE FINAL INTERPRETATION OF ACF/PACF HERE.]

Particular attention was given to whether significant autocorrelation occurs around daily and weekly lags.

### 4.2 Daily and Weekly Seasonality
The second analysis examined average traffic according to the time of day and day of week.

[INSERT FIGURE: Average traffic by hour of day]

Figure X. Average Internet traffic by hour of day for the highest-traffic area.

[INSERT FIGURE: Average traffic by day of week]

Figure X. Average Internet traffic by day of week for the highest-traffic area.

The hourly profile provides evidence regarding intraday seasonality, while the day-of-week profile provides evidence of differences between weekdays and weekends.

[WRITE FINAL INTERPRETATION HERE.]

These results inform the forecasting methodology because strong recurring daily patterns can be represented explicitly by SARIMA and can also be learned implicitly by the neural models when sufficient historical observations are supplied.

### 4.3 Stationarity Check
An Augmented Dickey-Fuller test was also conducted to provide supporting evidence for the treatment of stationarity in the statistical model.

Statistic	Result
ADF statistic	[INSERT RESULT]
p-value	[INSERT RESULT]
1% critical value	[INSERT RESULT]
5% critical value	[INSERT RESULT]
10% critical value	[INSERT RESULT]

The ADF result was interpreted together with the visual characteristics of the time series rather than being used as the sole basis for determining stationarity.

[WRITE FINAL INTERPRETATION HERE.]

## 5. Related Work
### 5.1 Overview
Network traffic forecasting has been studied using statistical time-series models, machine-learning techniques, and deep-learning architectures.

Ferreira et al. provide a broad survey and tutorial covering ARMA, ARIMA, SARIMA, recurrent neural networks, LSTM, GRU, and CNN approaches to network traffic forecasting. Their study also compares methods using both predictive quality and computational cost, making it particularly relevant to the experimental design adopted here.

Cao and Liu investigate LSTM-based traffic forecasting specifically for cellular networks. They motivate LSTM by the temporal correlations and nonlinear characteristics of cellular traffic and train the network using backpropagation through time. Their experiments evaluate forecasting using MSE and MAE.

Azari et al. compare LSTM and ARIMA for cellular traffic prediction using real network traffic data. Their results indicate that LSTM can perform strongly when sufficiently large and fine-grained training data are available, while ARIMA can remain competitive in some settings with lower complexity.

Escriche, Vassaki and Peters provide a broader comparative study of cellular traffic prediction mechanisms. Their work evaluates statistical, machine-learning and deep-learning methods on the same dataset and demonstrates that model performance can depend on the forecasting horizon. This is particularly relevant to the current study because the present experiment focuses specifically on one-step-ahead prediction.

More recent work by Hussien, Nashaat and Abdel-Kader evaluates multiple approaches to 5G network traffic prediction, including SARIMA, LSTM and CNN, and reports both predictive performance and computational time. This provides additional motivation for comparing these three model families within a common experimental framework.

These studies support the use of multiple modelling paradigms rather than assuming that one model family will universally outperform the others.

## 6. Model Selection
Three models were selected:

SARIMA;

LSTM;

1D CNN.

The models were selected because they provide substantially different approaches to sequential forecasting.

Model	Model family	Main characteristic investigated
SARIMA	Statistical	Autoregressive and seasonal dependencies
LSTM	Recurrent neural network	Sequential and nonlinear dependencies
1D CNN	Convolutional neural network	Local temporal patterns

### 6.1 SARIMA
SARIMA extends ARIMA by explicitly modelling seasonal behaviour.

Its general form is:

S
A
R
I
M
A
(
p
,
d
,
q
)
(
P
,
D
,
Q
,
s
)

where 
p
,
d
,
q
 represent the non-seasonal autoregressive, differencing and moving-average components, while 
P
,
D
,
Q
,
s
 represent their seasonal counterparts and the seasonal period.

SARIMA was selected because the exploratory analysis provides evidence about recurring temporal patterns and autocorrelation. It also provides an interpretable statistical benchmark against which the neural models can be compared.

A further advantage is relatively low model complexity compared with deep neural networks. A limitation is that the model depends on assumptions about the underlying time-series structure and may be less flexible when traffic behaviour is strongly nonlinear or contains abrupt irregular changes.

### 6.2 LSTM
LSTM is a recurrent neural-network architecture designed to model sequential information using gated memory mechanisms.

LSTM was selected because mobile network traffic contains temporal dependencies and potentially nonlinear relationships. Previous cellular-network research has specifically applied LSTM to traffic forecasting, and comparisons with ARIMA have shown that the relative performance of the two approaches can depend on data granularity and training-set size.

The main advantage of LSTM in this experiment is its ability to learn temporal relationships directly from sequences.

Its disadvantages include:

greater computational cost;

more hyperparameters;

longer training time;

reduced interpretability compared with SARIMA;

sensitivity to preprocessing and architecture choices.

### 6.3 1D CNN
The third model is a one-dimensional convolutional neural network applied along the temporal dimension.

The CNN receives a sequence of historical traffic observations and applies temporal convolution filters to identify local patterns.

CNN was selected because traffic signals can contain short-term patterns such as local increases, decreases, peaks and transitions. CNNs also provide a substantially different neural architecture from recurrent networks.

Compared with LSTM, a temporal CNN can process sequence elements in parallel during training and may therefore provide computational advantages. However, its ability to represent long-range dependencies depends on the receptive field created by the network architecture.

## 7. Forecasting Methodology
### 7.1 Forecasting Formulation
Let 
x
t
 represent Internet traffic in a geographical area at time 
t
.

The objective is to estimate:

x
^
t
+
1
=
f
(
x
t
−
L
+
1
,
.
.
.
,
x
t
)

where 
L
 is the historical sequence length.

Because observations are recorded every 10 minutes, predicting 
x
t
+
1
 corresponds to predicting traffic during the next 10-minute interval.

### 7.2 Input Representation
The neural models use a fixed historical window.

The selected sequence length was:

L
=
[
I
N
S
E
R
T
 
F
I
N
A
L
 
S
E
Q
U
E
N
C
E
 
L
E
N
G
T
H
]

If 
L
=
144
, the model receives the previous 24 hours of traffic.

The final sequence length was selected based on the temporal characteristics observed during exploratory analysis and practical computational considerations.

### 7.3 Preprocessing
The following preprocessing procedure was applied:

raw text files were processed and aggregated by geographical square and time interval;

Internet traffic values were checked for missing values;

observations were ordered chronologically;

training, validation and test periods were separated chronologically;

neural-network inputs were scaled using a scaler fitted only on the training data;

input-output sequences were generated using a sliding window;

predictions were transformed back to the original traffic scale before evaluation.

Randomly shuffling observations between training and test periods was avoided to prevent temporal leakage.

## 8. Train, Validation and Test Strategy
The experiment uses a chronological split.

The training data precede the validation data, and the validation period precedes the final test period.

The final evaluation period is:

16 December to 22 December

This period is treated as unseen future data for the final evaluation.

[INSERT FIGURE: TRAIN/VALIDATION/TEST TIMELINE]

Figure X. Chronological division of the dataset into training, validation and test periods.

## 9. Model Architectures
### 9.1 SARIMA
The SARIMA model was fitted using the statsmodels implementation.

The selected model configuration was:

SARIMA([INSERT p,d,q])([INSERT P,D,Q],[INSERT s])

The parameters were determined using the exploratory analysis and model-selection procedure described in the notebook.

### 9.2 LSTM
The LSTM architecture consisted of:

Input sequence
     ↓
LSTM layer
     ↓
Dropout
     ↓
Dense layer
     ↓
Output neuron

The final architecture was:

LSTM units: [INSERT]
Dropout: [INSERT]
Dense units: [INSERT]
Batch size: [INSERT]
Maximum epochs: [INSERT]
Optimizer: Adam
Loss: Mean Squared Error

Early stopping based on validation loss was used to reduce unnecessary training.

### 9.3 CNN
The temporal CNN architecture consisted of:

Input sequence
     ↓
Conv1D
     ↓
Activation
     ↓
Conv1D
     ↓
Activation
     ↓
Pooling
     ↓
Dense
     ↓
Output

The final configuration was:

Filters: [INSERT]
Kernel sizes: [INSERT]
Pooling: [INSERT]
Dense units: [INSERT]
Batch size: [INSERT]
Maximum epochs: [INSERT]
Optimizer: Adam
Loss: Mean Squared Error

## 10. Experimental Setup
The models were evaluated on the three geographical areas with the highest total Internet traffic.

Experimental Area	Square ID
Area 1	[INSERT TOP AREA]
Area 2	[INSERT SECOND AREA]
Area 3	[INSERT THIRD AREA]

The evaluation period was 16–22 December.

The experiments were performed using:

Specification	Value
Platform	[Kaggle/local]
CPU	[INSERT]
GPU	[INSERT]
RAM	[INSERT]
Python	[INSERT]
TensorFlow	[INSERT]
Statsmodels	[INSERT]

## 11. Evaluation Metrics
Three primary metrics were used.

###11.1 Mean Absolute Error
M
A
E
=
1
n
∑
i
=
1
n
∣
y
i
−
y
^
i
∣

MAE represents the average absolute forecasting error.

###11.2 Root Mean Squared Error
R
M
S
E
=
1
n
∑
i
=
1
n
(
y
i
−
y
^
i
)
2

RMSE gives greater influence to large errors and is therefore useful for identifying models that produce particularly large deviations during traffic peaks.

### 11.3 Mean Absolute Percentage Error
M
A
P
E
=
100
n
∑
i
=
1
n
∣
y
i
−
y
^
i
y
i
∣

Because traffic can be zero or close to zero, a small denominator safeguard was used in the implementation.

MAPE should therefore be interpreted alongside MAE and RMSE rather than in isolation.

## 12. Experimental Results
### 12.1 Area 1
[INSERT FIGURE: SARIMA actual vs predicted — Area 1]

Figure X. SARIMA one-step-ahead predictions for Area 1 during 16–22 December.

[INSERT FIGURE: LSTM actual vs predicted — Area 1]

Figure X. LSTM one-step-ahead predictions for Area 1 during 16–22 December.

[INSERT FIGURE: CNN actual vs predicted — Area 1]

Figure X. CNN one-step-ahead predictions for Area 1 during 16–22 December.

Performance
Model	MAE	MAPE (%)	RMSE
SARIMA	[INSERT]	[INSERT]	[INSERT]
LSTM	[INSERT]	[INSERT]	[INSERT]
CNN	[INSERT]	[INSERT]	[INSERT]

[WRITE DATA-BASED INTERPRETATION HERE.]

## 13. Area 2 Results
[INSERT FIGURE: SARIMA actual vs predicted — Area 2]

[INSERT FIGURE: LSTM actual vs predicted — Area 2]

[INSERT FIGURE: CNN actual vs predicted — Area 2]

Model	MAE	MAPE (%)	RMSE
SARIMA	[INSERT]	[INSERT]	[INSERT]
LSTM	[INSERT]	[INSERT]	[INSERT]
CNN	[INSERT]	[INSERT]	[INSERT]

[WRITE DATA-BASED INTERPRETATION HERE.]

## 14. Area 3 Results
[INSERT FIGURE: SARIMA actual vs predicted — Area 3]

[INSERT FIGURE: LSTM actual vs predicted — Area 3]

[INSERT FIGURE: CNN actual vs predicted — Area 3]

Model	MAE	MAPE (%)	RMSE
SARIMA	[INSERT]	[INSERT]	[INSERT]
LSTM	[INSERT]	[INSERT]	[INSERT]
CNN	[INSERT]	[INSERT]	[INSERT]

[WRITE DATA-BASED INTERPRETATION HERE.]

## 15. Computational Performance
Training and prediction times were measured using a monotonic performance timer.

Training time includes model fitting but excludes package installation and environment startup.

Prediction time measures the time required to generate forecasts for the evaluation period.

Model	Training Time (s)	Prediction Time (s)
SARIMA	[INSERT]	[INSERT]
LSTM	[INSERT]	[INSERT]
CNN	[INSERT]	[INSERT]

[WRITE COMPARATIVE INTERPRETATION HERE.]

The computational results should be interpreted together with predictive performance. A model with slightly lower error but substantially higher computational cost may have different practical implications from a model with similar accuracy and much lower computational requirements.

## 16. Comparative Analysis
The results should be considered across three dimensions:

predictive accuracy;

computational cost;

robustness across geographical areas.

### 16.1 Predictive Accuracy
The complete results are summarised below.

Area	Model	MAE	MAPE (%)	RMSE
[Area 1]	SARIMA	[ ]	[ ]	[ ]
[Area 1]	LSTM	[ ]	[ ]	[ ]
[Area 1]	CNN	[ ]	[ ]	[ ]
[Area 2]	SARIMA	[ ]	[ ]	[ ]
[Area 2]	LSTM	[ ]	[ ]	[ ]
[Area 2]	CNN	[ ]	[ ]	[ ]
[Area 3]	SARIMA	[ ]	[ ]	[ ]
[Area 3]	LSTM	[ ]	[ ]	[ ]
[Area 3]	CNN	[ ]	[ ]	[ ]

[INSERT FIGURE: Model comparison across the three areas]

Figure X. Comparison of forecasting errors across the three geographical areas.

The results demonstrate whether model performance is consistent across areas or whether it depends on the characteristics of the traffic series.

[WRITE FINAL DATA-BASED COMPARISON HERE.]

### 16.2 Relationship Between Traffic Characteristics and Model Performance
The EDA identified differences in:

total traffic volume;

temporal variability;

daily seasonality;

weekly behaviour;

autocorrelation;

peak intensity.

These characteristics provide a basis for interpreting differences in forecasting performance.

For example, a model that performs well on a highly regular series may not necessarily perform equally well on an area with irregular spikes.

[INSERT DATA-BASED DISCUSSION LINKING EDA TO MODEL RESULTS.]

## 17. Analysis of Poor Predictions
At least one period with particularly poor prediction was identified automatically using the forecasting errors.

[INSERT FIGURE: Worst prediction period]

Figure X. Example of a period where forecasting error is particularly large.

The selected period was:

[INSERT DATE/TIME RANGE]

The affected model(s) were:

[INSERT MODEL]

The observed behaviour was:

[DESCRIBE SPIKE/DROP/ANOMALY/TRANSITION]

A possible explanation is that the traffic pattern during this period differs from the historical patterns available to the model.

For SARIMA, unusually large deviations can be difficult to model when the observed behaviour is not adequately represented by the fitted seasonal and autoregressive structure.

For LSTM, prediction errors may occur when an abrupt change differs from the patterns learned from historical sequences.

For CNN, local convolutional filters may capture short-term patterns effectively while having limitations when the relevant dependency extends beyond the receptive field.

These interpretations should be supported by the actual prediction plot rather than assumed in advance.

## 18. Model-Specific Discussion
### 18.1 SARIMA
SARIMA provides an interpretable statistical baseline and explicitly represents autoregressive and seasonal behaviour.

Its main strengths are:

relatively transparent model structure;

direct representation of temporal dependence;

comparatively low computational complexity;

strong suitability for regular seasonal patterns.

Its limitations include:

dependence on model specification;

difficulty representing complex nonlinear relationships;

sensitivity to abrupt structural changes;

potential computational cost when repeatedly fitting complex models to many series.

The experimental results show:

[INSERT SARIMA-SPECIFIC OBSERVATIONS.]

### 18.2 LSTM
LSTM provides a flexible nonlinear sequential model.

Its strengths include:

ability to learn nonlinear temporal relationships;

ability to use historical sequences directly;

suitability for complex sequential patterns.

Its limitations include:

greater training cost;

more hyperparameters;

sensitivity to sequence length and scaling;

lower interpretability than SARIMA.

The experimental results show:

[INSERT LSTM-SPECIFIC OBSERVATIONS.]

### 18.3 CNN
The temporal CNN provides a different neural approach by applying convolution over historical sequences.

Its strengths include:

efficient extraction of local temporal patterns;

parallel computation;

potentially lower training cost than recurrent architectures;

flexible nonlinear modelling.

Its limitations include:

dependence on kernel size and receptive field;

potentially weaker representation of very long-range dependencies unless the architecture is sufficiently deep or dilated;

less direct interpretability than SARIMA.

The experimental results show:

[INSERT CNN-SPECIFIC OBSERVATIONS.]

## 19. Overall Findings
The experiment provides evidence about three distinct forecasting approaches.

The principal findings are:

Traffic heterogeneity:
[INSERT FINDING]

Temporal structure:
[INSERT FINDING]

SARIMA performance:
[INSERT FINDING]

LSTM performance:
[INSERT FINDING]

CNN performance:
[INSERT FINDING]

Cross-area variation:
[INSERT FINDING]