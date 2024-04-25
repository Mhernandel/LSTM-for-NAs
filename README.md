# LSTM-for-NAs
Use Long Short-Term Memory Networks to find missing values in your data

## What makes this method useful:

1.	The opportunity to deal with high dimensional multivariate time series forecasting without high computational cost as other methods like Autoregressive integrated moving average (ARIMA) will require. 
2.	The chance to capture complex non-linear relationships by controlling the errors prediction were methods like LASSO usually fail. 
3.	It captures complex dynamical phenomena like the combination of using Bayesian inference with Gaussian Process for multivariate time-series without too much previous calculation (large covariance matrices). 
4.	The method can be used to predict short-term and long-term data.
5.	Because of the memory storage that the method provides, this method, allows to predict missing values based on deep historical context, which traditional methods might overlook. 
6.	The method has an inherent capacity to generalize across noisy data, learning to ignore outliers and focus on underlying patterns in the data without specification or control validation like other methods.
   
## Considerations to use the method:

1.	It should be a Missing Completely at Random (MCAR) category of missing data; All the variables and observations have the same probability of being missing and other explanatory variables are not taking in consideration.
2.	It should be a time-series type of data.
3.	It’s particularly useful for datasets where past values influence future values.

## How this method works:

This method uses loops to inform later events about previous events assigning likelihood to possible outcomes, similar to how memory works in our brains. LSTM is an especial kind of network that can be used to fill the gaps in data using it’s time frame context, this method is explicitly designed to avoid the long-term dependency problem (avoid long-term bias). “The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.” (Oláh, 2015)

<img width="330" alt="image" src="https://github.com/Mhernandel/LSTM-for-NAs/assets/71413078/5991a687-44eb-4587-ada1-aba87581ce7a">

### References:
•	Lai, G., Chang, W.-C., Yang, Y., & Liu, H. (2017). Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks.https://arxiv.org/pdf/1703.07015.pdf
•	Oláh, C. (2015, August 27). Understanding LSTM Networks. Colah's Blog. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
•	Dinesh Y. (2019, December 5). Beginner’s Guide to RNN & LSTMs. https://medium.com/@humble_bee/rnn-recurrent-neural-networks-lstm-842ba7205bbf
•	Gregor, K., Danihelka, I., Graves, A., Rezende, D. J., & Wierstra, D. (2015). DRAW: A recurrent neural network for image generation. arXiv. https://arxiv.org/pdf/1503.04069.pdf







