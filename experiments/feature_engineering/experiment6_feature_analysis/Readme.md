### Feature Analysis

**Realizations:**  
- Linear and non-linear correlation should be measured by the features 
to get full insights about the dependency of the features
- Features need to be measured with the shifted stock price, because it should be measured
how strong the correlation of the features is with the future stock price (shift by one step)
- Features that strongly correlate with each other should be analyzed further 
in order to decide whether to keep them or not. 
-> If all of those correlating features are used it is inefficient
as the information is already in a single feature of those. The model might also falsely focus on this single information
because it is more apparent in the input layer than the information of the other features.
- Features might correlate differently with each other for different companies, some features might be more relevant 
for specific companies