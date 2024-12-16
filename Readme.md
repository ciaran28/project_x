Notebook Depression

Technical
1. Ingest Data and python libraries
2. We need to contextualise the data with visuals.
- Age is uniformly distributed (do we loose generational cohort effects?)
- From a cursory look at data from US and UK, married couples seem to be over represented in the data at 58%. Therefore the sample set may not be representative of the population. 


3. Describe the data with summary statists, and understand the data types (we need to understand the effort required in wrangling the data)
- lobject data types. will need to be hot encoded prior to training. 
- The data is already well currated. there are no duplicates etc. the data wrangling is therefore minimal. 

4. feature engineering 
- Comment on the generational affect (remove this - )


5. Machine Learning 
- Remove the name column
- Split the dataset between training and test
- Initialise XGBoost model. (Talk about reasons behind the model selection) (consider possible improvements - end )

problem is non liner - hence gradient boosting, forests

need to build out bigger evaluation set . this is a complex process. how do even define it. evaluation framework must be in place before we spend time on propmt engineering/ llm model selection. we need to define good.

