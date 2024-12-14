# Approach

- Reusability of code.
    - Poetry packaging 
- Data Quality Considerations 
    - Check for duplication of data. Remove


- Sentiment analsis
    - data set is small (fine tuning wouldn't work)
    - out of the box fine tuned model from hugging face should do well 
    - bert-base-uncased or roberta-base --> Provide Neutral / Positive / Negative 
    - Throttling at 512 tokens. used a bigger model, but noticed a collapse in performance. Use small model and choose to chunk it. Explored summarizing. ran out of time before being able to evaluate the approaches. concerned that context would be lost with summarizing . 
    - cleansed the data. 


- Technical 
    - Batch - reduce costs. 
    - semantic kernel - token effeciency
    - need to deal with speeds (a lot of for loops - could be removed )
    - migrate to c# 
    - PII data cleansing (gpt)
    - There is a lot of subjectivity inherent in what is meant by sentiment of the call. what are we trying to capture, whether the customer is experiencing negative by the the actual conversation, i.e actions as per the representative.. or is it negative emotion coming into the call.. or both. This will pick up both. Therefore it should not be read as providing negative indicators on the customer service. it may well point to it. It shows that the customer is not completely happy. This may skew the results. For example transcript 100 indicates a negative outcome.. however on a qualitative assesment it could be viewed as a positive outcome as the issue is resolved. The negativitiy stems from a break down in it process (updating the policy). I believe this is still valuable for the business (qualitiative assessment )... it does show that the ground truth / evaluation techniques may need ironing out first. 
    5. Qualitative Analysis
    Manually inspect cases where the model performs poorly:
    Look at samples where the ground truth label and model prediction differ.
    Identify patterns (e.g., specific phrases, sarcasm, or ambiguous sentiment) that cause errors.
    Use this analysis to improve the dataset or fine-tune the model.
    May need to retrain. 
    - Human centric approach suggests we need to have the customer self identify after a conversation to build the evaluation sets. 


    Actions: (llmops) Evaluation should be using mlflow evaluation. We may need to improve the prompts. Can use prompt flow variants. 

- Business
    - What's the use case
    - improvmenent in customer satifaciton . Can we find data that shows how quality outcomes translates to profits
    - reduction in fines (FCA -treating customers fairly )
    - This doesn't need to be a silver bullet. Even identifying 10% of the cases that could slip through the net could cover the cost of the system 
    - Cost of system needs to be improved. 
    - 