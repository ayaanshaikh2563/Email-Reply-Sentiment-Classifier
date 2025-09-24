# Part-A answer

## Question: Compare results and explain which model you’d use in production and why?

After training and evaluating the models, DistilBERT achieved the highest F1 score and better handled the nuances in email replies compared to Logistic Regression and LightGBM. While Logistic Regression and LightGBM are fast and lightweight, they struggle with context and complex language patterns in short text. Therefore, I would choose DistilBERT for production because it provides more accurate, context-aware predictions, which is crucial for correctly classifying prospect replies, even though it requires slightly more resources.

-----

# Short Answer Reasoning

## Question 1: If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

I would use data augmentation like synonym replacement, paraphrasing, or back-translation to expand the dataset. I’d also leverage pre-trained models like DistilBERT for transfer learning, which performs well even with small data. Additionally, few-shot learning or generating synthetic examples using GPT could boost training without needing thousands of new labels.

## Question 2: How would you ensure your reply classifier doesn't produce biased or unsafe outputs in production?

I’d test the model across different groups and styles to detect bias, set confidence thresholds to flag uncertain predictions for human review, and monitor outputs continuously. Periodic audits and fairness-aware retraining would help keep the model safe and unbiased over time.

## Question 3: Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

I’d provide the LLM with detailed context about the prospect and use few-shot prompting with quality examples to guide tone and style. Structured prompts with clear instructions and prompt chaining would help avoid generic outputs and make the openers more personalized and relevant.


