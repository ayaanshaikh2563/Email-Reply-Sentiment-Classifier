# Short Answer Reasoning

## Question 1: If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

With only 200 labeled samples, I would apply data augmentation techniques like synonym replacement, back-translation, and paraphrasing to expand the dataset artificially. Additionally, I would leverage pre-trained models like DistilBERT for transfer learning, requiring minimal fine-tuning. Few-shot learning or synthetic data generation using GPT-based models could also help enhance the training data.

## Question 2: How would you ensure your reply classifier doesn't produce biased or unsafe outputs in production?

I would implement bias detection via thorough testing across demographics and communication styles and set confidence thresholds to flag uncertain predictions for human review. Ongoing monitoring, fairness-aware training, and periodic audits would ensure the model remains fair and safe.

## Question 3: Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

I would use prompts rich in prospect context, providing detailed information and industry specifics. Few-shot prompting with high-quality, varied examples ensures tone and relevance. Structured prompts with clear instructions to avoid generic phrases, combined with prompt chaining techniques, would guide the LLM to deliver tailored, effective openers.
