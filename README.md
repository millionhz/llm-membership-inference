# Membership Inference on Proprietary Large Language Models

Large Language Models (LLMs) have become ubiquitous, offering impressive capabilities in generating text for applications ranging from chatbots to creative content. However, training these models often involves vast datasets that may include both publicly available information and private or proprietary data. This practice raises serious concerns about copyright infringement and privacy violations. In many cases, individuals or organizations may need to verify whether their specific data was included in a model’s training set.

This project aims to investigate whether we can use model outputs as a signal to identify whether a certain document was part of the model's training dataset or not. We propose several techniques to infer whether documents were part of the training dataset and evaluate the effectiveness of these techniques using ROC curves and binary classification metrics.

See the [Releases](https://github.com/millionhz/llm-membership-inference/releases) page for our report.