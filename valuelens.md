👤 Eduardo de la Cruz Fernández,*,1, [ORCID](https://orcid.org/0009-0009-2691-4330) 👤 Marcelo Karanik,2, [ORCID](https://orcid.org/0000-0001-8848-3681) and 👤 Sascha Ossowski,3, [ORCID](https://orcid.org/0000-0003-2483-9508)  
🏛️ CETINIA, Universidad Rey Juan Carlos, Madrid, Spain  

*∗Corresponding Author. Email: eduardo.cruz@urjc.es*  
*1 Model design, programming and documentation.*  
*2 Model review and documentation.*  
*3 General review.*

[💾 Repository](https://github.com/segoedu/value-lens)  
[▶️ Video](https://www.youtube.com/watch?v=TevkbV_W9Ls) 


## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Value Lens](#2-value-lens)
  - [2.1 Value detection model](#21-value-detection-model)
  - [2.2 Functional Interface](#22-functional-interface)
- [3. Simulation and Results](#3-simulation-and-results)
  - [3.1 Data Preprocessing and Value Detection](#31-data-preprocessing-and-value-detection)
  - [3.2 Analysis of Results](#32-analysis-of-results)
- [4. Conclusions](#4-conclusions)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- Figures:
  - [Figure 1. Stage 1: Value Theory Conceptualisation](#figure-1-stage-1-value-theory-conceptualisation)
  - [Figure 2. Stage 2: Value Detection](#figure-2-stage-2-value-detection)

- Tables:
  - [Table 1. Intensity Scale](#table-1-intensity-scale)
  - [Table 2. Performance Comparison](#table-2-performance-comparison)
  - [Table 3. Performance Comparison per Value](#table-3-performance-comparison-per-value)
  - [Table 4. Macro and Micro AVG for Value Lens and BERT-based models](#table-4-macro-and-micro-avg-for-value-lens-and-bert-based-models)

## Abstract

The autonomous decision-making process, which is increasingly applied to computer systems, requires that the choices made by these systems align with human values. In this context, systems must assess how well their decisions reflect human values. To achieve this, it is essential to identify whether each available action promotes or undermines these values. This article presents Value Lens, a text-based model designed to detect human values using generative artificial intelligence, specifically Large Language Models (LLMs). The proposed model operates in two stages: the first aims to formulate a formal theory of values, while the second focuses on identifying these values within a given text. In the first stage, an LLM generates a description based on the established theory of values, which experts then verify. In the second stage, a pair of LLMs is employed: one LLM detects the presence of values, and the second acts as a critic and reviewer of the detection process. The results indicate that Value Lens performs comparably to, and even exceeds, the effectiveness of other models that apply different methods for similar tasks. 

## 1 Introduction

As Artificial Intelligence (AI) systems are increasingly used in daily life to act autonomously, efforts are underway to ensure that their decisions are not only in the interest of the systems’ stakeholders, but also respect moral values and ethical considerations. Still, explicitly modelling human values and determining as to how far decisions are aligned with them is a challenging task. 

Diverse theories, often rooted in Psychology [5, 13, 15, 19], define human values, their characteristics, and how they interact. Several computational models make proposals as to how to incorporate human values into AI decision-making [9, 24, 25, 16, 14]. To this respect, a key aspect is to determine how a decision or course of action promotes or demotes certain values. For this purpose, a model must first detect which values are relevant in a decision context, to then determine their degree of alignment with regard to the decision alternatives. This problem has been addressed in several approaches [29, 7, 2, 28, 18, 17], some of which were proposed in the Human Value Detection 2023 at SemEval-2023 conference [11], whose objective is to use Natural Language Processing techniques to detect human values in text. 

This article describes Value Lens, which relies entirely on Large Language Models (LLMs) to identify human values within a text. The model operates in two stages: the first stage involves conceptualising a specific theory of values, while the second stage focuses on detecting these values. This detection is carried out using a dual approach, which involves using two LLMs for detection and confirmation. 

The remainder of the article is organized as follows: Section 2 outlines the Value Lens model, Section 3 presents the tests conducted and the results achieved, and Section 4 summarizes the key points of the proposed model. 

## 2 Value Lens

To introduce the Value Lens, the Value Detection Model is explained first, followed by the description of the functional interface. 

### 2.1 Value detection model

The proposed model consists of two main stages: (1) conceptualisation and (2) detecting and analysing human values in text. In the first stage, it learns the key characteristics of a reference value theory, which serves as the foundation for value detection. To enhance the understanding of this value theory, contextual data is used to create a more comprehensive description of values (Fig. 1). 

###### Figure 1. Stage 1: Value Theory Conceptualisation.

![Figure 1: Value Theory Conceptualisation.](https://www.dropbox.com/scl/fi/8e61sjbu3tpqc651cws75/figura-1.jpg?rlkey=n0rbihz22jpi1tic5iqd13ouc&st=1yu8leld&dl=1 "Figure 1: Value Theory Conceptualisation.")

The LLM1 utilises a knowledge transfer prompt to integrate complementary knowledge from various sources. This enables the model to effectively leverage external data and produce more accurate and comprehensive outputs about value specification. The prompt specifies how basic information about each value in the text is extracted. 

Then, each value’s name, description, grouping, tags, and examples are stored in a JSON file. Subsequently, a human expert reviews and refines the tags and examples to enhance the quality of the specification. This process also involves adding new tags and examples suggested by the expert. As a result of stage 1, a JSON file with the enriched value specification is returned. 

In the second stage (Fig. 2), a text analysis is conducted based on the value theory conceptualised in the first stage. To accomplish this, a second knowledge transfer prompt is used to specify the task of detecting values by integrating enriched value specifications with the text being analysed. This prompt determines whether the text contains implicit or explicit references to values outlined in value theory. 

###### Figure 2. Stage 2: Value Detection.

![Figure 2: Stage 2: value detection.](https://www.dropbox.com/scl/fi/gnxipypy79beqxmgl78s0/figura-2.jpg?rlkey=zu0eltsxxqviviojcx0pg6ybm&st=4vms2qnn&dl=1 "Figure 2: Stage 2: Value Setection.")

The primary goal is to assess how central each identified value is concerning the overall meaning and motivation of the text. As a result, the text is labelled with the list of all values identified by LLM2. While this information is beneficial, it lacks details about the strength of each detected value’s relationship with the text. Understanding this intensity is essential to establishing value alignment for effective decision-making processes [10]. 

A third knowledge transfer prompt is used for the LLM3 to obtain those intensities. The task defined in this prompt integrates the labelled text, returned by the LLM2, and an intensity scale for values. The LLM3 must determine the degree to which the text promotes or demotes each value. This evaluation should be grounded in textual evidence such as rhetorical emphasis, emotional tone, framing, repetition, placement, and any normative language that affirms or undermines the value. Table 1 lists and describes the levels of the intensity scale. 

###### Table 1. Intensity Scale

| Intensity Level |  #  | Description |
|:----------------:|:-----:|:------------|
| Strong support | 🟩🟩 | The text passionately promotes and defends the value, emphasising its importance. This value is central to the message, reinforced with emotional, moral, and logical urgency. |
| Mild support | 🟩 | The text clearly but gently aligns with the value through positive mention or subtle endorsement, without significant elaboration, insistence, or emphasis. |
| Neutral | ⬜ | The text mentions the value without signalling any clear support or opposition. The tone is factual, balanced and incidental. |
| Mild resistance | 🟥 | The text subtly questions, downplays, or introduces alternative perspectives to its value. This opposition is indirect, hedged, or expressed through soft scepticism. |
| Strong resistance | 🟥🟥 | The text challenges, criticises, or undermines the value directly and forcefully. This includes explicit argumentation, a negative emotional tone, or repeated rejection. |
| Reframing | 🟨 | The text acknowledges its value but redirects its meaning and context. It introduces a new perspective that shifts emphasis without expressing outright support or opposition. |
| No values | ⬛ | The text is technical or descriptive, lacking evaluative. |

At the end of this stage, the text includes labels that indicate the values and intensities associated with it. A justification is also required, providing a clear explanation supported by textual evidence. This approach offers a comprehensive perspective on the insights regarding the values in the text. 

### 2.2 Functional Interface

The Value Lens interface facilitates the characterisation of a value theory and how its values are referred to in text. For this purpose, several academic papers on a specific value theory are carefully selected for conceptualisation. Once the LLM1 generates the specifications based on the documents, the interface displays these specifications in an organised manner for review. 

Experts evaluate the conceptual consistency of the results. They make the necessary adjustments if they identify any errors or discrepancies among the concepts, tags, and examples. Additionally, if they find more suitable tags or examples, they incorporate them into the specification to enhance the overall conceptual understanding. This conceptualisation process serves as a preliminary step for one-time value discovery. However, the new document input and expert review must interact continuously. In other words, the aim is to promote continuous improvement in defining the value theory. 

On this basis, the text to be analysed is loaded from the interface and, using the enriched value descriptions, it is examined by the LLM2. After analysis, the values detected are used as a label associated with the text. The values in the label do not have a particular order; they are simply a value presence indicator. 

Then, the labelled text and the value intensity scale are used by the LLM3 to determine the alignment of the text concerning the detected values. This description includes values, their intensities, and explanations of the alignments. 

The complete Value Lens code is available at the following repository: https://github.com/segoedu/value-lens, and a video describing the system’s functionality can be found at https://www.youtube.com/watch?v=TevkbV_W9Ls. 

## 3 Simulation and Results

This section presents a series of tests to evaluate the model for detecting and analysing human values in text. Then, the results obtained are compared with those of other detection models. Simulations and result analysis have been made in Python3 using the Google Colaboratory environment [4]. The queries were made using the Groq API interface [6] with the meta-llama/llama-4-scout-17b-16e-instruct language model (with temperature = 0.0). 

### 3.1 Data Preprocessing and Value Detection

For simulations, Schwartz’s value theory [20, 19] was selected not only because it is broadly accepted in psychology, but also as a text dataset labelled with Schwartz’s value definition is available. It has also been used widely, allowing for measuring and comparing the proposed model’s performance. 

In the conceptualisation stage, two articles on Schwartz’s value theory are utilised: one presents the foundational model of ten basic values [21], while the other expands this model to include nineteen values [22]. These two descriptions offer sufficient background to understand value theory. This theory defines each value and presents a model illustrating how values interact based on their motivational compatibilities and incompatibilities. Experts have added tags and examples that align with Schwartz’s theory of values to enhance the description provided by the LLM1. 

In the value detection stage, the model shown in Fig. 2 has been adapted to use the Touché24-ValueEval dataset [23] instead of a single text. This dataset contains 59,662 short example texts labelled with the nineteen values of Schwartz’s theory. For the simulations, 7,600 short texts, corresponding to the validation, have been used as input for the LLM2. For the detection process, the labels of these examples were removed and passed to the LLM2, ensuring an unbiased process. In the detection phase, the LLM2 creates a labelled dataset, linking the corresponding values to each sample text. 

Using the intensity scale and the enriched value description, explained in section 2.1, the LLM3 analyses the LLM2-labelled dataset to determine the alignment of the values detected in each text. This information is incorporated into each label, generating the dataset labelled with the presence plus intensity of the values. After the detection and analysis, the original Touché24-ValueEval dataset is utilised to calculate precision, recall, macro F1-score, and micro F1-score. 

### 3.2 Analysis of Results

The initial assessment aims to compare, using the micro F1-score (Table 2), the Value Lens model’s performance against the following similar value detection methods: BERT-based [3], Hierocles of Alexandria [12], Arthur Schopenhauer [27], Philo of Alexandria [26], SCaLAR NITK [8] and Edward Said [1]. 

###### Table 2.  Performance Comparison.

Model | Micro F1-score  
---|:---:  
Hierocles of Alexandria | 0.390  
Arthur Schopenhauer | 0.350  
Value Lens | 0.328  
BERT-based | 0.307  
Philo of Alexandria | 0.290  
SCaLAR NITK | 0.280  
Edward Said | 0.280 

Although Hierocles of Alexandria obtains the maximum value of F1-score micro, the Value Lens model shows adequate behaviour with a value close to the second-best model (Arthur Schopenhauer). Table 3 shows the results for each value concerning the best model to give a more precise idea of the proposed model’s performance. 

Although Table 3 shows that the Hierocles of Alexandria model obtains better results for detecting most values, the F1-Score of the Value Lens model is very close in most classes and even outperforms it in six of them. In addition, Value Lens does not have an F1-score at zero, as is the case for the best model for value Humility. 

###### Table 3. Performance Comparison per Value.

| Value | Hierocles | Value Lens |
|---|:---:|:---:|
| Self-direction: thought | 0.150 | **0.160** |
| Self-direction: action | **0.270** | 0.260 |
| Stimulation | **0.300** | 0.140 |
| Hedonism | 0.370 | **0.400** |
| Achievement | **0.450** | 0.440 |
| Power: dominance | **0.420** | 0.390 |
| Power: resources | **0.490** | 0.280 |
| Face | **0.310** | 0.220 |
| Security: personal | **0.420** | 0.100 |
| Security: societal | **0.490** | 0.370 |
| Tradition | 0.460 | **0.540** |
| Conformity: rules | 0.510 | **0.530** |
| Conformity: interpersonal | **0.240** | 0.140 |
| Humility | 0.000 | **0.100** |
| Benevolence: caring | **0.340** | 0.290 |
| Benevolence: dependability | **0.330** | 0.160 |
| Universalism: concern | **0.470** | 0.360 |
| Universalism: nature | 0.630 | **0.690** |
| Universalism: tolerance | **0.270** | 0.140 |

An interesting aspect is to compare the Value Lens model with these fine-tuned BERT-based models that have similar global performance (similar micro F1-score). To do this, one of these models has been selected to compare performance using average precision and recall. Table 4 shows that the BERT-based model has a precision greater than the recall. This means that when BERT predicts a class, it is more likely to be correct, but tends to omit many positive instances. Value Lens, on the other hand, tends to be more comprehensive. Its recall is significantly higher than its precision; therefore, it identifies many more positive instances but increases false positives. 

###### Table 4. Macro and Micro AVG for Value Lens and BERT-based models

| Model       | Macro Precision | Macro Recall | Micro Precision | Micro Recall |
|-------------|:---------------:|:------------:|:---------------:|:------------:|
| Value Lens  |     0.320       | 0.400        | 0.250           | 0.480        |
| BERT-based  |     0.340       | 0.190        | 0.400           | 0.250        |

In addition, the Value Lens model has an F1-Score macro of 0.301 against 0.232 of the BERT-based model. The macro F1-score averages the F1-score of each class without weighting by the number of instances. A higher macro F1-score suggests that the Value Lens model is generally better at balancing accuracy and recall across the different categories, especially for the less frequent classes. This underscores the model’s ability to better generalise across different classes, including those with fewer training examples. 

## 4 Conclusions

This paper describes the Value Lens model based on LLMs for human value detection and compares its performance to similar models. Value Lens offers a promising approach to value alignment analysis that differs from traditional models. 

Unlike classical methods, this model does not require a training stage to identify values in the text. Instead, it incorporates a stage of conceptualisation based on a specific theory of values, enabling the detection of the values and their intensities within a given text. 

The results show that Value Lens performs in line with similar models. It also has competitive performance and even exceeds the results of the best model analysed in several categories of values. Specifically, against fine-tuned models, with similar global performance, Value Lens is a superior detection model, as evidenced by its higher macro F1 and better performance in most individual classes. 

Another noteworthy aspect is that Value Lens can identify the intensity of each detected value’s presence, thanks to its structure of a second LLM that criticises the detection results. This improves decision-making by considering value promotion or demotion. 

## Acknowledgements

Work been supported by grant VAE: TED2021-131295B-C33 funded by MCIN/AEI/10.13039/501100011033 and by the “European Union NextGenerationEU/PRTR”, and by grant COSASS: PID2021-123673OB-C32 funded by MCIN/AEI/10.13039/501100011033 and by “ERDF A way of making Europe”. 

## References

[1] A. N. Aydin, S. Shaar, and C. Cardie. Edward said at touché: Human value detection using transformers and upsampling. In G. Faggioli, N. Ferro, P. Galuscakova, and A. G. S. Herrera, editors, Working Notes Papers of the CLEF 2024 Evaluation Labs, volume 3740 of CEUR Workshop Proceedings, pages 3379–3383, Sept. 2024. URL http://ceur-ws.org/Vol-3740/paper-323.pdf. 

[2] C. Fang, Q. Fang, and D. Nguyen. Epicurus at semeval-2023 task 4: Improving prediction of human values behind arguments by leveraging their definitions, 2023. URL https://arxiv.org/abs/2302.13925. 

[3] S. García-Rodríguez, M. Karanik, and A. Pina-Zapata. Value promotion scheme elicitation using natural language processing: A model for value-based agent architecture. In Value Engineering in Artificial Intelligence, page 104–120, Berlin, Heidelberg, 2025. Springer-Verlag. ISBN 978-3-031-85462-0. doi: 10.1007/978-3-031-85463-7_7. 

[4] Google. Google colaboratory. https://colab.research.google.com/, 2025. Accessed: 2025-05-01. 

[5] V. V. Gouveia, T. L. Milfont, and V. M. Guerra. Functional theory of human values: Testing its content and structure hypotheses. Personality and Individual Differences, 60:41–47, 4 2014. ISSN 01918869. 

[6] Groq. Groq api. https://console.groq.com/landing/llama-api, 2025. Accessed: 2025-05-01. 

[7] S. Huang, E. Durmus, M. McCain, K. Handa, A. Tamkin, J. Hong, M. Stern, A. Somani, X. Zhang, and D. Ganguli. Values in the wild: Discovering and analyzing values in real-world language model interactions, 2025. URL https://arxiv.org/abs/2504.15236. 

[8] P. K, D. R. K, C. T. Reddy, and A. K. M. Scalar nitk at touché: Comparative analysis of machine learning models for human value identification. In G. Faggioli, N. Ferro, P. Galuscakova, and A. G. S. Herrera, editors, Working Notes Papers of the CLEF 2024 Evaluation Labs, volume 3740 of CEUR Workshop Proceedings, pages 3407–3413, Sept. 2024. URL http://ceur-ws.org/Vol-3740/paper-328.pdf. 

[9] M. Karanik, H. Billhardt, A. Fernández, and S. Ossowski. On the relevance of value system structure for automated value-aligned decision-making. In Proceedings of the 39th ACM/SIGAPP Symposium on Applied Computing, SAC ’24, page 679–686, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400702433. doi: 10.1145/3605098.3636057. 

[10] M. Khamassi, M. Nahon, and R. Chatila. Strong and weak alignment of large language models with human values. Scientific Reports, 14(1):19399, 2024. doi: 10.1038/s41598-024-70031-3. 

[11] J. Kiesel, M. Alshomary, N. Mirzakhmedova, M. Heinrich, N. Handke, H. Wachsmuth, and B. Stein. SemEval-2023 task 4: ValueEval: Identification of human values behind arguments. In A. K. Ojha, A. S. Do˘gruöz, G. Da San Martino, H. Tayyar Madabushi, R. Kumar, and E. Sartori, editors, Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 2287–2303, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.semeval-1.313. URL https://aclanthology.org/2023.semeval-1.313/. 

[12] S. Legkas, C. Christodoulou, M. Zidianakis, D. Koutrintzes, M. Dangioglou, and G. Petasis. Hierocles of alexandria at touché: Multi-task & multi-head custom architecture with transformer-based models for human value detection. In G. Faggioli, N. Ferro, P. Galuscakova, and A. G. S. Herrera, editors, Working Notes Papers of the CLEF 2024 Evaluation Labs, volume 3740 of CEUR Workshop Proceedings, pages 3419–3432, sep 2024. URL http://ceur-ws.org/Vol-3740/paper-330.pdf. 

[13] G. R. Maio. Mental representations of social values. Advances in Experimental Social Psychology, 42:1–43, 2010. ISSN 0065-2601. 

[14] A. Nurwidyantoro, M. Shahin, M. Chaudron, W. Hussain, H. Perera, R. A. Shams, and J. Whittle. Towards a human values dashboard for software development: An exploratory study. In Proceedings of the 15th ACM / IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM), ESEM ’21, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450386654. doi: 10.1145/3475716.3475770. 

[15] M. Rokeach. Rokeach value survey. The nature of human values., 1967. 

[16] T. L. Saaty and N. Begicevic. The scope of human values and human activities in decision making. Applied Soft Computing, 10(4):963–974, 2010. ISSN 1568-4946. doi: 10.1016/j.asoc.2010.04.002. Optimisation Methods & Applications in Decision-Making Processes. 

[17] S. Saha and R. Srihari. Rudolf christoph eucken at semeval-2023 task 4: An ensemble approach for identifying human values from arguments, 2023. URL https://arxiv.org/abs/2305.05335. 

[18] D. Schroter, D. Dementieva, and G. Groh. Adam-smith at semeval-2023 task 4: Discovering human values in arguments with ensembles of transformer-based models, 2023. URL https://arxiv.org/abs/2305.08625. 

[19] S. H. Schwartz. Universals in the content and structure of values. In M. P. Zanna, editor, Advances in Experimental Social Psychology, volume 25, pages 1–65. Academic Press, Ontario, Canada, 1992. 

[20] S. H. Schwartz. Are there universal aspects in the structure and contents of human values? Journal of Social Issues, 50:19–45, 1 1994. ISSN 00224537. 

[21] S. H. Schwartz. An overview of the schwartz theory of basic values. Online readings in Psychology and Culture, 2(1):11, 2012. 

[22] S. H. Schwartz, G. Melech, A. Lehmann, S. Burgess, M. Harris, and V. Owens. Extending the cross-cultural validity of the theory of basic human values with a different method of measurement. Journal of Cross-Cultural Psychology, 32:519–542, 9 2001. ISSN 0022-0221. 

[23] T. V. Team. Touché24-valueeval (2024-08-09) [data set]. Conference and Labs of the Evaluation Forum (CLEF), Grenoble, France. Zenodo, 2024. 

[24] T. L. van der Weide, F. Dignum, J. J. C. Meyer, H. Prakken, and G. A. W. Vreeswijk. Practical reasoning using values. In P. McBurney, I. Rahwan, S. Parsons, and N. Maudet, editors, Argumentation in Multi-Agent Systems, pages 79–93, Berlin, Heidelberg, 2010. Springer Berlin Heidelberg. ISBN 978-3-642-12805-9. 

[25] A. Wyner and T. Zurek. Towards a formalisation of motivated reasoning and the roots of conflict. In Value Engineering in Artificial Intelligence, page 28–45, Berlin, Heidelberg, 2023. Springer-Verlag. ISBN 978-3-031-58204-2. doi: 10.1007/978-3-031-58202-8_3. 

[26] V. Yeste, M. Coll-Ardanuy, and P. Rosso. Philo of alexandria at touché: A cascade model approach to human value detection. In G. Faggioli, N. Ferro, P. Galuscakova, and A. G. S. Herrera, editors, Working Notes Papers of the CLEF 2024 Evaluation Labs, volume 3740 of CEUR Workshop Proceedings, pages 3503–3508, Sept. 2024. URL http://ceur-ws.org/Vol-3740/paper-338.pdf. 

[27] H. Yunis. Arthur schopenhauer at touché 2024: Multi-lingual text classification using ensembles of large language models. In G. Faggioli, N. Ferro, P. Galuscakova, and A. G. S. Herrera, editors, Working Notes Papers of the CLEF 2024 Evaluation Labs, volume 3740 of CEUR Workshop Proceedings, pages 3509–3517, Sept. 2024. URL http://ceur-ws.org/Vol-3740/paper-339.pdf. 

[28] C. Zhang, P. Liu, Z. Xiao, and H. Fei. Mao-zedong at semeval-2023 task 4: Label represention multi-head attention model with contrastive learning-enhanced nearest neighbor mechanism for multi-label text classification, 2023. URL https://arxiv.org/abs/2307.05174.

[29] W. Zhu, Y. Xie, G. Song, and X. Zhang. Eavit: Efficient and accurate human value identification from text data via llms, 2025. URL https://arxiv.org/abs/2505.12792.