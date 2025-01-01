# EmpathyRobot: A Dataset and Benchmark for Empathetic Task Planning of Robotic Agent

Official repository for the paper "[EmpathyRobot: A Dataset and Benchmark for Empathetic Task Planning of Robotic Agent]()".

[[📖 Paper]()] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/EmpathaticEmbodiedAI/EmpathyRobot/tree/main)] [[🔥model](https://huggingface.co/EmpathaticEmbodiedAI/EmpathyRobotRY_LlaMa3_8B_RLHF/tree/main)]

## 👀 About EmpathyRobot

**Empathy** is a fundamental instinct and essential need for humans, as they both demonstrate empathetic actions toward others and receive empathetic support. Although existing emotion agents have explored how to understand humans’ empathetic needs, they lack to further enable robots to generate **empathy-oriented task planning**, neglecting the **evaluation of empathetic behaviors**. 

<p align="center">
    <img src="figs/figure1.png" width="80%"> <br>
</p>

To address this gap, we introduce **EmpathyRobot**, the **first** dataset specifically designed to benchmark and enhance the empathetic actions of agents across diverse scenarios. This dataset contains **10,000** samples based on human feedback, encompassing information from various modalities and corresponding empathetic task planning sequences, including
navigation and manipulation. Agents are required to **perform actions based on their understanding of both the visual scene and human emotions**. 

<p align="center">
    <img src="figs/figure2.png" width="80%"> <br>
</p>

### Key statistics of EmpathyRobot:

You can download the dataset on [Hugging Face Dataset](https://huggingface.co/datasets/EmpathaticEmbodiedAI/EmpathyRobot/tree/main).

| Statistic                                         | Number |
| :------------------------------------------------ | ------ |
| Total Data Points                                 | 10k    |
| Characters                                        | 100    |
| Input Action-Video                                | 20     |
| Scenarios and Dialogues per Character-Action pair | 5      |
| Empathy Response per Data Point                   | 2      |
| Optional Action Space for Output                  | 50     |
| Average Length of Action-Video                    | 16.28s |
| Max Length of Action-Video                        | 24.60s |
| Min Length of Action-Video                        | 9.40s  |

## 📈 Evaluation Results

Performance on three key stages: *Scenario Understanding* (internal empathetic process), *Empathetic Planning* (formulating an empathetic outcome), and *Empathetic Actions* (implementing the response in a real-world context). 

| Task/Metric                | GPT-4o    | GPT-4-turbo | GPT-4-vision | LLaVA |
| -------------------------- | --------- | ----------- | ------------ | ----- |
| **Scenario Understanding** |           |             |              |       |
| Bleu-1                     | **19.1**  | 14.1        | 15.2         | 13.7  |
| Bleu-4                     | **5.3**   | 3.1         | 3.3          | 2.7   |
| ROUGE-L                    | **23.7**  | 20.4        | 21.4         | 15.6  |
| CIDEr                      | **8.8**   | 1.6         | 3.1          | 7.2   |
| SPICE                      | **14.8**  | 10.1        | 12.1         | 8.9   |
| BERTScore                  | **0.622** | 0.612       | 0.615        | 0.576 |
| **Empathetic Planning**    |           |             |              |       |
| Bleu-1                     | **30.8**  | 25.7        | 25.9         | 13.1  |
| Bleu-4                     | **12.0**  | 6.9         | 6.4          | 2.6   |
| ROUGE-L                    | **26.1**  | 23.5        | 23.4         | 17.3  |
| CIDEr                      | **25.9**  | 14.9        | 15.5         | 3.7   |
| SPICE                      | **16.7**  | 14.5        | 11.8         | 8.4   |
| BERTScore                  | **0.641** | 0.621       | 0.625        | 0.568 |
| **Empathetic Actions**     |           |             |              |       |
| Overlap                    | 27.60     | 32.14       | **35.20**    | 17.19 |
| TF-IDF                     | 21.03     | 24.76       | **27.69**    | 12.09 |
| LCS                        | 25.17     | 28.92       | **29.58**    | 15.21 |

## :white_check_mark: Citation
