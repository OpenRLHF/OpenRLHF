# OpenLLaMA2

<div style="font-size: 1.5rem;">
  <a href="./README.md">English</a> |
  <a href="./README_cn.md">Chinese</a>
</div>

</br>

<h1 align="center">OpenLLaMA2</h1>
<div align="center">
  <a href="[https://github.com/catqaq/ChatPiXiu ↗](https://github.com/catqaq/ChatPiXiu)">
    <img src="./docs/imgs/pixiu.png" alt="Logo" height="210">
  </a>

<p align="center">
    <h3>LLaMA2 training framework for everyone!</h3>
      <a href="https://github.com/openllmai/OpenLLaMA2/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/catqaq/ChatPiXiu" />
      </a>
      <a href="https://github.com/catqaq/ChatPiXiu/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/catqaq/ChatPiXiu?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenLLaMA2/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/openllmai/OpenLLaMA2?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenLLaMA2/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/openllmai/OpenLLaMA2?color=0088ff" />
      <a href="https://github.com/openllmai/OpenLLaMA2/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/openllmai/OpenLLaMA2?color=ccf" />
      </a>
      <br/>
      <em>Open-source ChatGPT / Comprehensive / Lightweight / Easy-to-use</em>
      <br/>
      <a href="https://zhuanlan.zhihu.com/p/622065348/"><strong>Articles</strong></a>
        ·
      <a href="https://zhuanlan.zhihu.com/p/622065348"><strong>Videos</strong></a>
    </p>

</p>
</div>

> **The code is open-source, feel free to use it, contributions are welcome! Note: The license of the model depends on the provider of the model.**

- [💥Latest News](#latest-news)
- [💫OpenNLP Plan](#OpenNLP-plan)
- [💫OpenLLaMA2](#OpenLLaMA2-project)
- [⛏️Usage Steps](#usage-steps)
- [📄Running Example](#running-example)
- [📄Result Display](#result-display)
- [🛠️Common Errors](#common-errors)
- [💐References & Acknowledgements](#references--acknowledgements)
- [🌟Sponsor Us](#sponsor-us)
- [🌈Starchart](#starchart)
- [🏆Contributors](#contributors)

## Latest News

- 2023/8/13: LLaMA 7B + SFT+ RM PPO + DeepSpeed training features finished

- 2023/07/30: OpenLLaMA2 project officially launched:
  - Initial code submission
  - Division of labor among members

## OpenLLaMA2 Project

OpenLLaMA2 is the third official open source project of the OpenNLP plan, and it is the first practice project of openllmai. LLaMA2 is great and powerful, but there are currently no practical LLaMA2 SFT/RLHF training frameworks in the open source community. This project aims to develop an industrial-grade LLaMA2 SFT/RLHF training framework.

The sister project of this project is [chinese-llama2 ↗](https://github.com/OpenLLMAI/chinese-llama2), which aims to fine-tune the Chinese LLaMA2 using SFT/RLHF.

### 2. Development Plan and Organization Division of Labor

#### 2.1 Development Plan:

- Learning and Sharing: LLaMA2 series model learning, DeepSpeed, Ray and other framework learning, etc.;
- Implementation of LLaMA2 SFT/RLHF High-Performance Training Framework

#### 2.2 Organization Division of Labor:

- SFT: Xianyu, Hope, etc.
- RLHF: Chuqi, Li, etc.
- Ray distributed computing framework: qwang, etc.

### 2. Join Us

Who are we?

We are Xianyu Intelligence \[xianyu.ai/openllmai\], members include butnot limited to: researchers, engineers, students, enthusiasts from different industries and fields. We have a common goal: to promote the development of open source NLP, and make the technology more accessible and beneficial to all.

**How to Join?**

1. Email us at openllmai@xianyu.ai. Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenLLaMA2 ↗](https://github.com/openllmai/OpenLLaMA2) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of OpenLLaMA2 project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Usage Steps

1. Clone the repository: `git clone https://github.com/openllmai/OpenLLaMA2.git`
1. Install the required dependencies: `pip install -r requirements.txt`
1. Run the example script: `python example.py`
1. For more details, please refer to the `README.md` file in the repository.

## Running LLaMA Example

```python
# launch nvidia container
cd examples/scripts
./docker_run.sh

# cd in container
cd /openllama2/scripts

# train SFT model
./train_sft_llama.sh

# train RM model
./train_rm_llama.sh

# train PPO model
./train_ppo_llama.sh
```

## Result Display

After completing the training and evaluation, you can evaluate of your model by using the `generate` function:

```python
./inference_llama.sh "Please introduce GPT model."
```

This will display a table with the results of your model, including metrics like loss, accuracy, F1 score, and more.

## Common Errors

Here are some common errors that you might encounter when using OpenLLaMA2:

1. **Dependency issues:** Make sure you have installed all the required dependencies listed in `requirements.txt`.
1. **Data loading issues:** Make sure your dataset is in the correct format and that the file path is correct.
1. **Memory issues:** Training large models can be memory-intensive. Consider reducing the batch size or using a machine with more memory if you encounter memory errors.

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

## Sponsor Us

Your sponsorship can help us maintain and improve OpenLLaMA2. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/openllmai).

## Starchart


[![Star History Chart](https://api.star-history.com/svg?repos=openllmai/OpenLLaMA2&type=Date)](https://star-history.com/#openllmai/OpenLLaMA2&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/openllmai/OpenLLaMA2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openllmai/OpenLLaMA2" />
</a>

______________________________________________________________________

*OpenLLaMA2 © 2023 OpenLLMAI. All Rights Reserved.*