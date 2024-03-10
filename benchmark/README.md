# OpenRLHF Benchmark

## Setting
We basically follow [deepspeed-chat's benchmark settings](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/BenckmarkSetting.md) with some differences
- max_prompt_seq_len=1024 and max_answer_seq_len=1024
- init reward model from actor model with same model size
- disable Hybrid Engine as it's not work well with ZeRO-3 [microsoft/DeepSpeed#4469](https://github.com/microsoft/DeepSpeed/issues/4469)

For all model sizes and comparisions, we use a fixed global train_batch_size=1024, then we adjust micro_train_batch_size and micro_rollout_batch_size to the largest size that doesn't cause out of memory error.

**Hardware**
- A800-SXM-80GB

**Software**
- torch==2.1.2
- deepspeed==0.13.2
- transformers==4.38.2
- vllm==0.3.1
- flash-attn==2.3.6

**Hyperparameter**
- rollout_batch_size：1024
- train_batch_size: 128

## Benchmark
<table>
    <tr>
        <td rowspan="2">Size</td>
        <td rowspan="2">Method</td>
        <td rowspan="2">GPU</td>
        <td colspan="3">Generation</td>
        <td colspan="3">Training</td>
        <td colspan="3">End-to-End</td>
    </tr>
    <tr>
        <td>micro_batch_size</td>
        <td>Latency</td>
        <td>TFLOPs</td>
        <td>micro_batch_size</td>
        <td>Latency</td>
        <td>TFLOPs</td>
        <td>Latency</td>
        <td>TFLOPs</td>
        <td>Samples/sec</td>
    </tr>
    <tr>
        <td rowspan="4">LLama2-7B</td>
        <td>DeepspeedChat</td>
        <td>8</td>
        <td>8</td>
        <td>697.12</td>
        <td>5.32</td>
        <td>8</td>
        <td>492.01</td>
        <td>60.34</td>
        <td>1189.13</td>
        <td>28.08</td>
        <td>0.86</td>
    </tr>
    <tr>
        <td>OpenRLHF</td>
        <td>8</td>
        <td>8</td>
        <td>920.81</td>
        <td>4.03</td>
        <td>8</td>
        <td>256.97</td>
        <td>115.52</td>
        <td>1177.78</td>
        <td>28.36</td>
        <td>0.87</td>
    </tr>
    <tr>
        <td>OpenRLHF(ray)</td>
        <td>1:1:4:2</td>
        <td>16</td>
        <td>1498.77</td>
        <td>2.48</td>
        <td>8</td>
        <td>395.69</td>
        <td>75.02</td>
        <td>1894.46</td>
        <td>17.63</td>
        <td>0.54</td>
    </tr>
    <tr>
        <td>OpenRLHF(ray+vllm)</td>
        <td>1:1:2:2:2</td>
        <td>16</td>
        <td>481.86</td>
        <td>7.70</td>
        <td>8</td>
        <td>403.29</td>
        <td>75.02</td>
        <td>885.16</td>
        <td>37.73</td>
        <td>1.16</td>
    </tr>
    <tr>
        <td rowspan="3">LLama2-13B</td>
        <td>DeepspeedChat</td>
        <td>8</td>
        <td>2</td>
        <td>2590.72</td>
        <td>2.74</td>
        <td>2</td>
        <td>1218.76</td>
        <td>46.61</td>
        <td>3809.36</td>
        <td>16.78</td>
        <td>0.27</td>
    </tr>
    <tr>
        <td>OpenRLHF(ray)</td>
        <td>1:1:4:2</td>
        <td>8</td>
        <td>2722.51</td>
        <td>2.61</td>
        <td>4</td>
        <td>949.10</td>
        <td>59.86</td>
        <td>3671.61</td>
        <td>17.41</td>
        <td>0.28</td>
    </tr>
    <tr>
        <td>OpenRLHF(ray+vllm)</td>
        <td>1:1:2:2:2</td>
        <td>8</td>
        <td>1425.75</td>
        <td>4.98</td>
        <td>4</td>
        <td>967.15</td>
        <td>58.74</td>
        <td>2392.91</td>
        <td>26.71</td>
        <td>0.43</td>
    </tr>
    <tr>
        <td rowspan="2">LLama2-34B</td>
        <td>DeepspeedChat</td>
        <td>8</td>
        <td colspan="9">N/A</td>
    </tr>
    <tr>
        <td>OpenRLHF(ray+vllm)</td>
        <td>1:1:2:2:2</td>
        <td>4</td>
        <td>5518.07</td>
        <td>3.84</td>
        <td>2</td>
        <td>3296.18</td>
        <td>51.49</td>
        <td>8814.25</td>
        <td>21.66</td>
        <td>0.12</td>
    </tr>
    <tr>
        <td rowspan="3">LLama2-70B</td>
        <td>DeepspeedChat</td>
        <td>16</td>
        <td colspan="9">N/A</td>
    </tr>
    <tr>
        <td>OpenRLHF(ray+vllm)</td>
        <td>2:2:4:4:4</td>
        <td>4</td>
        <td>7122.65</td>
        <td>2.48</td>
        <td>4</td>
        <td>2699.97</td>
        <td>52.28</td>
        <td>9822.62</td>
        <td>16.17</td>
        <td>0.10</td>
    </tr>
    <tr>
        <td>NeMo-Aligner</td>
        <td colspan="10">TBD</td>
    </tr>
</table>