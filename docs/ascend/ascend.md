# OpenRLHF x Ascend

我们在 OpenRLHF 上增加对华为昇腾设备的支持。**本代码仓由社区进行维护更新，不进行任何商业交付。**

## 硬件支持

* Atlas 800T A2

## 版本分支管理

### 分支说明

* **main 分支**：跟随 OpenRLHF 主仓演进。
* **v[OpenRLHF主仓tag].ascend 分支**： 对应适配 OpenRLHF 主仓 tag，例如，`v0.6.2.ascend` 对应 OpenRLHF 主仓的 `v0.6.2` tag。目的是在 main 分支演进及周边配套软件更新时，确保有稳定可用的代码版本。因此，在分支创建后，只做昇腾相关的 BugFix，不引入新的适配功能。

### 分支配套版本

**下表提供昇腾配套软件版本仅为建议，不作为任何商业交付承诺**。

<table>
  <tr>
    <th align="left">OpenRLHF 主仓Tag</th>
    <td>v0.7.5</td>
    <td>v0.6.2</td>
  </tr>
  <tr>
    <th align="left">对应的 OpenRLHF NPU 适配分支</th>
    <td>main</td>
    <td>v0.6.2.ascend</td>
  </tr>
  <tr>
    <th align="left">vLLM 版本</th>
    <td>v0.8.5.post1</td>
    <td>v0.7.3</td>
  </tr>
  <tr>
    <th align="left">vLLM Ascend 版本/分支</th>
    <td>v0.8.5rc1</td>
    <td>v0.7.3</td>
  </tr>
  <tr>
    <th align="left">torch npu 版本 (pip install 安装)</th>
    <td>2.5.1</td>
    <td>2.5.1</td>
  </tr>
  <tr>
    <th align="left">CANN 版本</th>
    <td><a href="https://github.com/vllm-project/vllm-ascend/blob/v0.7.3/docs/source/installation.md?plain=1#L72-L96">CANN 8.1.RC1</a></td>
    <td><a href="https://github.com/vllm-project/vllm-ascend/blob/v0.7.3/docs/source/installation.md?plain=1#L72-L96">CANN 8.1.RC1</a></td>
  </tr>
  <tr>
    <th align="left">不支持功能</th>
    <td>AutoTP</br>Hybrid Engine</br>Pytorch Compile</br>bitsandbytes</td>
    <td>Ring Attention</br>Hybrid Engine</br>Pytorch Compile</br>bitsandbytes</td>
  </tr>
</table>

**说明**

1. **注意**：当前 vLLM Ascend 的 **v0.8.5rc1** 版本会出现 device error 报错。在使用过程中，需要将[该行代码](https://github.com/vllm-project/vllm-ascend/blob/v0.8.5rc1/vllm_ascend/platform.py#L48)注释。

   ```python
   # os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
   ```

2. 在使用 vLLM Ascend 的 **v0.8.5rc1** 时，如果想要开启 V1 引擎，需要通过环境变量手动配置。

   ```shell
   export VLLM_USE_V1=1
   ```

## 环境准备

### vLLM

为了保证能够在 OpenRLHF 上正常使用 vLLM，需要安装 vLLM Ascend 插件（`vllm-ascend`）。vLLM Ascend 插件的安装方式和镜像请参考[安装教程](https://vllm-ascend.readthedocs.io/en/latest/installation.html)。

```shell
git clone -b [版本号] https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install .

git clone -b [版本号] https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

### Ring Attention Ascend

为了支持 Ring Attention 功能，我们在 [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) 的基础上适配昇腾 FA 算子接口，提供 [ring-attention-ascend](https://github.com/ji-huazhong/ring-attention-ascend) 进行使用。

```shell
git clone https://github.com/ji-huazhong/ring-attention-ascend.git
cd ring-attention-ascend
pip install -e .
```

### 源码安装

```shell
git clone -b [分支名] https://github.com/zhuo97/OpenRLHF.git
cd OpenRLHF
TARGET_DEVICE=NPU pip install -e .
```

### Ray

可通过如下方式在华为昇腾设备上启动 Ray:
```shell
# launch the master node of ray in container
ray start --head --port 6379

# if you want to launch ray on more nodes, use
ray start --address='MASTER-NODE-ADDRESS:6379'
```

训练脚本提交方式与英伟达 GPU 相同。

### 其他第三方库说明

| 软件                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [flash_attn](https://github.com/Dao-AILab/flash-attention)   | 原生不支持，通过在 transformers 适配昇腾FA算子进行支持（[PR](https://github.com/huggingface/transformers/pull/36696)）。 |
| [ring_flash_attn](https://github.com/zhuzilin/ring-flash-attention) | 原生不支持，通过提供 [ring_attn_ascend](https://github.com/ji-huazhong/ring-attention-ascend) 进行支持。 |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 原生不支持。                                                 |

## 支持的算法

### 精度对比

根据经验，我们期望在相同配置下，在华为昇腾设备上的 Loss 与英伟达 GPU 的 Loss/Reward 平均绝对误差小于 2%，具体计算方式如下：

```math
Mean Error=\frac{\sum^N_{i=1}|loss_i^{npu}-loss_i^{gpu}|}{N}\times 100 \%
```

其中，N 表示训练的步数。更多信息请参考[精度计算说明](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/LMaccuracy_0001.html)。

### 进展

已支持的算法仅在下表提供的版本进行过测试。

| 算法                             | 进展           | 与GPU误差 | torch 版本 | torch_npu 版本 | CANN 版本 | 详细结果                                                     |
| -------------------------------- | -------------- | --------- | ---------- | -------------- | --------- | ------------------------------------------------------------ |
| SFT                              | 已支持         | 0.19%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2567488539) |
| DPO                              | 已支持         | 1.81%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2735122006) |
| KTO                              | 已支持         | 0.37%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2642104300) |
| RM                               | 已支持         | 0.85%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2642104300) |
| PRM                              | 已支持         | 1.61%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2642104300) |
| PPO                              | 精度测试中     |           | 2.5.1      | 2.5.1          | 8.1.RC1   |                                                                                   |
| REINFORCE++                      | 已支持         | 1.94%     | 2.5.1      | 2.5.1          | 8.1.RC1   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2735138695) |
| REINFORCE++<br/>(vLLM V1引擎)    | 已支持         | 1.99%     | 2.5.1      | 2.5.1          | 8.1.RC1   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2903104059)  |
| REINFORCE++<br/>(Ring Attention) | 已支持，待优化 | 4.95%     | 2.5.1      | 2.5.1          | 8.1.RC1   |  [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2903104059) |
| GRPO                             | 已支持         | 0.61%     | 2.5.1      | 2.5.1          | 8.1.RC1   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2764993841) |
| GRPO <br>(vLLM V1引擎)           | 已支持         | 1.10%     | 2.5.1      | 2.5.1          | 8.1.RC1   |  [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2903104059)|


## 常见问题

* 使用 `--adam_offload` 参数可能存在长时间卡顿的情况，解决方法是删除 torch_extensions 的缓存文件，参考 [issue](https://github.com/deepspeedai/DeepSpeed/issues/2816#issuecomment-1450095538)。  

## 贡献者

[zhuo97](https://github.com/zhuo97), [zheliuyu](https://github.com/zheliuyu), [FightingZhen](https://github.com/FightingZhen), [obj12](https://github.com/obj12), [tongtong0613](https://github.com/tongtong0613), [Keilo001](https://github.com/Keilo001), [Tonyztj](https://github.com/Tonyztj), [ji-huazhong](https://github.com/ji-huazhong)
