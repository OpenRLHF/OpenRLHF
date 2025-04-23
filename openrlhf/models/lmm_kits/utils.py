from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.configuration_utils import PretrainedConfig
import importlib
import os


def _get_kit_root_path(pretrain_or_model=None,model_type=None):
    assert (pretrain_or_model is not None) ^ (model_type is not None), "only and only one of pretrain_or_model and model_type should be provided"
    if model_type is None:
        # Safely load config without excuting remote code.
        config = PretrainedConfig.from_pretrained(pretrain_or_model)
        model_type = config.model_type
    root_path = f".models.lmm_kits.{model_type}"
    #check if the module exists.
    if not importlib.util.find_spec(root_path,package="openrlhf"):
        # This is an llm or unsupported lmm.
        root_path = f".models.lmm_kits.llm"
    return root_path

def _get_hf_processor(pretrain, padding_side="left", strategy=None, use_fast=True):
    processor_kwargs = strategy.args.processor_kwargs
    # There maybe some patches for the processor
    load_patch(pretrain_or_model=pretrain)
    try:
        processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=False, use_fast=use_fast, **processor_kwargs)
    except OSError:
        # Corner case for gemma-3-1b, which has a processor class but no processor config.
        processor = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=False, use_fast=use_fast, **processor_kwargs)
    if isinstance(processor, PreTrainedTokenizerBase):
        from .llm.data_processor import LLMProcessor
        processor = LLMProcessor(processor)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return processor

def get_data_processor(pretrain_or_model, padding_side="left", strategy=None, use_fast=True):
    root_path = _get_kit_root_path(pretrain_or_model)
    module = importlib.import_module(f"{root_path}.data_processor",package="openrlhf")
    data_processor_cls = getattr(module, "DataProcessor")
    hf_processor = _get_hf_processor(pretrain_or_model, padding_side, strategy,use_fast=use_fast)
    data_processor = data_processor_cls(hf_processor,processor_kwargs=strategy.args.processor_kwargs)
    return data_processor

def load_patch(pretrain_or_model=None,model_type=None, use_liger_kernel=False):
    # only and only one of pretrain_or_model and model_type should be provided
    # use xor to check
    assert (pretrain_or_model is not None) ^ (model_type is not None), "only and only one of pretrain_or_model and model_type should be provided"
    root_path = _get_kit_root_path(pretrain_or_model,model_type)
    module = importlib.import_module(f"{root_path}.patch",package="openrlhf")
    Patch = getattr(module, "Patch")
    Patch.load_all_patches(use_liger_kernel=use_liger_kernel)

def get_generation_cls(pretrain_or_model, use_liger_kernel=False):
    model_type = PretrainedConfig.from_pretrained(pretrain_or_model).model_type
    load_patch(model_type=model_type, use_liger_kernel=use_liger_kernel)
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if model_type in CONFIG_MAPPING:
        # load_patch register customized config if needed, so we ensure AutoConfig loads our customized config, not remote config.
        # This also ensure type(config) mapped to our customized model class.
        config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=False)
    else:
        # Not a model customized by lmm-r1 or supported by transformers, so we trust remote code.
        config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
    if type(config) in AutoModelForImageTextToText._model_mapping:
        return AutoModelForImageTextToText._model_mapping[type(config)]
    else:
        # Compatible with LLMs
        if use_liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel
            _apply_liger_kernel(model_type)
        return _get_causallm_cls_from_pretrained(pretrain_or_model, trust_remote_code=True)

def hack_peft_model(peft_model):
    def get_inputs_embeds(*args,**kwargs):
        return peft_model.base_model.model.get_inputs_embeds(*args,**kwargs)
    def get_position_ids(*args,**kwargs):
        return peft_model.base_model.model.get_position_ids(*args,**kwargs)
    def offset_split_position_ids(*args,**kwargs):
        return peft_model.base_model.model.offset_split_position_ids(*args,**kwargs)
    peft_model.get_inputs_embeds = get_inputs_embeds
    peft_model.get_position_ids = get_position_ids
    peft_model.offset_split_position_ids = offset_split_position_ids
    return peft_model

def _get_causallm_cls_from_pretrained(pretrained_model_name_or_path,**kwargs):
    """
    Modified from AutoModelForCausalLM.from_pretrained, which returns the target model class, not the instance.
    """
    from transformers.models.auto.auto_factory import (
        cached_file,
        extract_commit_hash,
        find_adapter_config_file,
        is_peft_available,
        get_class_from_dynamic_module,
        add_generation_mixin_to_remote_model,
        _get_model_class,
        resolve_trust_remote_code,
        CONFIG_NAME,
    )
    import warnings
    import copy
    import json
    config = kwargs.pop("config", None)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    kwargs["_from_auto"] = True
    hub_kwargs_names = [
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "resume_download",
        "revision",
        "subfolder",
        "use_auth_token",
        "token",
    ]
    hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
    code_revision = kwargs.pop("code_revision", None)
    commit_hash = kwargs.pop("_commit_hash", None)
    adapter_kwargs = kwargs.pop("adapter_kwargs", None)
    token = hub_kwargs.pop("token", None)
    use_auth_token = hub_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        token = use_auth_token
    if token is not None:
        hub_kwargs["token"] = token
    if commit_hash is None:
        if not isinstance(config, PretrainedConfig):
            # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                **hub_kwargs,
            )
            commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
        else:
            commit_hash = getattr(config, "_commit_hash", None)
    if is_peft_available():
        if adapter_kwargs is None:
            adapter_kwargs = {}
            if token is not None:
                adapter_kwargs["token"] = token
        maybe_adapter_path = find_adapter_config_file(
            pretrained_model_name_or_path, _commit_hash=commit_hash, **adapter_kwargs
        )
        if maybe_adapter_path is not None:
            with open(maybe_adapter_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
                adapter_kwargs["_adapter_model_path"] = pretrained_model_name_or_path
                pretrained_model_name_or_path = adapter_config["base_model_name_or_path"]
    if not isinstance(config, PretrainedConfig):
        kwargs_orig = copy.deepcopy(kwargs)
        # ensure not to pollute the config object with torch_dtype="auto" - since it's
        # meaningless in the context of the config object - torch.dtype values are acceptable
        if kwargs.get("torch_dtype", None) == "auto":
            _ = kwargs.pop("torch_dtype")
        # to not overwrite the quantization_config if config has a quantization_config
        if kwargs.get("quantization_config", None) is not None:
            _ = kwargs.pop("quantization_config")
        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            trust_remote_code=trust_remote_code,
            code_revision=code_revision,
            _commit_hash=commit_hash,
            **hub_kwargs,
            **kwargs,
        )
        # if torch_dtype=auto was passed here, ensure to pass it on
        if kwargs_orig.get("torch_dtype", None) == "auto":
            kwargs["torch_dtype"] = "auto"
        if kwargs_orig.get("quantization_config", None) is not None:
            kwargs["quantization_config"] = kwargs_orig["quantization_config"]
    has_remote_code = hasattr(config, "auto_map") and AutoModelForCausalLM.__name__ in config.auto_map
    has_local_code = type(config) in AutoModelForCausalLM._model_mapping.keys()
    trust_remote_code = resolve_trust_remote_code(
        trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
    )
    # Set the adapter kwargs
    kwargs["adapter_kwargs"] = adapter_kwargs
    if has_remote_code and trust_remote_code:
        class_ref = config.auto_map[AutoModelForCausalLM.__name__]
        model_class = get_class_from_dynamic_module(
            class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs
        )
        _ = hub_kwargs.pop("code_revision", None)
        AutoModelForCausalLM.register(config.__class__, model_class, exist_ok=True)
        model_class = add_generation_mixin_to_remote_model(model_class)
        return model_class
    elif type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = _get_model_class(config, AutoModelForCausalLM._model_mapping)
        if model_class.config_class == config.sub_configs.get("text_config", None):
            config = config.get_text_config()
        return model_class
    raise ValueError(
        f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {AutoModelForCausalLM.__name__}.\n"
        f"Model type should be one of {', '.join(c.__name__ for c in AutoModelForCausalLM._model_mapping.keys())}."
    )