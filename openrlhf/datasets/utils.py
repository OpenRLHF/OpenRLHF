import os

from datasets import interleave_datasets, load_dataset, load_from_disk


def exist_and_not_none(d, key):
    return key in d and not d[key] is None


def blending_datasets(
    datasets,
    probabilities=None,
    strategy=None,
    seed=42,
    max_count=1e8,
    stopping_strategy="all_exhausted",
    dataset_split="train",
):
    """Blend multiple datasets with optional probability sampling.

    Args:
        datasets (str): Comma-separated list of dataset paths
        probabilities (str, optional): Comma-separated list of probabilities for sampling.
            If None, datasets will be concatenated without probability sampling.
        strategy: Training strategy object
        seed (int): Random seed
        max_count (int): Maximum number of samples per dataset
    """
    datasets = datasets.split(",")
    if probabilities is not None:
        probabilities = list(map(float, probabilities.split(",")))
        assert len(probabilities) == len(datasets)

    data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        elif strategy.args.use_ms:
            from modelscope.msdatasets import MsDataset

            namespace, dataset = dataset.split("/")
            data = MsDataset.load(dataset, namespace=namespace)
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        # Select dataset
        if dataset_split and dataset_split in data:
            data = data[dataset_split]
        data = data.select(range(min(max_count, len(data))))
        data_list.append(data)

    # merge datasets
    if strategy.is_rank_0():
        print(data_list)

    # If probabilities is None, concatenate datasets directly
    if probabilities is None:
        from datasets import concatenate_datasets

        dataset = concatenate_datasets(data_list)
    else:
        dataset = interleave_datasets(
            data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )

    return dataset
