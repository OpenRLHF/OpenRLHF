def add_vision_args(parser):
    group = parser.add_argument_group(title='vision args')
    group.add_argument("--task_type", type=str,
                       default="sft",
                       choices=["sft", "dpo"],
                       help="task type")
    group.add_argument("--model_arch", type=str, choices=["qwen2_vl"],
                       help="model arch",)
    group.add_argument("--dataset_config_path", type=str, default=None,
                       help="the dataset config")

    group.add_argument("--image_resolution", type=int, default=512 * 512,
                       help="The number of pixels of image below this resolution.")
    group.add_argument("--video_resolution", type=int, default=128 * 128,
                       help="The number of pixels of video below this resolution.")
    group.add_argument("--video_fps", type=float, default=2.0,
                       help="The frames to sample per second for video inputs.")
    group.add_argument("--video_maxlen", type=int, default=64,
                       help="The maximum number of sampled frames for video inputs.")

    group.add_argument("--efficient_eos", type=bool, default=False,
                       help="the efficient_eos of VisionEncoderUtils")
    group.add_argument("--processing_num_workers", type=int, default=18,
                       help="num workers processing process")
    group.add_argument("--train_on_prompt", type=bool, default=False,
                       help="train_on_prompt")
    group.add_argument("--mask_history", type=bool, default=False,
                       help="mask_history")
    group.add_argument("--overwrite_cache", type=bool, default=True,
                       help="overwrite_cache")
    group.add_argument("--local_process_index", type=int, default=0,
                       help="local_process_index")
    group.add_argument("--preprocessing_batch_size", type=int, default=1000,
                       help="preprocessing_batch_size")
    group.add_argument("--neat_packing", action="store_true",
                       help="enable sequence packing without cross-attention.")
    
    group.add_argument("--freeze_vision_tower", type=bool, default=True,
                       help="Whether ot not to freeze vision tower in training. default: True")

    return parser
