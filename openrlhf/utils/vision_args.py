def add_vision_args(parser):
    group = parser.add_argument_group(title='vision args')
    group.add_argument("--task_type", type=str,
                       default="sft",
                       choices=["sft", "dpo"],
                       help="task type")
    group.add_argument("--model_arch", type=str, choices=["qwen2_vl"],
                       help="model arch",)
    group.add_argument("--is_visual_model",
                       type=bool,
                       default=False,
                       help="Is the model a vision model? If so, it needs to be set to True.")
    group.add_argument("--freeze_vision_tower",
                       type=bool,
                       default=True,
                       help="whether ot not to freeze vision tower in training. default: True")

    group.add_argument("--processing_num_workers",
                       type=int,
                       default=16,
                       help="num workers processing process")

    group.add_argument("--image_token",
                       type=str,
                       default="<|image_pad|>",
                       help="the image token of your vision model")
    group.add_argument("--video_token",
                       type=str,
                       default="<|video_pad|>",
                       help="the video token of your vision model")


    return parser

def add_extra_dataset_args(parser):
    group = parser.add_argument_group(title='extra dataset args')
    group.add_argument("--conversation_tag", type=str, default="conversations",
                       help="the conversation tag of dataset")
    group.add_argument("--image_tag", type=str, default="images", help="the image tag of dataset")
    group.add_argument("--role_tag", type=str, default="from",
                       help="the role tag of dataset")
    group.add_argument("--content_tag", type=str, default="value",
                       help="the content tag of dataset")

    group.add_argument("--original_user_tag",
                       type=str,
                       default="human",
                       help="the user tag of dataset")
    group.add_argument("--original_assistant_tag",
                       type=str,
                       default="gpt",
                       help="the assistant tag of dataset")
    group.add_argument("--processed_user_tag",
                       type=str,
                       default="user",
                       help="the user tag value of dataset")
    group.add_argument("--processed_assistant_tag",
                       type=str,
                       default="assistant",
                       help="the assistant tag value of dataset")

    return parser
