# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processor class for Phi4MM
"""
import re
from typing import List, Optional, Tuple, Union
import math
from enum import Enum

import numpy as np
import scipy
import torch
import torchvision

from transformers import AutoFeatureExtractor, AutoImageProcessor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import (
    ImageInput,
    make_list_of_images,
    valid_images,
)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType, logging
from torch.nn.utils.rnn import pad_sequence


logger = logging.get_logger(__name__)

# Special tokens
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN = r'<\|image_\d+\|>'  # For backward compatibility
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN = r'<\|audio_\d+\|>'  # For backward compatibility
_IMAGE_SPECIAL_TOKEN = '<|endoftext10|>'
_AUDIO_SPECIAL_TOKEN = '<|endoftext11|>'
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>', or we can better name it (in `tokenizer_config.json`)
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'


class InputMode(Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


class Phi4MMImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Phi4MM image processor.
    """
    model_input_names = ["input_image_embeds", "image_sizes", "image_attention_mask"]

    def __init__(
        self,
        dynamic_hd,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dynamic_hd = dynamic_hd

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=384, mask_size=27, use_thumbnail=True):
        orig_width, orig_height = image.size

        w_crop_num = math.ceil(orig_width/float(image_size))
        h_crop_num = math.ceil(orig_height/float(image_size))
        if w_crop_num * h_crop_num > max_num:

            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)

        # Calculate the ratio
        ratio_width = target_width / orig_width
        ratio_height = target_height / orig_height
        if ratio_width < ratio_height:
            new_size = (target_width, int(orig_height * ratio_width))
            padding_width = 0
            padding_height = target_height - int(orig_height * ratio_width)
        else:
            new_size = (int(orig_width * ratio_height), target_height)
            padding_width = target_width - int(orig_width * ratio_height)
            padding_height = 0

        attention_mask = torch.ones((int(mask_size*target_aspect_ratio[1]), int(mask_size*target_aspect_ratio[0])))
        if padding_width >= 14:
            attention_mask[:, -math.floor(padding_width/14):] = 0
        if padding_height >= 14:
            attention_mask[-math.floor(padding_height/14):,:] = 0
        assert attention_mask.sum() > 0

        if min(new_size[1], target_height) < 10 or min(new_size[0], target_width) < 10:
            raise ValueError(f'the aspect ratio is very extreme {new_size}')

        image = torchvision.transforms.functional.resize(image, [new_size[1], new_size[0]],)

        resized_img = torchvision.transforms.functional.pad(image, [0, 0, padding_width, padding_height], fill=[255,255,255])

        return resized_img, attention_mask

    def pad_to_max_num_crops(self, images, max_crops=5):
        """
        images: B x 3 x H x W, B<=max_crops
        """
        B, _, H, W = images.shape
        if B < max_crops:
            pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
            images = torch.cat([images, pad], dim=0)
        return images

    def pad_mask_to_max_num_crops(self, masks, max_crops=5):
        B, H, W = masks.shape
        if B < max_crops:
            pad = torch.ones(max_crops - B, H, W, dtype=masks.dtype, device=masks.device)
            masks = torch.cat([masks, pad], dim=0)
        return masks

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # Basic settings.
        img_processor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])
        dyhd_base_resolution = 448

        # Dynamic HD
        base_resolution = dyhd_base_resolution
        images = [image.convert('RGB') for image in images]
        # cover 384 and 448 resolution
        mask_resolution = base_resolution // 14
        elems, image_attention_masks = [], []
        for im in images:
            elem, attention_mask = self.dynamic_preprocess(im, max_num=self.dynamic_hd, image_size=base_resolution, mask_size=mask_resolution)
            elems.append(elem)
            image_attention_masks.append(attention_mask)
        hd_images = [img_processor(im) for im in elems]
        global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(base_resolution, base_resolution), mode='bicubic',).to(im.dtype) for im in hd_images]
        shapes = [[im.size(1), im.size(2)] for im in hd_images]
        mask_shapes = [[mask.size(0), mask.size(1)] for mask in image_attention_masks]
        global_attention_mask = [torch.ones((1, mask_resolution, mask_resolution)) for _ in hd_images]
        hd_images_reshape = [im.reshape(1, 3,
                                            h//base_resolution,
                                            base_resolution,
                                            w//base_resolution,
                                            base_resolution
                                            ).permute(0,2,4,1,3,5).reshape(-1, 3, base_resolution, base_resolution).contiguous() for im, (h, w) in zip(hd_images, shapes)]
        attention_masks_reshape = [mask.reshape(1,
                                            h//mask_resolution,
                                            mask_resolution,
                                            w//mask_resolution,
                                            mask_resolution
                                            ).permute(0,1,3,2,4).reshape(-1, mask_resolution, mask_resolution).contiguous() for mask, (h, w) in zip(image_attention_masks, mask_shapes)]
        downsample_attention_masks = [mask[:,0::2,0::2].reshape(1,
                                            h//mask_resolution,
                                            w//mask_resolution,
                                            mask_resolution//2+mask_resolution%2,
                                            mask_resolution//2+mask_resolution%2
                                            ).permute(0,1,3,2,4) for mask, (h,w) in zip(attention_masks_reshape, mask_shapes)]
        downsample_attention_masks = [mask.reshape(mask.size(1)*mask.size(2), mask.size(3)*mask.size(4))for mask in downsample_attention_masks]
        num_img_tokens = [256 + 1 + int(mask.sum().item()) + int(mask[:,0].sum().item()) + 16 for mask in downsample_attention_masks]

        hd_images_reshape = [torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
        hd_masks_reshape = [torch.cat([_global_mask] + [_mask], dim=0) for _global_mask, _mask in zip(global_attention_mask, attention_masks_reshape)]
        max_crops = max([img.size(0) for img in hd_images_reshape])
        image_transformed = [self.pad_to_max_num_crops(im, max_crops) for im in hd_images_reshape]
        image_transformed = torch.stack(image_transformed, dim=0)
        mask_transformed = [self.pad_mask_to_max_num_crops(mask, max_crops) for mask in hd_masks_reshape]
        mask_transformed = torch.stack(mask_transformed, dim=0)

        returned_input_image_embeds = image_transformed
        returned_image_sizes = torch.tensor(shapes, dtype=torch.long)
        returned_image_attention_mask = mask_transformed
        returned_num_img_tokens = num_img_tokens

        data = {
            "input_image_embeds": returned_input_image_embeds,
            "image_sizes": returned_image_sizes,
            "image_attention_mask": returned_image_attention_mask,
            "num_img_tokens": returned_num_img_tokens,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


AudioInput = Tuple[Union[np.ndarray, torch.Tensor], int]
AudioInputs = List[AudioInput]


def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """Create a Mel filter-bank the same as SpeechLib FbankFC.

    Args:
        sample_rate (int): Sample rate in Hz. number > 0 [scalar]
        n_fft (int): FFT size. int > 0 [scalar]
        n_mel (int): Mel filter size. int > 0 [scalar]
        fmin (float): lowest frequency (in Hz). If None use 0.0.
            float >= 0 [scalar]
        fmax: highest frequency (in Hz). If None use sample_rate / 2.
            float >= 0 [scalar]

    Returns
        out (numpy.ndarray): Mel transform matrix
            [shape=(n_mels, 1 + n_fft/2)]
    """

    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negtive"
    assert fmin < fmax <= sample_rate / 2, "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    # Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)

    khi = max(khi, klo)

    # Spec 2: SpeechLib uses trianges in Mel space
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix


class Phi4MMAudioFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_audio_embeds", "audio_embed_sizes", "audio_attention_mask"]

    def __init__(self, audio_compression_rate, audio_downsample_rate, audio_feat_stride, **kwargs):
        feature_size = 80
        sampling_rate = 16000
        padding_value = 0.0
        super().__init__(feature_size, sampling_rate, padding_value, **kwargs)

        self.compression_rate = audio_compression_rate
        self.qformer_compression_rate = audio_downsample_rate
        self.feat_stride = audio_feat_stride

        self._eightk_method = "fillzero"
        self._mel = speechlib_mel(16000, 512, 80, fmin=None, fmax=7690).T

        self._hamming400 = np.hamming(400)  # for 16k audio
        self._hamming200 = np.hamming(200)  # for 8k audio

    def duration_to_frames(self, duration):
        """duration in s, estimated frames"""
        frame_rate = 10

        num_frames = duration * 1000 // frame_rate
        return num_frames

    def __call__(
        self,
        audios: List[AudioInput],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        # Ref: https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py#L161
        returned_input_audio_embeds = []
        returned_audio_embed_sizes = []
        audio_frames_list = []

        for audio_data, sample_rate in audios:
            audio_embeds = self._extract_features(audio_data, sample_rate)
            audio_frames = len(audio_embeds) * self.feat_stride
            audio_embed_size = self._compute_audio_embed_size(audio_frames)

            returned_input_audio_embeds.append(torch.tensor(audio_embeds))
            returned_audio_embed_sizes.append(torch.tensor(audio_embed_size).long())
            audio_frames_list.append(audio_frames)

        returned_input_audio_embeds = pad_sequence(
            returned_input_audio_embeds, batch_first=True
        )
        returned_audio_embed_sizes = torch.stack(returned_audio_embed_sizes, dim=0)
        audio_frames = torch.tensor(audio_frames_list)
        returned_audio_attention_mask = torch.arange(0, audio_frames.max()).unsqueeze(0) < audio_frames.unsqueeze(1) if len(audios) > 1 else None

        data = {
            "input_audio_embeds": returned_input_audio_embeds,
            "audio_embed_sizes": returned_audio_embed_sizes,
        }
        if returned_audio_attention_mask is not None:
            data["audio_attention_mask"] = returned_audio_attention_mask

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _extract_spectrogram(self, wav, fs):
        """Extract spectrogram features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        if wav.ndim > 1:
            wav = np.squeeze(wav)

        # by default, we extract the mean if stereo
        if len(wav.shape) == 2:
            wav = wav.mean(1)

        # Resample to 16000 or 8000 if needed
        if fs > 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 16000)
            fs = 16000
        elif 8000 < fs < 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 8000)
            fs = 8000
        elif fs < 8000:
            raise RuntimeError(f"Unsupported sample rate {fs}")

        if fs == 8000:
            if self._eightk_method == "resample":
                # Input audio is 8 kHz. Convert to 16 kHz before feature
                # extraction
                wav = scipy.signal.resample_poly(wav, 2, 1)
                fs = 16000
            # Do nothing here for fillzero method
        elif fs != 16000:
            # Input audio is not a supported sample rate.
            raise RuntimeError(f"Input data using an unsupported sample rate: {fs}")

        preemphasis = 0.97

        if fs == 8000:
            n_fft = 256
            win_length = 200
            hop_length = 80
            fft_window = self._hamming200
        elif fs == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
            fft_window = self._hamming400

        # Spec 1: SpeechLib cut remaining sample insufficient for a hop
        n_batch = (wav.shape[0] - win_length) // hop_length + 1
        # Here we don't use stride_tricks since the input array may not satisfy
        # memory layout requirement and we need writeable output
        # Here we only use list of views before copy to desination
        # so it is more efficient than broadcasting
        y_frames = np.array(
            [wav[_stride : _stride + win_length] for _stride in range(0, hop_length * n_batch, hop_length)],
            dtype=np.float32,
        )

        # Spec 2: SpeechLib applies preemphasis within each batch
        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - preemphasis * y_frames_prev) * 32768

        S = np.fft.rfft(fft_window * y_frames, n=n_fft, axis=1).astype(np.complex64)

        if fs == 8000:
            # Need to pad the output to look like 16 kHz data but with zeros in
            # the 4 to 8 kHz bins.
            frames, bins = S.shape
            padarray = np.zeros((frames, bins))
            S = np.concatenate((S[:, 0:-1], padarray), axis=1)  # Nyquist bin gets set to zero

        spec = np.abs(S).astype(np.float32)
        return spec

    def _extract_features(self, wav, fs):
        """Extract log filterbank features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        spec = self._extract_spectrogram(wav, fs)
        spec_power = spec**2

        fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)

        return log_fbank

    def _compute_audio_embed_size(self, audio_frames):
        integer = audio_frames // self.compression_rate
        remainder = audio_frames % self.compression_rate

        result = integer if remainder == 0 else integer + 1

        integer = result // self.qformer_compression_rate
        remainder = result % self.qformer_compression_rate
        result = integer if remainder == 0 else integer + 1  # qformer compression

        return result


class Phi4MMProcessor(ProcessorMixin):
    r"""
    Constructs a Phi4MM processor which raps an image processor, a audio processor, and a GPT tokenizer into a single processor.

    [`Phi4MMProcessor`] offers all the functionalities of [`Phi4MMImageProcessor`] and [`GPT2Tokenizer`]. See the
    [`~Phi4MMProcessor.__call__`] and [`~Phi4MMProcessor.decode`] for more information.

    Args:
        image_processor ([`Phi4MMImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`GPT2Tokenizer`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "audio_processor", "tokenizer"]
    tokenizer_class = "GPT2TokenizerFast"
    image_processor_class = "AutoImageProcessor"  # Phi4MMImageProcessor will be registered later
    audio_processor_class = "AutoFeatureExtractor"  # Phi4MMAudioFeatureExtractor will be registered later

    def __init__(self, image_processor, audio_processor, tokenizer):
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        images: Optional[ImageInput] = None,
        audios: Optional[AudioInputs] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forards the `text`
        and `kwargs` arguments to GPT2Tokenizer's [`~GPT2Tokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        Phi4MMImageProcessor's [`~Phi4MMImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **input_image_embeds** -- Pixel values to be fed to a model.
            - **image_sizes** -- List of tuples specifying the size of each image in `input_image_embeds`.
            - **image_attention_mask** -- List of attention masks for each image in `input_image_embeds`.
            - **input_audio_embeds** -- Audio embeddings to be fed to a model.
            - **audio_embed_sizes** -- List of integers specifying the size of each audio in `input_audio_embeds`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
        """
        image_inputs = self.image_processor(images, return_tensors=return_tensors) if images is not None else {}
        audio_inputs = self.audio_processor(audios, return_tensors=return_tensors) if audios is not None else {}
        inputs = self._convert_images_audios_text_to_inputs(
            image_inputs,
            audio_inputs,
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        # idenfity the input mode
        if len(image_inputs) > 0 and len(audio_inputs) > 0:
            input_mode = InputMode.VISION_SPEECH
        elif len(image_inputs) > 0:
            input_mode = InputMode.VISION
        elif len(audio_inputs) > 0:
            input_mode = InputMode.SPEECH
        else:
            input_mode = InputMode.LANGUAGE
        inputs["input_mode"] = torch.tensor([input_mode.value], dtype=torch.long)

        return inputs

    @property
    def special_image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.special_image_token)

    def get_special_image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.special_image_token)

    @property
    def chat_template(self):
        return self.tokenizer.chat_template

    def _convert_images_audios_text_to_inputs(
        self, images, audios, text, padding=False, truncation=None, max_length=None, return_tensors=None
    ):
        # prepare image id to image input ids
        if len(images) > 0:
            input_image_embeds = images["input_image_embeds"]
            image_sizes = images["image_sizes"]
            image_attention_mask = images["image_attention_mask"]
            num_img_tokens = images['num_img_tokens']
        else:
            input_image_embeds = torch.tensor([])
            image_sizes = torch.tensor([])
            image_attention_mask = torch.tensor([])
            num_img_tokens = torch.tensor([])

        # prepare audio id to audio input ids
        if len(audios) > 0:
            input_audio_embeds = audios["input_audio_embeds"]
            audio_embed_sizes = audios["audio_embed_sizes"]
            audio_attention_mask = audios.get("audio_attention_mask", None)
        else:
            input_audio_embeds = torch.tensor([])
            audio_embed_sizes = torch.tensor([])
            audio_attention_mask = None

        # Replace certain special tokens for compatibility
        # Ref: https://stackoverflow.com/questions/11475885/python-replace-regex
        if isinstance(text, str):
            text = [text]
        assert isinstance(text, list)
        processed_text = [re.sub(_COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN, _IMAGE_SPECIAL_TOKEN, t) for t in text]
        processed_text = [re.sub(_COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN, _AUDIO_SPECIAL_TOKEN, t) for t in processed_text]

        input_ids_list = [self.tokenizer(t).input_ids for t in processed_text]

        img_cnt, audio_cnt = 0, 0  # only needed for later assertion
        image_token_count_iter = iter(num_img_tokens)
        audio_embed_size_iter = iter(audio_embed_sizes.tolist())
        new_input_ids_list = []
        for input_ids in input_ids_list:
            i = 0
            while i < len(input_ids):
                token_id = input_ids[i]
                if token_id == _AUDIO_SPECIAL_TOKEN_ID:
                    token_count = next(audio_embed_size_iter)
                    audio_cnt += 1
                elif token_id == _IMAGE_SPECIAL_TOKEN_ID:
                    token_count = next(image_token_count_iter)
                    img_cnt += 1
                else:
                    i += 1
                    continue
                tokens = [token_id] * token_count
                input_ids = input_ids[:i] + tokens + input_ids[i + 1:]
                i += token_count
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            new_input_ids_list.append(input_ids)
        lengths = torch.tensor([len(input_ids) for input_ids in new_input_ids_list])
        max_len = lengths.max()
        input_ids = input_ids.new_full((len(new_input_ids_list), max_len), self.tokenizer.pad_token_id)
        # batched inference requires left padding
        for i in range(len(new_input_ids_list)):
            input_ids[i, max_len - len(new_input_ids_list[i]):] = new_input_ids_list[i]

        # If the below assertion fails, it might be that input pure-text
        # messages contain image/audio special tokens literally
        # (<|endoftext10|>, <|endoftext11|>).
        assert (
            img_cnt == len(num_img_tokens)
        ), (
            f"Number of image tokens in prompt_token_ids ({img_cnt}) "
            f"does not match number of images ({len(num_img_tokens)})"
        )
        assert (
            audio_cnt == len(audio_embed_sizes)
        ), (
            f"Number of audio tokens in prompt_token_ids ({audio_cnt}) "
            f"does not match number of audios ({len(audio_embed_sizes)})"
        )

        # prepare attention mask
        seq_range = torch.arange(max_len - 1, -1, -1)
        attention_mask = seq_range.unsqueeze(0) < lengths.unsqueeze(1)

        # prepare batch feature
        data = {
            "input_ids": input_ids,
            "input_image_embeds": input_image_embeds,
            "image_sizes": image_sizes,
            "image_attention_mask": image_attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "attention_mask": attention_mask,
            "num_img_tokens": num_img_tokens,
        }

        return BatchFeature(
            data=data
        )

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GPT2Tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GPT2Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + audio_processor_input_names))


AutoImageProcessor.register("Phi4MMImageProcessor", Phi4MMImageProcessor)
AutoFeatureExtractor.register("Phi4MMAudioFeatureExtractor", Phi4MMAudioFeatureExtractor)
