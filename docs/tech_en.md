# ViiTorVoice-NAR Technical Notes

## Model Architecture And Features

### Model Architecture

ViiTorVoice uses a non-autoregressive discrete masked language model. Instead of generating tokens one by one autoregressively, the model completes masked codebooks in the discrete audio token space.

The audio representation uses DualCodec 25 Hz 12-layer codebooks. The 12 discrete codebook layers carry semantic and acoustic information, and work with the masked LM structure to support voice cloning, local editing, and conditional control.

### Local Editing

The local editing task takes source audio, original text, and the complete edited text as input. The service first computes a diff between the original and edited text, then uses alignment results to locate the audio region to replace, and finally completes the masked audio tokens under text and audio-context constraints.

This avoids regenerating the whole utterance, and is suitable for local changes such as short words, short phrases, numbers, names, and discourse markers.

### Emotion Control And Paralinguistic Information Insertion

The model supports inserting emotion and paralinguistic information into text conditions, such as emotion tags, speaking style, laughter, pauses, or other expressive cues. During inference, CFG parameters such as `emotion_guidance_scale` and `nvv_guidance_scale` can strengthen the corresponding control signal.

When the text does not contain the corresponding tag, the related CFG parameters are ignored. When a tag is present, a higher guidance scale usually makes the control signal more obvious, but it should be tuned together with audio quality and naturalness.

### No-ref-text Voice Cloning

ViiTorVoice supports no-ref-text cloning. For voice cloning calls, users may provide only prompt audio without the transcript of that prompt; the model handles the reference audio with no-ref-text logic, reducing the usage burden.

In the HTTP API, `ref_text` defaults to an empty string and `allow_missing_ref_text` defaults to `true`, so the API can be used directly for voice cloning without prompt text.

### First Block Low-latency Inference

ViiTorVoice supports first block inference: given the total audio length and the first block length, the model only generates tokens in the first block to accelerate first-frame generation.

This mode targets low-latency synthesis and streaming-like experiences. End-to-end first-frame latency can reach around 60 ms. Subsequent audio blocks can continue to be generated block by block, balancing first-frame response speed and full-audio quality.

## Technical Approach

ViiTorVoice largely follows the model architecture of OmniVoice. The main differences are in training tasks, audio codebooks, tokenization, and post-training strategy.

### Training Task Extensions

In addition to regular speech generation tasks, training adds two extra branches.

The first branch is first block mode. During training, the total audio length and first block length are given, and the model only needs to generate tokens in the first block. This teaches the model low-latency first-block generation explicitly.

The second branch is edit mode. During training, a continuous segment of audio tokens is randomly selected and masked. The model must complete that segment using text conditions, audio context before and after the mask, and edit conditions. This branch maps directly to local editing during inference.

### DualCodec Codebook

The project uses DualCodec as the source of audio codebooks. Compared with a single mixed representation, DualCodec separates semantic and acoustic information more explicitly, helping the model learn content consistency, speaker preservation, and acoustic detail reconstruction at the same time.

The current setup uses 25 Hz 12-layer codebooks. The 25 Hz frame rate reduces token density along the time dimension, while the 12-layer structure provides enough discrete representation capacity, balancing quality and inference efficiency.

### Language-specific Tokenizer

Text tokenization is language-specific. Chinese, Japanese, and Korean use character-level tokenization to reduce alignment and local-editing ambiguity caused by word segmentation; English and other languages can use word-level or subword-level representations better suited to those languages.

For local editing, Chinese, Japanese, and Korean default to character-level alignment, while English defaults to word-level alignment. The HTTP API can override the automatic choice with `align_granularity`.

### Consistency Distillation And CFG Learning

Post-training uses consistency distillation so the model can keep generation quality with fewer inference steps. The training process also explicitly learns CFG mode, making the model more stable for emotion control, paralinguistic control, and other condition-enhanced scenarios.

This strategy further reduces inference steps, improves end-to-end inference speed, and lowers first-frame waiting time in low-latency scenarios.
