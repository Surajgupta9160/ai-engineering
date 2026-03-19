# 13 — Multimodal AI: Text, Images, Audio, and Video

## What Multimodal Means and Why It Matters

```
MODALITIES IN AI
============================================================

  UNIMODAL AI:                    MULTIMODAL AI:
  One input type                  Multiple input types

  ┌──────────┐                    ┌────────────────────────┐
  │  TEXT    │──> LLM ──> TEXT    │  TEXT  + IMAGE + AUDIO │──> LLM ──> TEXT
  └──────────┘                    └────────────────────────┘

WHY MULTIMODAL MATTERS:
  The real world is not just text.
  A doctor looks at X-rays (images) and describes findings (text).
  A call center analyzes tone (audio) and transcripts (text).
  A self-driving car sees road conditions (video) and road signs (image+text).

PRACTICAL IMPACT:
  ┌─────────────────────────────────────────────────────────┐
  │ Without multimodal:                                      │
  │   "Describe what's wrong with my code" (screenshot)     │
  │   → User must manually transcribe the error            │
  │                                                          │
  │ With multimodal:                                         │
  │   [Paste screenshot of error]                            │
  │   → AI sees the exact error message and line numbers     │
  └─────────────────────────────────────────────────────────┘

MODELS AND THEIR CAPABILITIES:
  GPT-4o          → text + images in, text + images out
  GPT-4o-mini     → text + images in, text out (cheaper)
  Claude 3.x      → text + images in, text out
  Gemini 1.5 Pro  → text + images + audio + video in, text out
  DALL-E 3        → text in, image out
  Whisper         → audio in, text out
  TTS             → text in, audio out
```

---

## OpenAI Vision API: URL and Base64

```python
# ============================================================
# OPENAI VISION API — IMAGE INPUT
# Two methods: URL (simpler) and Base64 (for local files)
# ============================================================

import os
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# METHOD 1: IMAGE FROM URL
# Easiest approach — model downloads the image directly
# ============================================================

def analyze_image_url(
    image_url: str,
    question: str = "Describe this image in detail.",
    detail: str = "auto",
) -> str:
    """
    Analyze an image from a public URL.

    Args:
        image_url: Publicly accessible URL to an image
        question: What you want to know about the image
        detail: "auto" (model decides), "low" (faster/cheaper), "high" (more detail)

    Returns:
        Model's description or answer

    Detail levels and token costs:
        "low"  → 85 tokens regardless of image size (fast, cheap)
                  Resolution: 512x512 internally
                  Use for: thumbnails, simple classification

        "high" → 85 tokens base + 170 tokens per 512x512 tile
                  Resolution: up to 2048x2048
                  Use for: reading text, detailed analysis, OCR

        "auto" → Model picks low or high based on image content
    """
    response = client.chat.completions.create(
        model="gpt-4o",                # Must use a vision-capable model
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    image_url,   # Must be public HTTP/HTTPS URL
                            "detail": detail,      # Token cost hint
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ============================================================
# METHOD 2: IMAGE FROM LOCAL FILE (Base64 encoded)
# Use this for local files or when URL is not publicly accessible
# ============================================================

def analyze_local_image(
    image_path: str,
    question: str = "Describe this image in detail.",
    detail: str = "auto",
) -> str:
    """
    Analyze a local image file by encoding it as Base64.

    Supported formats: JPEG, PNG, GIF, WEBP
    Size limit: 20MB per image

    Args:
        image_path: Path to a local image file
        question: What to ask about the image
        detail: "low", "high", or "auto"

    Returns:
        Model's response
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return f"Error: File not found: {image_path}"

    # Determine the MIME type from the file extension
    MIME_TYPES = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    suffix = image_path.suffix.lower()
    mime_type = MIME_TYPES.get(suffix, "image/jpeg")

    # Read the image file and encode it as Base64
    # Base64 converts binary data to ASCII text for JSON transport
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Data URL format: "data:[mime_type];base64,[encoded_data]"
    data_url = f"data:{mime_type};base64,{image_data}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    data_url,   # Base64 data URL
                            "detail": detail,
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ============================================================
# METHOD 3: MULTIPLE IMAGES IN ONE REQUEST
# ============================================================

def compare_images(
    image_urls: list[str],
    question: str = "Compare these images.",
) -> str:
    """
    Send multiple images in a single API call.

    The model receives all images simultaneously and can compare them,
    find differences, or answer questions that span multiple images.

    Args:
        image_urls: List of public image URLs
        question: Comparative question about the images

    Returns:
        Model's comparative analysis
    """
    # Build the content array: one image_url block per image, then the question
    content = []

    for i, url in enumerate(image_urls):
        content.append({
            "type": "text",
            "text": f"Image {i+1}:",   # Label each image for clarity
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "auto"},
        })

    # Add the question at the end
    content.append({"type": "text", "text": question})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=800,
    )
    return response.choices[0].message.content


# Test examples
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
print(analyze_image_url(url, "What do you see in this image?"))
```

---

## Image Use Cases with Code

```python
# ============================================================
# IMAGE USE CASE 1: OCR (Optical Character Recognition)
# Extract text from images, screenshots, photos of documents
# ============================================================

def extract_text_from_image(image_path: str) -> dict:
    """
    Extract all readable text from an image.

    Better than traditional OCR (Tesseract) for:
    - Handwritten text
    - Complex layouts (tables, columns)
    - Mixed text and graphics
    - Low-quality/skewed images

    Args:
        image_path: Path to the image file

    Returns:
        Extracted text and structure information
    """
    # Use "high" detail for OCR — we need to read small text
    raw_response = analyze_local_image(
        image_path,
        question=(
            "Extract ALL text visible in this image. "
            "Preserve the original layout, including line breaks. "
            "If you see a table, format it with | column separators. "
            "If text is partially obscured, note it with [unclear: ...]."
        ),
        detail="high",   # Critical for OCR — use high detail mode
    )

    return {
        "raw_text":  raw_response,
        "image":     image_path,
        "method":    "gpt-4o-vision",
    }


# ============================================================
# IMAGE USE CASE 2: CHART AND GRAPH ANALYSIS
# ============================================================

def analyze_chart(image_path: str) -> dict:
    """
    Extract data and insights from charts, graphs, and visualizations.

    Can handle: bar charts, line graphs, pie charts, scatter plots,
    heatmaps, candlestick charts, and more.

    Args:
        image_path: Path to chart image

    Returns:
        Structured analysis with data points and insights
    """
    analysis = analyze_local_image(
        image_path,
        question=(
            "Analyze this chart or graph carefully. Provide:\n"
            "1. Chart type (bar, line, pie, etc.)\n"
            "2. Title and axis labels if present\n"
            "3. All data points or approximate values you can read\n"
            "4. The main trend or insight (1-2 sentences)\n"
            "5. Any anomalies or noteworthy patterns\n"
            "Format your response as structured text, not a list."
        ),
        detail="high",
    )

    return {"analysis": analysis, "image": image_path}


# ============================================================
# IMAGE USE CASE 3: SCREENSHOT ANALYSIS
# Parse UIs, error messages, and code screenshots
# ============================================================

def analyze_screenshot(screenshot_path: str, context: str = "") -> dict:
    """
    Analyze a screenshot to understand its contents.

    Common use cases:
    - Debug an error from a screenshot
    - Extract data from a web page screenshot
    - Document a UI flow

    Args:
        screenshot_path: Path to the screenshot
        context: Optional context about what the screenshot shows

    Returns:
        Detailed analysis of the screenshot contents
    """
    question = (
        f"Analyze this screenshot. {context}\n"
        "Describe:\n"
        "1. What application or website is shown\n"
        "2. What is currently displayed (page content, dialog, error, etc.)\n"
        "3. Any error messages or warnings — quote them exactly\n"
        "4. What action the user was likely taking\n"
        "5. If this is an error: what is the likely cause and fix?"
    )

    result = analyze_local_image(screenshot_path, question, detail="high")
    return {"analysis": result, "screenshot": screenshot_path}


# ============================================================
# IMAGE USE CASE 4: STRUCTURED DATA EXTRACTION
# Parse forms, invoices, business cards, etc.
# ============================================================

def extract_invoice_data(image_path: str) -> dict:
    """
    Extract structured data from an invoice image.

    Returns data in a consistent format regardless of
    invoice layout or design.
    """
    import json as json_module

    response = analyze_local_image(
        image_path,
        question=(
            "Extract all data from this invoice as JSON with these fields:\n"
            '{"invoice_number": "", "date": "", "vendor": "", '
            '"customer": "", "line_items": [{"description": "", '
            '"quantity": 0, "unit_price": 0, "total": 0}], '
            '"subtotal": 0, "tax": 0, "total": 0, "currency": ""}\n'
            "If a field is not present, use null. Numbers should be numeric, not strings."
        ),
        detail="high",
    )

    # Try to parse the response as JSON
    # The model may include markdown code fences — strip them
    clean = response.strip().strip("```json").strip("```").strip()
    try:
        return json_module.loads(clean)
    except json_module.JSONDecodeError:
        return {"raw_response": response, "parse_error": "Response was not valid JSON"}
```

---

## DALL-E 3: Image Generation

```python
# ============================================================
# DALL-E 3 — TEXT TO IMAGE GENERATION
# ============================================================

import os
import requests
from pathlib import Path
from openai import OpenAI

client = OpenAI()


def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    n: int = 1,
    save_path: str = None,
) -> dict:
    """
    Generate an image using DALL-E 3.

    DALL-E 3 vs DALL-E 2:
      DALL-E 3 has much better prompt adherence, text rendering, and quality.
      Use DALL-E 3 unless you need multiple images at once (DALL-E 3: n=1 only).

    Parameters:
        prompt: Description of the image to generate. Be specific!
                DALL-E 3 automatically enhances your prompt.

        size: Image dimensions
              "1024x1024" — Square (most versatile)
              "1792x1024" — Landscape (16:9 approximately)
              "1024x1792" — Portrait (9:16 approximately)

        quality: "standard" — Faster and cheaper
                 "hd"       — More detail, better consistency ($0.08 vs $0.04)

        style: "vivid" — Hyper-real, dramatic, vibrant colors
               "natural" — More realistic, less stylized photography

        n: Number of images (DALL-E 3 only supports n=1)

        save_path: Optional path to save the generated image

    Returns:
        Dictionary with URL, revised_prompt, and optional local path
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        style=style,
        n=n,
        response_format="url",   # "url" (temporary, expires in 1h) or "b64_json"
    )

    image_data = response.data[0]
    image_url  = image_data.url

    # DALL-E 3 always rewrites your prompt to improve quality
    # The revised_prompt shows what it actually used
    revised_prompt = image_data.revised_prompt

    result = {
        "url":             image_url,
        "revised_prompt":  revised_prompt,
        "original_prompt": prompt,
        "size":            size,
        "quality":         quality,
        "style":           style,
    }

    # Optionally download and save the image locally
    # URLs expire after 1 hour — save to disk for persistence
    if save_path:
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        Path(save_path).write_bytes(img_response.content)
        result["local_path"] = save_path
        print(f"Image saved to: {save_path}")

    return result


# PROMPTING TIPS FOR DALL-E 3
# ============================================================

PROMPT_EXAMPLES = {
    "photorealistic": (
        "A photorealistic close-up photograph of a red maple leaf "
        "covered in morning dew drops, taken with a macro lens, "
        "soft natural lighting, shallow depth of field, "
        "high resolution, shot on Canon R5"
    ),

    "illustration": (
        "A detailed digital illustration of a futuristic city at night, "
        "cyberpunk style, neon lights reflecting in rain-wet streets, "
        "flying cars, holographic advertisements, art by Syd Mead"
    ),

    "simple_diagram": (
        "A clean white-background technical diagram showing the TCP/IP "
        "network stack with labeled layers: Application, Transport, "
        "Internet, Link. Each layer in a different color box with arrows."
    ),

    "with_text": (
        "A professional business card design with the text 'John Smith' "
        "in bold, 'Senior Engineer' beneath it, and a minimalist tech "
        "company logo. Clean, modern typography. Dark blue and white."
    ),
}

# Generate an example image
result = generate_image(
    prompt=PROMPT_EXAMPLES["photorealistic"],
    size="1024x1024",
    quality="hd",
    style="natural",
    save_path="/tmp/generated_image.png",
)
print(f"Image URL: {result['url']}")
print(f"Revised prompt: {result['revised_prompt']}")


def generate_variations_workflow(
    base_description: str,
    count: int = 4,
) -> list[dict]:
    """
    Generate multiple creative variations of the same concept.

    Since DALL-E 3 only supports n=1, we make multiple API calls
    with slightly varied prompts for diversity.
    """
    style_variations = ["vivid", "natural", "vivid", "natural"]
    quality_variations = ["standard", "standard", "hd", "hd"]

    results = []
    for i in range(min(count, 4)):
        variation_prompt = (
            f"{base_description}. "
            f"Variation {i+1}: {'High contrast' if i % 2 == 0 else 'Soft tones'}."
        )
        result = generate_image(
            prompt=variation_prompt,
            style=style_variations[i],
            quality=quality_variations[i],
        )
        results.append(result)
        print(f"Generated variation {i+1}/{count}")

    return results
```

---

## Stable Diffusion Locally

```python
# ============================================================
# STABLE DIFFUSION — LOCAL IMAGE GENERATION
# pip install diffusers transformers torch accelerate
# Requires: NVIDIA GPU with 6GB+ VRAM (or run on CPU, slowly)
# ============================================================

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import os


def load_stable_diffusion(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = None,
) -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion model from HuggingFace Hub.

    Popular models:
      runwayml/stable-diffusion-v1-5   — Fast, well-rounded (SD 1.5)
      stabilityai/stable-diffusion-xl-base-1.0  — Higher quality (SDXL)
      prompthero/openjourney            — Midjourney-style

    Args:
        model_id: HuggingFace model ID or local path
        device: "cuda" (GPU), "mps" (Apple Silicon), "cpu"

    Returns:
        Loaded pipeline ready for generation
    """
    # Auto-detect best available device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"    # Apple Silicon GPU
        else:
            device = "cpu"    # Slow but works anywhere

    print(f"Loading model {model_id} on {device}...")

    # Load with float16 for GPU (halves memory usage)
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,    # Disable for non-restricted use cases
    )

    # Use a faster scheduler (DPM-Solver++ requires fewer steps)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    pipe = pipe.to(device)

    # Enable memory-efficient attention (reduces VRAM usage ~30%)
    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe


def generate_image_local(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str = "blurry, bad quality, distorted, ugly",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 20,   # More steps = better quality, slower
    guidance_scale: float = 7.5,    # Higher = follows prompt more strictly
    seed: int = None,
    output_path: str = "output.png",
) -> Image.Image:
    """
    Generate an image with Stable Diffusion locally.

    Args:
        pipe: Loaded pipeline from load_stable_diffusion()
        prompt: What you want to generate
        negative_prompt: What to avoid in the generation
        width, height: Output dimensions (multiples of 64)
        num_inference_steps: Quality vs speed (20=fast, 50=quality)
        guidance_scale: Prompt adherence strength (7.5 is a good default)
        seed: Random seed for reproducibility (None = random)
        output_path: Where to save the image

    Returns:
        PIL Image object
    """
    # Set seed for reproducibility — same seed = same image
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    # Run the diffusion pipeline
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    image = result.images[0]   # Get first (and only) generated image

    # Save to disk
    image.save(output_path)
    print(f"Image saved to {output_path}")

    return image


# Usage
# pipe = load_stable_diffusion()
# image = generate_image_local(
#     pipe,
#     prompt="A cozy coffee shop in autumn, warm lighting, digital art",
#     negative_prompt="blurry, low quality, distorted faces",
#     num_inference_steps=30,
#     seed=42,
#     output_path="/tmp/coffee_shop.png",
# )
```

---

## Whisper API: All Parameters, Formats, Timestamps

```python
# ============================================================
# WHISPER API — AUDIO TRANSCRIPTION AND TRANSLATION
# ============================================================

import os
from pathlib import Path
from openai import OpenAI

client = OpenAI()


def transcribe_audio(
    audio_file_path: str,
    language: str = None,
    prompt: str = None,
    response_format: str = "json",
    temperature: float = 0.0,
    timestamp_granularities: list[str] = None,
) -> dict:
    """
    Transcribe an audio file to text using Whisper.

    Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
    Maximum file size: 25MB

    Args:
        audio_file_path: Path to audio file

        language: Optional ISO-639-1 language code to speed up transcription
                  e.g., "en", "es", "fr", "de", "ja", "zh"
                  Omit to let Whisper auto-detect the language.

        prompt: Optional text to guide Whisper's output.
                Use this to provide context, specify terminology,
                or hint at spelling: "This is a podcast about Python programming"
                Also useful for maintaining style/punctuation across chunks.

        response_format: Output format:
                  "json"        — {"text": "..."} (default)
                  "text"        — plain text
                  "srt"         — SubRip subtitle format
                  "vtt"         — WebVTT subtitle format
                  "verbose_json"— includes segments with timestamps, confidence

        temperature: Sampling temperature (0.0 = deterministic, more accurate)
                     Increase if you get repetition or stuck loops.

        timestamp_granularities: What level of timestamps to include
                  ["word"]      — timestamp per word
                  ["segment"]   — timestamp per segment (sentence/phrase)
                  ["word", "segment"] — both

    Returns:
        Transcription data in the requested format
    """
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        return {"error": f"File not found: {audio_path}"}

    # Build request parameters — only include non-None values
    params = {
        "model":           "whisper-1",
        "response_format": response_format,
        "temperature":     temperature,
    }
    if language:
        params["language"] = language
    if prompt:
        params["prompt"] = prompt
    if timestamp_granularities and response_format == "verbose_json":
        params["timestamp_granularities"] = timestamp_granularities

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            **params,
        )

    # The response type changes based on response_format
    if response_format == "verbose_json":
        return {
            "text":     transcript.text,
            "language": transcript.language,
            "duration": transcript.duration,
            "segments": [
                {
                    "id":    s.id,
                    "start": s.start,
                    "end":   s.end,
                    "text":  s.text,
                }
                for s in transcript.segments
            ],
        }
    elif response_format == "json":
        return {"text": transcript.text}
    else:
        # srt, vtt, text — returned as a string
        return {"text": str(transcript)}


def transcribe_with_word_timestamps(audio_path: str) -> dict:
    """
    Transcribe audio with per-word timestamps.
    Requires verbose_json format and word granularity.
    """
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],   # Both levels
        )

    # Extract word-level timing
    words = []
    if hasattr(transcript, "words"):
        words = [
            {"word": w.word, "start": w.start, "end": w.end}
            for w in transcript.words
        ]

    return {
        "text":     transcript.text,
        "language": transcript.language,
        "duration": transcript.duration,
        "words":    words,      # Per-word timestamps
        "segments": [           # Per-segment timestamps
            {"start": s.start, "end": s.end, "text": s.text}
            for s in transcript.segments
        ],
    }


def translate_audio_to_english(audio_path: str) -> str:
    """
    Translate non-English audio directly to English text.

    This is a single-step operation: audio → English text.
    Supports the same languages as transcription (~99 languages).

    Note: Only translates TO English (not between arbitrary language pairs).
    For other target languages: transcribe first, then use an LLM to translate.
    """
    with open(audio_path, "rb") as f:
        translation = client.audio.translations.create(
            model="whisper-1",
            file=f,
        )
    return translation.text


def transcribe_long_audio(audio_path: str, chunk_length_ms: int = 300000) -> str:
    """
    Transcribe audio files larger than 25MB by splitting into chunks.

    pip install pydub
    pydub requires ffmpeg: brew install ffmpeg

    Args:
        audio_path: Path to audio file (any length)
        chunk_length_ms: Chunk size in milliseconds (default: 5 minutes)

    Returns:
        Complete concatenated transcription
    """
    from pydub import AudioSegment
    import tempfile

    audio = AudioSegment.from_file(audio_path)

    chunks = []
    previous_transcript = ""   # Use as context/prompt for next chunk

    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start : start + chunk_length_ms]

        # Save chunk to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            chunk.export(tmp.name, format="mp3")
            tmp_path = tmp.name

        try:
            # Use the end of the previous transcript as context prompt
            # This helps maintain consistency across chunk boundaries
            prompt = previous_transcript[-100:] if previous_transcript else None

            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    prompt=prompt,   # Context from previous chunk
                )

            chunks.append(result.text)
            previous_transcript = result.text

            print(f"Transcribed chunk {i+1}: {result.text[:50]}...")
        finally:
            os.unlink(tmp_path)   # Always clean up temporary files

    return " ".join(chunks)


# Example usage
# result = transcribe_audio(
#     "interview.mp3",
#     language="en",
#     prompt="This is a technical interview about software engineering",
#     response_format="verbose_json",
# )
# print(f"Transcript: {result['text']}")
# print(f"Duration: {result['duration']:.1f}s")
```

---

## Local Whisper and faster-whisper

```python
# ============================================================
# LOCAL WHISPER — Run transcription without API calls
# pip install openai-whisper
# pip install faster-whisper  (3x faster with same accuracy)
# ============================================================

# ---- OPTION 1: Official openai-whisper ----

import whisper

def transcribe_local_whisper(
    audio_path: str,
    model_size: str = "base",
    language: str = None,
) -> dict:
    """
    Transcribe audio locally using OpenAI's Whisper model.

    Model sizes (larger = better quality but slower/more RAM):
      tiny   — 39M params,  ~1x speed, low quality
      base   — 74M params,  ~1x speed, decent quality (default)
      small  — 244M params, ~2x slower, good quality
      medium — 769M params, ~5x slower, very good
      large  — 1550M params, ~10x slower, best quality
      large-v3 — Latest and best model

    Args:
        audio_path: Path to audio file (most formats supported via ffmpeg)
        model_size: Whisper model size to use
        language: ISO language code, or None for auto-detection

    Returns:
        Transcription with segments and language
    """
    # Load model (cached after first download)
    model = whisper.load_model(model_size)

    # Transcribe
    result = model.transcribe(
        audio_path,
        language=language,       # None = auto-detect
        verbose=False,           # True = print each segment as it's processed
        word_timestamps=True,    # Include per-word timestamps
        fp16=False,              # fp16=True for GPU, False for CPU
    )

    return {
        "text":     result["text"],
        "language": result["language"],
        "segments": [
            {
                "start": seg["start"],
                "end":   seg["end"],
                "text":  seg["text"],
            }
            for seg in result["segments"]
        ],
    }


# ---- OPTION 2: faster-whisper (recommended) ----
# 3-4x faster than original, same accuracy, less memory

from faster_whisper import WhisperModel


def transcribe_faster_whisper(
    audio_path: str,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
) -> dict:
    """
    Transcribe using faster-whisper (recommended for production).

    faster-whisper uses CTranslate2 backend for 3-4x speedup.

    Args:
        audio_path: Path to audio file
        model_size: Same sizes as original whisper
        device: "auto", "cpu", "cuda"
        compute_type: "auto", "int8", "float16", "float32"
                      int8 = fastest/less accurate, float32 = slowest/accurate

    Returns:
        Transcription with word-level timestamps
    """
    # Initialize model (cached after first use)
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    # Transcribe and get word-level timestamps
    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=5,             # Higher = better quality, slower (default 5)
        word_timestamps=True,    # Enable per-word timestamps
        vad_filter=True,         # Voice Activity Detection: skip silence
    )

    # Consume the generator to get all segments
    segments = []
    all_words = []

    for segment in segments_iter:
        segments.append({
            "start": round(segment.start, 3),
            "end":   round(segment.end, 3),
            "text":  segment.text.strip(),
        })

        if segment.words:
            for word in segment.words:
                all_words.append({
                    "word":        word.word.strip(),
                    "start":       round(word.start, 3),
                    "end":         round(word.end, 3),
                    "probability": round(word.probability, 3),
                })

    full_text = " ".join(s["text"] for s in segments)

    return {
        "text":              full_text,
        "language":          info.language,
        "language_probability": info.language_probability,
        "duration":          info.duration,
        "segments":          segments,
        "words":             all_words,
    }


# Usage example
# result = transcribe_faster_whisper("podcast.mp3", model_size="small")
# print(result["text"])
# for word in result["words"][:10]:
#     print(f"  [{word['start']:.2f}s] {word['word']}")
```

---

## Text-to-Speech: All 6 Voices, tts-1 vs tts-1-hd

```python
# ============================================================
# OPENAI TEXT-TO-SPEECH (TTS)
# Convert text to natural-sounding speech audio
# ============================================================

from pathlib import Path
from openai import OpenAI

client = OpenAI()

# All available voices and their characteristics
VOICES = {
    "alloy":   "Neutral, balanced — good all-purpose voice",
    "echo":    "Analytical, precise — good for technical content",
    "fable":   "British English, expressive — good for storytelling",
    "onyx":    "Deep, authoritative — good for formal content",
    "nova":    "Bright, energetic — good for friendly/casual content",
    "shimmer": "Clear, gentle — good for educational content",
}

# Model comparison
MODELS = {
    "tts-1":    "Lower latency, optimized for real-time, some compression artifacts",
    "tts-1-hd": "Higher quality, better for pre-recorded content, slightly slower",
}


def text_to_speech(
    text: str,
    voice: str = "alloy",
    model: str = "tts-1",
    output_format: str = "mp3",
    speed: float = 1.0,
    output_path: str = "output.mp3",
) -> str:
    """
    Convert text to speech using OpenAI TTS.

    Args:
        text: Text to convert (max ~4096 characters per call)

        voice: One of: alloy, echo, fable, onyx, nova, shimmer
               See VOICES dict above for descriptions

        model: "tts-1" (faster, real-time friendly)
               "tts-1-hd" (higher quality, pre-recorded content)

        output_format: Audio format:
               "mp3"  — most compatible, smaller files (default)
               "opus" — lowest latency, best for streaming
               "aac"  — good quality/size balance
               "flac" — lossless, largest files
               "wav"  — uncompressed, high quality
               "pcm"  — raw audio, for custom processing

        speed: Speaking rate multiplier (0.25 to 4.0)
               1.0 = normal speed
               0.75 = 25% slower (for comprehension)
               1.5 = 50% faster (for skimming)

        output_path: Where to save the audio file

    Returns:
        Path to the saved audio file
    """
    if voice not in VOICES:
        raise ValueError(f"Invalid voice. Choose from: {list(VOICES.keys())}")

    if model not in MODELS:
        raise ValueError(f"Invalid model. Choose from: {list(MODELS.keys())}")

    if not 0.25 <= speed <= 4.0:
        raise ValueError("Speed must be between 0.25 and 4.0")

    # Generate speech
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=output_format,
        speed=speed,
    )

    # Save to file
    response.stream_to_file(output_path)
    print(f"Audio saved to: {output_path}")

    return output_path


def generate_all_voice_samples(text: str, output_dir: str = "/tmp/tts_samples"):
    """
    Generate samples of all 6 voices for comparison.
    Useful for choosing the right voice for your use case.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {len(VOICES)} voice samples...")
    print(f"Text: {text[:60]}...")

    for voice, description in VOICES.items():
        output_path = f"{output_dir}/{voice}.mp3"
        text_to_speech(
            text=text,
            voice=voice,
            model="tts-1",
            output_path=output_path,
        )
        print(f"  {voice:10s}: {description}")

    print(f"\nAll samples saved to {output_dir}/")


def text_to_speech_streaming(text: str, voice: str = "alloy") -> bytes:
    """
    Generate speech with streaming for low-latency playback.

    For real-time applications (voice bots, assistants),
    use streaming to start playing audio before generation is complete.

    Returns:
        Raw audio bytes (MP3 format)
    """
    # Use tts-1 (not hd) for lower latency in streaming
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="opus",   # Opus has lowest latency for streaming
    )

    # Collect all chunks (in production, yield chunks for real-time playback)
    audio_chunks = []
    for chunk in response.iter_bytes(chunk_size=4096):
        audio_chunks.append(chunk)
        # In production: play each chunk immediately instead of collecting
        # e.g., sound_device.play(chunk) or send to WebSocket

    return b"".join(audio_chunks)


# Demo: generate a sample
text_to_speech(
    text=(
        "Welcome to the AI engineering course. Today we will explore "
        "text-to-speech technology and how it enables voice interfaces."
    ),
    voice="nova",
    model="tts-1-hd",   # High quality for pre-recorded content
    speed=1.0,
    output_path="/tmp/welcome.mp3",
)

# Compare all voices
generate_all_voice_samples(
    "The quick brown fox jumps over the lazy dog.",
    output_dir="/tmp/voice_comparison",
)
```

---

## Gemini for Video Understanding

```python
# ============================================================
# GEMINI FOR VIDEO UNDERSTANDING
# pip install google-generativeai
# ============================================================

import os
import time
import google.generativeai as genai
from pathlib import Path


# Configure with API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def analyze_video_gemini(
    video_path: str,
    question: str = "Describe what happens in this video.",
    model: str = "gemini-1.5-pro",
) -> str:
    """
    Analyze a video file using Google Gemini.

    Gemini 1.5 Pro can understand:
    - Up to 1 hour of video
    - What is being said (audio + video together)
    - Actions, objects, text on screen
    - Scene transitions and temporal events

    Supported formats: mp4, mpeg, mov, avi, wmv, mpegps, flv
    Max size: 2GB

    Args:
        video_path: Path to local video file
        question: What you want to know about the video
        model: "gemini-1.5-pro" or "gemini-1.5-flash"

    Returns:
        Gemini's analysis of the video
    """
    video_file = Path(video_path)
    if not video_file.exists():
        return f"Error: File not found: {video_path}"

    print(f"Uploading video: {video_file.name} ({video_file.stat().st_size / 1e6:.1f}MB)...")

    # Upload the video to Gemini's File API
    # Large files are uploaded once and can be reused in multiple requests
    uploaded_file = genai.upload_file(
        path=str(video_file),
        display_name=video_file.name,
    )

    print(f"File uploaded. State: {uploaded_file.state.name}")

    # Wait for the file to be processed (video processing takes a few seconds)
    max_wait = 120   # seconds
    waited   = 0
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(5)
        waited += 5
        if waited > max_wait:
            return f"Error: Video processing timed out after {max_wait}s"

        # Refresh the file state
        uploaded_file = genai.get_file(uploaded_file.name)
        print(f"  Processing... ({waited}s)")

    if uploaded_file.state.name != "ACTIVE":
        return f"Error: Video processing failed. State: {uploaded_file.state.name}"

    print("Video ready. Analyzing...")

    # Initialize the model
    gemini = genai.GenerativeModel(model_name=model)

    # Send the video and question to Gemini
    response = gemini.generate_content(
        contents=[
            uploaded_file,   # The uploaded video file
            question,        # The question about the video
        ],
        request_options={"timeout": 120},   # Video analysis can take a while
    )

    return response.text


def analyze_video_timestamps(video_path: str) -> dict:
    """
    Extract a timestamped summary of video events.

    Asks Gemini to identify when specific things happen in the video
    and return them in a structured format.
    """
    gemini = genai.GenerativeModel("gemini-1.5-pro")

    # Upload video
    uploaded = genai.upload_file(str(video_path))

    # Wait for processing
    while uploaded.state.name == "PROCESSING":
        time.sleep(5)
        uploaded = genai.get_file(uploaded.name)

    if uploaded.state.name != "ACTIVE":
        return {"error": "Processing failed"}

    prompt = """
Analyze this video and provide a timestamped summary.

For each significant event, scene change, or key moment, provide:
- Timestamp (MM:SS format)
- Description of what happens

Also provide:
- Total duration estimate
- Main topics or themes
- Any text shown on screen

Format your response as structured text with clear timestamps.
"""

    response = gemini.generate_content([uploaded, prompt])

    return {
        "analysis":  response.text,
        "video":     str(video_path),
        "model":     "gemini-1.5-pro",
    }


def extract_audio_from_video_gemini(video_path: str) -> str:
    """
    Extract and transcribe speech from a video using Gemini.

    Unlike Whisper which only processes audio, Gemini understands
    both the audio and visual context together.
    """
    gemini = genai.GenerativeModel("gemini-1.5-pro")
    uploaded = genai.upload_file(str(video_path))

    while uploaded.state.name == "PROCESSING":
        time.sleep(5)
        uploaded = genai.get_file(uploaded.name)

    response = gemini.generate_content([
        uploaded,
        (
            "Transcribe all spoken dialogue from this video. "
            "Include speaker labels if multiple people are talking. "
            "Use [MUSIC] or [SOUND EFFECT] for non-speech audio."
        ),
    ])

    return response.text
```

---

## Multimodal RAG: Indexing Images as Descriptions

```python
# ============================================================
# MULTIMODAL RAG
# Index images by generating text descriptions, then search with text
# ============================================================

import os
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()


class MultimodalDocumentIndex:
    """
    A document index that can handle both text and images.

    For images: generate a rich text description, then embed the description.
    For text: embed directly.

    This enables "What images show a dog?" using semantic text search,
    even though the underlying data is images.
    """

    def __init__(self):
        self.documents: list[dict] = []   # [{id, type, content, description, embedding}]

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for text."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],   # Truncate to model's context limit
        )
        return response.data[0].embedding

    def describe_image(self, image_path: str) -> str:
        """
        Generate a rich text description of an image for indexing.

        The description is what gets embedded and searched.
        Write a comprehensive description to maximize searchability.
        """
        from pathlib import Path
        import base64

        img = Path(image_path)
        suffix = img.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        mime = mime_map.get(suffix, "image/jpeg")

        with open(img, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}", "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image comprehensively for a search index. Include:\n"
                            "1. All visible objects and their relationships\n"
                            "2. Colors, textures, and visual attributes\n"
                            "3. Any text visible in the image (quote it exactly)\n"
                            "4. The setting or scene\n"
                            "5. Any people and their attributes (without identifying them)\n"
                            "6. The mood or tone\n"
                            "7. What this image might be used for or searched with\n"
                            "Be thorough — this text is the ONLY way this image will be found."
                        ),
                    },
                ],
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def add_image(self, image_path: str, metadata: dict = None) -> str:
        """
        Add an image to the index by generating and embedding its description.

        Args:
            image_path: Path to the image file
            metadata: Optional metadata (tags, source, date, etc.)

        Returns:
            Document ID
        """
        print(f"Indexing image: {Path(image_path).name}")

        # Generate a rich text description of the image
        description = self.describe_image(image_path)

        # Embed the description (not the image itself)
        embedding = self.embed_text(description)

        doc_id = f"img_{len(self.documents):04d}"
        self.documents.append({
            "id":          doc_id,
            "type":        "image",
            "path":        str(image_path),
            "description": description,
            "embedding":   embedding,
            "metadata":    metadata or {},
        })

        print(f"  Indexed as {doc_id}: {description[:80]}...")
        return doc_id

    def add_text(self, text: str, source: str = None, metadata: dict = None) -> str:
        """Add a text document to the index."""
        embedding = self.embed_text(text)
        doc_id = f"txt_{len(self.documents):04d}"
        self.documents.append({
            "id":          doc_id,
            "type":        "text",
            "content":     text,
            "description": text[:200],   # First 200 chars as preview
            "embedding":   embedding,
            "metadata":    {"source": source, **(metadata or {})},
        })
        return doc_id

    def search(self, query: str, top_k: int = 5, filter_type: str = None) -> list[dict]:
        """
        Search the index using semantic similarity.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            filter_type: "image", "text", or None (both)

        Returns:
            List of matching documents sorted by relevance
        """
        import math

        if not self.documents:
            return []

        query_embedding = self.embed_text(query)

        def cosine_sim(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x**2 for x in a))
            mag_b = math.sqrt(sum(x**2 for x in b))
            return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

        results = []
        for doc in self.documents:
            if filter_type and doc["type"] != filter_type:
                continue
            score = cosine_sim(query_embedding, doc["embedding"])
            results.append({
                "id":          doc["id"],
                "type":        doc["type"],
                "score":       round(score, 4),
                "description": doc["description"][:150],
                "path":        doc.get("path"),
                "metadata":    doc.get("metadata", {}),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def answer_question(self, question: str) -> str:
        """
        Retrieve relevant images and text, then answer a question.
        This is the complete RAG pipeline with multimodal retrieval.
        """
        # Retrieve relevant documents
        retrieved = self.search(question, top_k=3)

        if not retrieved:
            return "No relevant documents found."

        # Build context: include descriptions and image content
        context_parts = []
        image_contents = []

        for doc in retrieved:
            if doc["type"] == "image" and doc.get("path"):
                # For images: include the description as text context
                context_parts.append(f"[Image: {doc['description']}]")

                # Also include the actual image in the request for richer answers
                import base64
                with open(doc["path"], "rb") as f:
                    b64 = base64.standard_b64encode(f.read()).decode()
                suffix = Path(doc["path"]).suffix.lower()
                mime   = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                           "png": "image/png"}.get(suffix[1:], "image/jpeg")
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
                })
            else:
                context_parts.append(f"[Text document: {doc['description']}]")

        # Build the message with both images and text context
        content = []
        content.extend(image_contents)   # Include actual images
        content.append({
            "type": "text",
            "text": (
                f"Context:\n{chr(10).join(context_parts)}\n\n"
                f"Question: {question}"
            ),
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
        )
        return response.choices[0].message.content


# Usage example
index = MultimodalDocumentIndex()

# Index a mix of images and text documents
# index.add_image("/path/to/chart.png", metadata={"topic": "sales"})
# index.add_image("/path/to/diagram.png", metadata={"topic": "architecture"})
# index.add_text("Q3 sales grew by 15% YoY...", source="q3_report.pdf")

# Search and answer
# results = index.search("sales performance data")
# answer = index.answer_question("What was the sales performance in Q3?")
```

---

## Practice Questions

```
PRACTICE QUESTIONS — MULTIMODAL AI
============================================================

VISION:
1.  What is the difference between detail="low" and detail="high" in the
    OpenAI Vision API? Give a use case where you would explicitly choose
    each option. What are the token cost implications?

2.  You have a PDF invoice that you need to extract data from programmatically.
    Describe two approaches (one using Vision, one using a PDF library) and
    the tradeoffs of each.

3.  Write a function that takes a folder of screenshots and returns a list
    of {"filename": ..., "contains_error": bool, "error_message": str}.
    Use GPT-4o Vision.

4.  When would you choose the Base64 encoding method over URL for images?
    What are the size limits? Write the encoding helper function.

IMAGE GENERATION:
5.  Compare DALL-E 3 "vivid" vs "natural" style. Write two prompts for
    generating a product photo, one for each style. Which would you use
    for an e-commerce site and why?

6.  DALL-E 3 always rewrites your prompt. How do you know what prompt
    it actually used? Is this a feature or a problem? When would you
    want to prevent prompt rewriting?

7.  You want to generate 10 variations of a logo concept. DALL-E 3 only
    supports n=1 per call. Write the code to generate 10 variations
    efficiently, including how you would vary the prompts for diversity.

AUDIO (WHISPER / TTS):
8.  List all six OpenAI TTS voices and describe when you would use each.
    Which model (tts-1 vs tts-1-hd) would you use for a podcast intro
    vs a real-time voice assistant? Why?

9.  Your audio file is 45 minutes long (about 150MB). The Whisper API
    has a 25MB limit. Write the complete code to handle this, including
    how you maintain context across chunk boundaries.

10. What is the "prompt" parameter in the Whisper API for? Give two
    concrete examples where providing a prompt significantly improves
    transcription accuracy.

11. faster-whisper claims to be 3-4x faster than original whisper.
    What is the technical reason for this speedup? What trade-offs
    does it make to achieve this?

VIDEO:
12. Gemini can process up to 1 hour of video. Whisper processes audio.
    For a 1-hour video lecture, compare these approaches:
    (a) Gemini video analysis
    (b) Extract audio → Whisper transcription → GPT-4 analysis
    What are the cost, accuracy, and use-case differences?

13. Write a function using Gemini that watches a 10-minute product demo
    video and returns: {"features_shown": [...], "pricing_mentioned": bool,
    "call_to_action": str, "key_timestamps": [...]}

MULTIMODAL RAG:
14. Explain why we embed the IMAGE DESCRIPTION instead of the image
    itself when building a multimodal vector index. What would you
    lose if you tried to embed the raw image bytes?

15. You are building a visual search engine for a furniture catalog with
    5000 product images. Design the complete pipeline: indexing, storage,
    search, and response generation. What would each component be?
    (Hint: think about batch processing, vector database, caching)
```
