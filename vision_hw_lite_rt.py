
#!/usr/bin/env python3
"""
Scene-dependent pipeline (Perplexity-only guidance using final path image).
- STT (Whisper) for query
- YOLO detection to find requested target in the video
- Semantic segmentation (ADE20K) to compute walkable mask
- DepthPro depth estimation
- Inflation + A* path planning + obstacle inspection
- Final guidance via Perplexity Sonar API using final path image (base64 data URI) + text
- TTS (VITS) to speak final guidance
"""

import os
import math
import csv
import logging
from pathlib import Path
from math import sqrt, ceil
from heapq import heappush, heappop
import string
import gc
import base64
import spacy
import numpy as np
import cv2
import PIL.Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import soundfile as sf
import scipy.io.wavfile as wavfile

from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, VitsModel, AutoModelForSequenceClassification, AutoTokenizer
from perplexity import Perplexity

# DepthPro imports (must be available)
from depth_pro import create_model_and_transforms, load_rgb
from mit_semseg.models import ModelBuilder, SegmentationModule

# ---------------------------
# CONFIG
# ---------------------------
CAMERA_INTRINSICS = {'fx': 910.0, 'fy': 910.0, 'cx': 960.0, 'cy': 540.0}
INFLATION_RADIUS_M = 0.12
DETECTION_CONF_THRESH = 0.5
MIN_WALKABLE_AREA = 5000

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_PATH = "/dev/video42"
AUDIO_PATH = "test.wav"

logging.basicConfig(level=logging.ERROR)
LOGGER = logging.getLogger("scene_pipeline")

# ---------------------------
# Perplexity client init (assume available)
# ---------------------------
try:
    perplexity_client = Perplexity()
    LOGGER.info("Perplexity client initialized.")
except Exception as e:
    LOGGER.error(f"Perplexity init failed: {e}")
    perplexity_client = None

# ---------------------------
# Semantic segmentation names and utilities
# ---------------------------
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

walkable_classes = [
    'floor','sidewalk','earth','road','runway','path','ground',
    'pavement','grass','field','flooring','soil','land','route','rug','carpet','carpeting'
]
walkable_ids = [idx-1 for idx,name in names.items() if name in walkable_classes]

def load_bert(device):
    intent_classifier_path = "./intent_classifier_bert_final"
    query_model = AutoModelForSequenceClassification.from_pretrained(intent_classifier_path).to(device)
    query_tokenizer = AutoTokenizer.from_pretrained(intent_classifier_path)
    print(f"Intent Classifier Device{next(query_model.parameters()).device}")
    query_model.eval()
    id2label = {0: "scene-dependent", 1: "path-finding", 2: "general"}
    return query_model, query_tokenizer, id2label


def load_segmentation(device):
    """
    Loads ADE20K semantic segmentation module onto device.
    """
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated', fc_dim=2048,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup', fc_dim=2048, num_class=150,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.to(device)
    segmentation_module.eval()
    return segmentation_module

PALETTE = (np.arange(150, dtype=np.uint8) * 17)[:, None]
PALETTE = np.concatenate([PALETTE, (PALETTE * 2) % 255, (PALETTE * 3) % 255], axis=1)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def visualize_segmentation(pred):
    color = PALETTE[pred % PALETTE.shape[0]]
    return color.astype(np.uint8)

# ---------------------------
# STT / TTS
# ---------------------------
def load_stt_tts(device):
    # Whisper STT (local directory expected)
    asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    asr_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    # VITS TTS
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
    tts_processor = AutoProcessor.from_pretrained("facebook/mms-tts-eng")
    return asr_model, asr_processor, tts_model, tts_processor

# def transcribe_audio(asr_model, asr_processor, audio_path, device):
#     """
#     Converts a .wav audio file to text using Whisper STT.
#     """
#     try:
#         audio_input, sample_rate = sf.read(audio_path, dtype="float32")
#         inputs = asr_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")

#         if hasattr(asr_model, "dtype"):
#             inputs = {k: v.to(device=device, dtype=asr_model.dtype) for k, v in inputs.items()}
#         else:
#             inputs = {k: v.to(device) for k, v in inputs.items()}

#         with torch.inference_mode():
#             predicted_ids = asr_model.generate(**inputs)
#         transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#         print(f"🗣️ Transcription: {transcription}")
#         return transcription
#     except Exception as e:
#         LOGGER.error(f"STT error: {e}")
#         return None

def transcribe_audio(asr_model, asr_processor, audio_path, device):
    """
    Converts a .wav audio file to text using Whisper STT.
    """
    try:
        # Always load as float32 mono
        audio_input, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)

        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Normalize range [-1,1]
        audio_input = np.clip(audio_input / np.max(np.abs(audio_input) + 1e-6), -1.0, 1.0)

        inputs = asr_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device, dtype=torch.float32) for k, v in inputs.items()}

        with torch.inference_mode():
            predicted_ids = asr_model.generate(
                **inputs,
                task="transcribe",
                language="en",
                max_new_tokens=128
            )

        transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"🗣️ Transcription: {transcription}")
        return transcription.strip()
    except Exception as e:
        LOGGER.error(f"STT error: {e}")
        return None


def synthesize_speech(tts_model, tts_processor, text, output_path, device):
    """
    Converts text to speech (.wav) using VITS TTS.
    """
    try:
        inputs = tts_processor(text=text, return_tensors="pt").to(device)
        with torch.inference_mode():
            speech = tts_model(**inputs).waveform
        speech_np = speech.squeeze().cpu().numpy()
        wavfile.write(output_path, 16000, speech_np)
        LOGGER.info(f"🎧 Speech saved to: {output_path}")
        return output_path
    except Exception as e:
        LOGGER.error(f"TTS error: {e}")
        return None

def classify_query(query, query_model, query_tokenizer, id2label, device):
    inputs = query_tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=query_tokenizer.model_max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = query_model(**inputs)
        logits = outputs.logits
        predicted_class_id = int(torch.argmax(logits, dim=-1).cpu().numpy().item())
        intent = id2label[predicted_class_id]
        score = torch.softmax(logits, dim=-1)[0, predicted_class_id].item()
    print(f"Query: {query}")
    print(f"Predicted intent: {intent} (confidence: {score:.2f})")
    return intent


# ---------------------------
# DepthPro helpers (unchanged logic)
# ---------------------------
def compute_depth_map(image_path, device):
    image, _, f_px = load_rgb(image_path)
    img_pil = PIL.Image.open(image_path)
    img = np.array(img_pil)
    f_px = np.float64(19 * np.sqrt(img.shape[1]**2.0 + img.shape[0]**2.0) / np.sqrt(36**2 + 24**2))
    model, transform = create_model_and_transforms(
        device=device,
        precision=torch.half if device.type == "cuda" else torch.float32,
    )
    with torch.no_grad():
        pred = model.infer(transform(image).to(device), f_px=f_px)
    depth = pred["depth"].detach().cpu().numpy().squeeze()
    # free model memory if large
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass
    return depth

def pixel_to_camera_coords(u, v, depth, K):
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z], dtype=np.float32)

def segment_3d_distance(p1, p2, depth, K):
    line = np.linspace(p1, p2, num=20).astype(int)
    points3d = []
    for (y, x) in line:
        if 0<=y<depth.shape[0] and 0<=x<depth.shape[1]:
            z = depth[y, x]
            if z > 0:
                pt3d = pixel_to_camera_coords(x, y, z, K)
                points3d.append(pt3d)
    if len(points3d) < 2:
        return 0.0
    dists = [np.linalg.norm(points3d[i+1]-points3d[i]) for i in range(len(points3d)-1)]
    return float(np.sum(dists))

# ---------------------------
# A* + helpers (unchanged)
# ---------------------------
def heuristic(a,b): return math.hypot(b[0]-a[0], b[1]-a[1])

def astar(grid, start, goal, allow_diagonal=True):
    neighbors=[(-1,0),(1,0),(0,-1),(0,1)]
    if allow_diagonal: neighbors += [(-1,-1),(1,-1),(1,1),(-1,1)]
    close_set=set(); came_from={}; gscore={start:0}; fscore={start:heuristic(start,goal)}
    oheap=[(fscore[start],start)]
    max_x,max_y=grid.shape
    while oheap:
        _,current=heappop(oheap)
        if current==goal:
            path=[current]
            while current in came_from: current=came_from[current]; path.append(current)
            return path[::-1]
        close_set.add(current)
        for dx,dy in neighbors:
            nx,ny=current[0]+dx, current[1]+dy
            if nx<0 or nx>=max_x or ny<0 or ny>=max_y: continue
            if grid[nx,ny]==1: continue
            tentative_g=gscore[current]+math.hypot(dx,dy)
            if tentative_g < gscore.get((nx,ny),1e9):
                came_from[(nx,ny)]=current
                gscore[(nx,ny)]=tentative_g
                fscore[(nx,ny)]=tentative_g+heuristic((nx,ny),goal)
                heappush(oheap,(fscore[(nx,ny)],(nx,ny)))
    return []

def rdp(points, epsilon=2.0):
    if len(points)<3: return points
    (sx,sy),(ex,ey)=points[0],points[-1]
    max_dist,idx=0,-1
    for i in range(1,len(points)-1):
        px,py=points[i]
        num=abs((ey-sy)*px-(ex-sx)*py+ex*sy-ey*sx)
        den=sqrt((ey-sy)**2+(ex-sx)**2)
        dist=num/den if den!=0 else 0
        if dist>max_dist: max_dist,idx=dist,i
    if max_dist>epsilon:
        left=rdp(points[:idx+1],epsilon)
        right=rdp(points[idx:],epsilon)
        return left[:-1]+right
    else:
        return [points[0],points[-1]]

def inflate_walkable_mask_by_depth(walkable_mask, depth, fx, inflation_m):
    valid = depth[depth > 0]
    if valid.size == 0:
        median_z = 2.0
    else:
        median_z = float(np.median(valid))
    meters_per_pixel = median_z / float(fx) if fx != 0 else 0.01
    if meters_per_pixel <= 0:
        meters_per_pixel = 0.01
    inflate_px = int(ceil(inflation_m / meters_per_pixel))
    if inflate_px < 1 and inflation_m > 0:
        inflate_px = 1
    nonwalk = (walkable_mask == 0).astype(np.uint8) * 255
    if inflate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_px*2+1, inflate_px*2+1))
        dilated = cv2.dilate(nonwalk, kernel, iterations=1)
    else:
        dilated = nonwalk
    inflated_walkable = (dilated == 0).astype(np.uint8)
    LOGGER.info(f"Inflation: inflate_px={inflate_px}")
    return inflated_walkable

# ---------------------------
# Segmentation / Walkable mask
# ---------------------------
def get_walkable_mask(frame, device, segmentation_module, save_outputs=True):
    pil_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_data = to_tensor(pil_img)
    singleton_batch = {'img_data': img_data[None].to(device)}
    output_size = img_data.shape[1:]
    with torch.inference_mode():
        scores = segmentation_module(singleton_batch, segSize=output_size)
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy().astype(np.int32)

    walkable_mask = np.isin(pred, walkable_ids).astype(np.uint8)

    contours,_ = cv2.findContours(walkable_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(walkable_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_WALKABLE_AREA:
            cv2.drawContours(filtered,[cnt],-1,1,thickness=-1)

    if save_outputs:
        color_img = visualize_segmentation(pred)
        color_path = OUTPUT_DIR / "segmentation_color.png"
        cv2.imwrite(str(color_path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        overlay = (0.45 * color_img + 0.55 * cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype(np.uint8)
        overlay_path = OUTPUT_DIR / "segmentation_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("Generated walkable mask!!")

    return filtered, pred


def visual_guidance_fallback(best_fname, start_distance, end_distance):

    with open(str(best_fname), "rb") as img_file:
            b64_str = base64.b64encode(img_file.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{b64_str}"
    """
    Generate final guidance using Perplexity.
    Saves the final output to visual_guidance.txt.
    """
    final_guidance_text = None

    try:
        # ✅ Construct prompt for both models
        prompt = (
            "You are an assistive AI helping a blind user understand their surroundings.\n"
            "Do NOT describe visuals directly. Use spatial and directional language.\n\n"
            f"Start Distance (depth Z):{start_distance:.2f}\n\n"
            f"End Distance (depth Z): {end_distance:.2f} meters.\n\n"
            "Generate a short, clear, step-by-step navigation guidance to reach the target safely.\n"
            "Use concise sentences and practical instructions.\n"
            "Give the response in paragraph format and don't give it in numbering format\n"
        )

        # ✅ Try generating via Perplexity first
        if perplexity_client is not None:
            try:
                LOGGER.info("🧠 Generating visual guidance using Perplexity API...")
                completion = perplexity_client.chat.completions.create(
                    model="sonar",
                    messages=[
                        {"role": "system", "content": "You are an assistive navigation AI for the visually impaired. Do Not describe visuals directly. Use spatial and directional language.\n"},
                        {"role": "user", "content": [{"type": "text", "text":f"{prompt}"},
                        {"type": "image_url", "image_url": {"url": data_uri}}]}
                    ],
                )
                final_guidance_text = completion.choices[0].message.content.strip()
                LOGGER.info("✅ Perplexity response received successfully.")
            except Exception as e:
                LOGGER.warning(f"⚠️ Perplexity API failed: {e}")

        # ✅ Save output
        if final_guidance_text:
            final_guidance_text = final_guidance_text + " There is no walkable path found for the destination you want to reach. Please try to move the camera a little bit on your either sides, to increase the visibility of the camera."
            out_path = OUTPUT_DIR / "visual_guidance.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_guidance_text)
            LOGGER.info(f"📝 Guidance saved at {out_path}")
        else:
            LOGGER.error("❌ No guidance could be generated by either Perplexity or Phi-3.")

    except Exception as e:
        LOGGER.error(f"❌ Error during guidance generation: {e}")
        final_guidance_text = None

    return {
        "raw_instructions": None,
        "final_guidance": final_guidance_text,
        "scene_caption": None,
        "margin_points_info": None
    }


def no_target_fallback(captured_video, user_query, target_label, target_found):

    with open(captured_video, "rb") as video_file:
        b64_str = base64.b64encode(video_file.read()).decode("utf-8")
        # Find the actual MIME type (e.g. 'video/mp4')
        data_uri = f"data:video/mp4;base64,{b64_str}"

    """
    Generate final guidance using Perplexity.
    Saves the final output to no_target_guidance.txt.
    """
    final_guidance_text = None

    try:
        # ✅ Construct prompt for both models
        if target_found:
            prompt = (
                "You are an assistive AI helping a blind user understand their surroundings.\n"
                "Do NOT describe visuals directly. Use spatial and directional language.\n\n"
                f"User Query:{user_query}\n\n"
                f"Target Label:{target_label}\n\n"
                "Generate a short, clear, answer to help the user navigate safely.\n"
                "Also Tell the user that in which direction should he move the camera to place the target object in the centre of the frame.\n"
                "Use concise sentences and practical instructions.\n"
                "Give the response in paragraph format and don't give it in numbering format\n"
            )
        else:
            prompt = (
                "You are an assistive AI helping a blind user understand their surroundings.\n"
                "Do NOT describe visuals directly. Use spatial and directional language.\n\n"
                f"User Query:{user_query}\n\n"
                f"Target Label:{target_label}\n\n"
                "Generate a short, clear, answer to help the user navigate safely.\n"
                "Use concise sentences and practical instructions.\n"
                "Give the response in paragraph format and don't give it in numbering format\n"
            )

        # ✅ Try generating via Perplexity first
        if perplexity_client is not None:
            try:
                LOGGER.info("🧠 Generating visual guidance using Perplexity API...")
                completion = perplexity_client.chat.completions.create(
                    model="sonar",
                    messages=[
                        {"role": "system", "content": "You are an assistive navigation AI for the visually impaired. Do Not describe visuals directly. Use spatial and directional language.\n"},
                        {"role": "user", "content": [{"type": "text", "text":f"{prompt}"},
                        {"type": "video_url", "video_url": {"url": data_uri}}]}
                    ],
                )
                final_guidance_text = completion.choices[0].message.content.strip()
                LOGGER.info("✅ Perplexity response received successfully.")
            except Exception as e:
                LOGGER.warning(f"⚠️ Perplexity API failed: {e}")

        # ✅ Save output
        if final_guidance_text:
            if target_found:
                final_guidance_text = final_guidance_text + " There is no step by step guidance found for the destination you want to reach as target label was not detected in the centre of the frame. Please try to move the camera to increase the visibility of the camera."
            else:
                final_guidance_text = final_guidance_text + " There is no step by step guidance found for the destination you want to reach as target label was not detected"
            out_path = OUTPUT_DIR / "no_target_guidance.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_guidance_text)
            LOGGER.info(f"📝 Guidance saved at {out_path}")
        else:
            LOGGER.error("❌ No guidance could be generated by either Perplexity or Phi-3.")

    except Exception as e:
        LOGGER.error(f"❌ Error during guidance generation: {e}")
        final_guidance_text = None

    return {
        "raw_instructions": None,
        "final_guidance": final_guidance_text,
        "scene_caption": None,
        "margin_points_info": None
    }


# ---------------------------
# Main single-image pipeline (without LLaVA)
# ---------------------------
def process_single_image_with_caption_and_llm(image_path, device, target_label, user_query):
    """
    Note: LLaVA removed entirely.
    """
    segmentation_module = load_segmentation(device)
    final_path_fname = OUTPUT_DIR / "pp_per_obs_resize_final_path.jpg"
    best_fname = OUTPUT_DIR /"captured_target_frame_upscaled.jpg"
    # 1) YOLO detection
    yolo = YOLO("yolo12x.pt")
    frame = cv2.imread(str(image_path))
    if frame is None:
        LOGGER.error(f"Failed to read image: {image_path}")
        return None
    h, w, _ = frame.shape
    results = yolo(frame, imgsz=640, conf=DETECTION_CONF_THRESH, verbose = False)

    # find target bounding box closest to image center (if multiple detected)
    center_x, center_y = w / 2, h / 2
    best_target = None

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo.names.get(cls_id, str(cls_id)).lower()

            if target_label in label:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                obj_x = (x1 + x2) / 2
                obj_y = (y1 + y2) / 2
                dx = (obj_x - center_x) / w
                dy = (obj_y - center_y) / h
                offset_x = abs(dx)
                offset_y = abs(dy)
                print(f"Detected '{label}' | offset_x={offset_x:.3f}, offset_y={offset_y:.3f}")

                if offset_x < 0.1 and offset_y < 0.1:
                    best_target = {
                        'label': label,
                        'bbox': (float(x1), float(y1), float(x2), float(y2))
                    }

    if not best_target:
        LOGGER.warning("Target not detected by YOLO.")
        return None

    target = best_target
    print(f"✅ Selected most centered target '{target['label']}")

    # No scene_caption (LLaVA removed). We'll rely on final path image + raw instructions for Perplexity.
    scene_caption = None

    # 3) Run semantic segmentation (walkable mask) and DepthPro depth
    start_orig = (h - 10, w // 2)
    print("Generating walkable mask.........")
    walkable_mask, pred = get_walkable_mask(frame, device, segmentation_module, save_outputs=True)
    depth = compute_depth_map(image_path, device)

    # inflate walkable
    inflated_walkable = inflate_walkable_mask_by_depth(walkable_mask, depth, CAMERA_INTRINSICS['fx'], INFLATION_RADIUS_M)

    walk_vis = cv2.cvtColor(walkable_mask * 255, cv2.COLOR_GRAY2BGR)
    cv2.circle(walk_vis, (start_orig[1], start_orig[0]), 6, (255, 0, 0), -1)

    x1, y1, x2, y2 = target['bbox']
    end_candidate = (int(y2), int((x1 + x2) // 2))  # (row, col)

    def nearest_walkable(mask, pt):
        r, c = pt
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c] == 1:
            return (r, c)
        coords = np.argwhere(mask == 1)
        if coords.size == 0:
            return pt
        dists = np.linalg.norm(coords - np.array([r, c]), axis=1)
        idx = np.argmin(dists)
        nearest = tuple(coords[idx])
        return (int(nearest[0]), int(nearest[1]))
    print("Finding walkable path........")
    end_orig = nearest_walkable(walkable_mask, end_candidate)
    end_inflated = nearest_walkable(inflated_walkable, end_candidate)
    start_inflated = nearest_walkable(inflated_walkable, start_orig)

    cv2.circle(walk_vis, (end_orig[1], end_orig[0]), 6, (0, 255, 0), -1)
    cv2.circle(walk_vis, (end_inflated[1], end_inflated[0]), 6, (0, 0, 255), -1)
    cv2.circle(walk_vis, (start_inflated[1], start_inflated[0]), 6, (255, 255, 0), -1)
    cv2.imwrite(str(OUTPUT_DIR / "walkable_mask_with_endpoints_dbg.jpg"), walk_vis)

    # A* on inflated mask
    grid_infl = (1 - inflated_walkable).astype(np.uint8)

    def pix_to_grid(pt, grid):
        r, c = pt
        gr = max(0, min(grid.shape[0] - 1, int(r)))
        gc = max(0, min(grid.shape[1] - 1, int(c)))
        return (gr, gc)

    start_grid_infl = pix_to_grid(start_inflated, grid_infl)
    end_grid_infl = pix_to_grid(end_inflated, grid_infl)
    raw_path_infl = astar(grid_infl, start_grid_infl, end_grid_infl, allow_diagonal=True)

    start_distance = float(depth[h-10, w//2]) if 0<=h-10<depth.shape[0] and 0<=w//2<depth.shape[1] else 0.0
    print(f"Start Distance (depth Z) is: {start_distance:.2f} m")
    x1b, y1b, x2b, y2b = target['bbox']
    end_distance = float(depth[int(y2b), int((x1b+x2b)//2)]) if 0<=int(y2b)<depth.shape[0] else 0.0
    print(f"End Distance (depth Z) is: {end_distance:.2f} m")

    if not raw_path_infl:
        print("No walkable path found, fallback to visual guidance")
        # create a minimal textual guidance request using Perplexity and the (nonexistent) final path image
        final_guidance_text = visual_guidance_fallback(best_fname, start_distance, end_distance)
        print(final_guidance_text)
        return None

    full_raw_path_nodes = list(raw_path_infl)

    # Bridging for end
    final_infl_rowcol = full_raw_path_nodes[-1]
    if final_infl_rowcol != end_orig:
        LOGGER.info("Inflated endpoint differs from original endpoint. Computing bridging path on original mask.")
        grid_orig = (1 - walkable_mask).astype(np.uint8)
        final_infl_grid_node = (max(0, min(grid_orig.shape[0]-1, int(final_infl_rowcol[0]))),
                                max(0, min(grid_orig.shape[1]-1, int(final_infl_rowcol[1]))))
        end_orig_grid_node = (max(0, min(grid_orig.shape[0]-1, int(end_orig[0]))),
                              max(0, min(grid_orig.shape[1]-1, int(end_orig[1]))))
        raw_bridge_end = astar(grid_orig, final_infl_grid_node, end_orig_grid_node, allow_diagonal=True)
        if raw_bridge_end:
            if raw_bridge_end[0] == full_raw_path_nodes[-1]:
                full_raw_path_nodes += raw_bridge_end[1:]
            else:
                full_raw_path_nodes += raw_bridge_end

    # Bridging for start
    if start_inflated != start_orig:
        LOGGER.info("Original start point not in inflated mask. Computing bridging path on original mask.")
        grid_orig = (1 - walkable_mask).astype(np.uint8)
        start_infl_grid_node = (max(0, min(grid_orig.shape[0]-1, int(start_inflated[0]))),
                                max(0, min(grid_orig.shape[1]-1, int(start_inflated[1]))))
        start_orig_grid_node = (max(0, min(grid_orig.shape[0]-1, int(start_orig[0]))),
                                max(0, min(grid_orig.shape[1]-1, int(start_orig[1]))))
        raw_bridge_start = astar(grid_orig, start_orig_grid_node, start_infl_grid_node, allow_diagonal=True)
        if raw_bridge_start:
            full_raw_path_nodes = raw_bridge_start + full_raw_path_nodes

    # Convert path to pixel points (x,y) for drawing / simplification
    combined_pix_points = [(int(node[1]), int(node[0])) for node in full_raw_path_nodes]
    simp_pts = rdp(combined_pix_points, epsilon=20.0)
    if len(simp_pts) < 2:
        LOGGER.warning("Simplified path too short.")
        return None

    # Build instructions & detect nearby obstacles along the simplified path
    instructions = []
    prev_angle = 270.0
    margin_points_info = []
    for i in range(1, len(simp_pts)):
        (x1p, y1p), (x2p, y2p) = simp_pts[i - 1], simp_pts[i]
        dx, dy = (y2p - y1p), (x2p - x1p)
        angle = math.degrees(math.atan2(dx, dy))
        turn = angle - prev_angle
        while turn > 180: turn -= 360
        while turn < -180: turn += 360
        if abs(turn) > 10:
            instructions.append(f"Turn {abs(turn):.0f}° {'right' if turn > 0 else 'left'}.")
        # compute 3D distance for this segment
        dist = segment_3d_distance((y1p, x1p), (y2p, x2p), depth, CAMERA_INTRINSICS)
        steps = math.ceil(dist / 0.7)
        instructions.append(f"Walk forward {dist:.2f} meters ({steps} steps).")

        # instructions.append(f"Walk forward {dist:.2f} meters.")
        prev_angle = angle

        # search patch in original pred map for nearby obstacles around margin point
        margin_pt = (y2p, x2p)
        search_radius = 50
        y0, x0 = margin_pt
        y_min, y_max = max(0, y0-search_radius), min(pred.shape[0], y0+search_radius+1)
        x_min, x_max = max(0, x0-search_radius), min(pred.shape[1], x0+search_radius+1)
        patch = pred[y_min:y_max, x_min:x_max]

        largest_area = 0.0
        sel_class = None
        sel_centroid = None
        distance_thresh = 50

        for cls in np.unique(patch):
            if cls in walkable_ids or cls < 0: continue
            mask = (patch == cls).astype(np.uint8) * 255
            if mask.sum() == 0: continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area <= 0: continue
                d = cv2.pointPolygonTest(cnt, (x2p - x_min, y2p - y_min), True)
                min_dist = abs(d)
                if min_dist < distance_thresh and area > largest_area:
                    largest_area = area
                    sel_class = int(cls)
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx_patch = int(M['m10'] / M['m00'])
                        cy_patch = int(M['m01'] / M['m00'])
                        cx = cx_patch + x_min
                        cy = cy_patch + y_min
                        sel_centroid = (cy, cx)

        if sel_class is not None and sel_centroid is not None:
            class_name = names.get(sel_class+1, f"class_{sel_class}")
            dist_to_obj = segment_3d_distance((y2p, x2p), (sel_centroid[0], sel_centroid[1]), depth, CAMERA_INTRINSICS)
            margin_points_info.append(( (x2p, y2p), class_name, sel_centroid, dist_to_obj ))

    # Save visuals (final path image expected to be used for Perplexity)
    walk_vis_final = cv2.cvtColor(walkable_mask * 255, cv2.COLOR_GRAY2BGR)
    cv2.circle(walk_vis_final, (start_orig[1], start_orig[0]), 6, (255, 0, 0), -1)
    cv2.circle(walk_vis_final, (end_orig[1], end_orig[0]), 6, (0, 255, 0), -1)
    for i in range(1, len(simp_pts)):
        cv2.line(walk_vis_final, simp_pts[i - 1], simp_pts[i], (255, 0, 255), 2)
    cv2.imwrite(str(OUTPUT_DIR / "walkable_path.jpg"), walk_vis_final)

    final = frame.copy()
    for i in range(1, len(simp_pts)):
        cv2.line(final, simp_pts[i - 1], simp_pts[i], (255, 0, 255), 2)
        mid = ((simp_pts[i - 1][0] + simp_pts[i][0]) // 2, (simp_pts[i - 1][1] + simp_pts[i][1]) // 2)
        dist_seg = segment_3d_distance((simp_pts[i - 1][1], simp_pts[i - 1][0]), (simp_pts[i][1], simp_pts[i][0]), depth, CAMERA_INTRINSICS)
        cv2.putText(final, f"{dist_seg:.2f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.circle(final, (start_orig[1], start_orig[0]), 6, (255, 0, 0), -1)
    cv2.circle(final, (end_orig[1], end_orig[0]), 6, (0, 255, 0), -1)
    final_path_fname = OUTPUT_DIR / "pp_per_obs_resize_final_path.jpg"
    cv2.imwrite(str(final_path_fname), final)

    print("✅ Finished path planning. Outputs saved in ./output")

    # 4) Generate final guidance using Perplexity Sonar API using the final path image + text
    final_guidance_text = None
    try:
        # Build textual instruction string
        # merged_scene_text = ""  # no LLaVA scene; rely on image
        # joined_instructions = "\n".join(instructions)
        # print("\n--- RAW INSTRUCTIONS (distance format) ---")
        # for ins in instructions:
        #     print(ins)

        # print("\n--- RAW INSTRUCTIONS (steps format) ---")
        # for ins in instructions:
        #     if "Walk forward" in ins:
        #         # extract the distance value from the instruction
        #         try:
        #             dist_str = ins.split("Walk forward")[1].split("meters")[0].strip().split()[0]
        #             dist_val = float(dist_str)
        #             steps = math.ceil(dist_val / 0.7)
        #             print(ins.replace(f"{dist_val:.2f} meters", f"{steps} steps"))
        #         except Exception:
        #             print(ins)
        #     else:
        #         print(ins)
        # Print both formats (for debugging)
        print("\n--- RAW INSTRUCTIONS (distance format) ---")
        for ins in instructions:
            print(ins)

        print("\n--- RAW INSTRUCTIONS (steps format) ---")
        step_instructions = []  # this list will contain only step-based instructions
        for ins in instructions:
            if "Walk forward" in ins:
                try:
                    dist_str = ins.split("Walk forward")[1].split("meters")[0].strip().split()[0]
                    dist_val = float(dist_str)
                    steps = math.ceil(dist_val / 0.7)
                    step_ins = ins.replace(f"{dist_val:.2f} meters", f"{steps} steps")
                    step_instructions.append(step_ins)
                    print(step_ins)
                except Exception:
                    step_instructions.append(ins)
                    print(ins)
            else:
                step_instructions.append(ins)
                print(ins)

        # Now feed only the step-based ones to Perplexity
        joined_instructions = "\n".join(step_instructions)
        # Create data URI of final path image
        with open(str(final_path_fname), "rb") as img_file:
            b64_str = base64.b64encode(img_file.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{b64_str}"
        start_dist_step = math.ceil(start_distance / 0.7)
        end_dist_step = math.ceil(end_distance / 0.7)
        # Construct messages according to Perplexity Sonar image+text payload
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistive AI helping a blind user navigate to a target "
                    "who cannot see anything around him, so do not give any visual instructions."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":
                    f"Target label: {target_label}\n\n"
                    f"User Query: {user_query}\n\n"
                    f"Raw path instructions (step-by-step):\n"
                    f"Start Distance (depth Z) is: {start_dist_step:.2f} m\n"
                    f"End Distance (depth Z) is: {end_dist_step:.2f} m\n\n"
                    "--- RAW INSTRUCTIONS ---\n"
                    f"{joined_instructions}\n\n"
                    "Requirements for the final guidance:\n"
                    "- Use given final path image and path instructions to give a single clear step-by-step guidance "
                    "where you should use the scene image to understand where the user reaches after every instruction.\n"
                    "- For each path segment / margin point, warn the user of the nearby obstacles if any and "
                    "use the image to identify them.\n"
                    f"Also the first instruction should always be to reach the starting point, which should be walk straight {start_dist_step:.2f} steps, this should be in steps only, however do not mention the name (starting point) explicitly, after this the rest of the instructions should follow. Also since we do not know about the obstacles between the user and the start point so tell the user to be aware of obstacles.\n"
                    "- Use short simple sentences, numbered steps, and avoid technical jargon. Keep instructions concise and actionable. Also when you are giving steps in the instructions, give them in words, not digits. Do not use meters in any instructions" 
                    "Give the response in paragraph format and don't give it in numbering format\n"
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ],
            },
        ]

        if perplexity_client is None:
            raise RuntimeError("Perplexity client not initialized")

        print("🧠 Generating guidance using Perplexity Sonar API (image + text)...")
        completion = perplexity_client.chat.completions.create(
            messages=messages,
            model="sonar"
        )
        final_guidance_text = completion.choices[0].message.content.strip()
        print("✅ Perplexity Sonar response received.")

        if final_guidance_text:
            with open(OUTPUT_DIR / "final_guidance.txt", "w", encoding="utf-8") as f:
                f.write(final_guidance_text)
            LOGGER.info("📝 Final guidance saved to file.")
    except Exception as e:
        LOGGER.error(f"Perplexity final guidance error: {e}")
        final_guidance_text = None

    # cleanup
    try:
        del segmentation_module
        del walkable_mask, inflated_walkable, pred, depth
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    # return same dict structure
    return {
        "raw_instructions": instructions,
        "final_guidance": final_guidance_text,
        "scene_caption": None,
        "margin_points_info": margin_points_info
    }

# ---------------------------
# Video scanning helper (unchanged signature & logic)
# ---------------------------
def process_video_and_select_frame(video_path, device, target_label, user_query):
    """
    Continuously reads frames from webcam (/dev/video42) until the target object
    is detected and roughly centered (within ±10% of frame center).
    If no target object is centered within 10 seconds, saves that 10s video clip
    and calls no_target_fallback().
    """
    CENTER_TOLERANCE = 0.1   # ±10% tolerance
    TIMEOUT = 15             # seconds
    OUTPUT_CLIP = "no_target_fallback_clip.mp4"

    yolo = YOLO("yolo12x.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        LOGGER.error(f"Cannot open video: {video_path}")
        return {"error": f"Cannot open video: {video_path}"}

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    center_x, center_y = frame_w / 2, frame_h / 2

    print("🎥 Starting live feed from webcam... Press Ctrl+C to stop.")
    found_frame = None
    start_time = time.time()

    # For saving fallback clip
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_CLIP, fourcc, fps, (frame_w, frame_h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                LOGGER.warning("No frame read from webcam. Retrying...")
                continue

            out.write(frame)  # record frame for possible fallback
            target_found = False
            results = yolo(frame, imgsz=640, conf=DETECTION_CONF_THRESH, verbose = False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo.names.get(cls_id, str(cls_id)).lower()

                    if target_label in label:
                        target_found = True
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        obj_x = (x1 + x2) / 2
                        obj_y = (y1 + y2) / 2

                        dx = (obj_x - center_x) / frame_w
                        dy = (obj_y - center_y) / frame_h

                        offset_x = abs(dx)
                        offset_y = abs(dy)

                        # print(f"Detected '{label}' | offset_x={offset_x:.3f}, offset_y={offset_y:.3f}")
                        print(f"Detected '{label}'")
                        if offset_x <= CENTER_TOLERANCE and offset_y <= CENTER_TOLERANCE:
                            print("✅ Target object is centered! Capturing frame...")
                            found_frame = frame.copy()
                            raise StopIteration

            # Check timeout
            if time.time() - start_time > TIMEOUT:
                print(f"⏱️ No target centered within {TIMEOUT}s. Saving fallback video...")
                raise TimeoutError

    except StopIteration:
        pass
    except TimeoutError:
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        LOGGER.warning(f"No target detected in center within {TIMEOUT}s.")
        return no_target_fallback(OUTPUT_CLIP, user_query, target_label, target_found)
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
    finally:
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    if found_frame is None:
        msg = f"Target '{target_label}' not centered before stopping."
        LOGGER.warning(msg)
        return {"error": msg}

    # Save the captured frame
    captured_frame_path = Path(f"{OUTPUT_DIR}/captured_target_frame_upscaled.jpg")
    h, w = found_frame.shape[:2]
    new_w = 768
    scale = new_w / w
    new_h = int(h * scale)
    upscaled_frame = cv2.resize(found_frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(captured_frame_path), upscaled_frame)

    LOGGER.info(f"✅ Saved best centered frame: {captured_frame_path}")

    return process_single_image_with_caption_and_llm(str(captured_frame_path), device, target_label, user_query)





# def process_video_and_select_frame(video_path, device, target_label):
#     """
#     Continuously reads frames from webcam (/dev/video42) until the target object
#     is detected and roughly centered (horizontal offset within 4–6% of frame center).
#     Saves that frame and passes it to the next stage of the pipeline.
#     """
#     CENTER_TOLERANCE = 0.05  # ±5% tolerance around center
#     yolo = YOLO("yolov9e.pt")
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         LOGGER.error(f"Cannot open video: {video_path}")
#         return {"error": f"Cannot open video: {video_path}"}

#     frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     center_x, center_y = frame_w / 2, frame_h / 2

#     print("🎥 Starting live feed from webcam... Press Ctrl+C to stop.")
#     found_frame = None

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 LOGGER.warning("No frame read from webcam. Retrying...")
#                 continue

#             results = yolo(frame, imgsz=640, conf=DETECTION_CONF_THRESH)
#             for r in results:
#                 for box in r.boxes:
#                     cls_id = int(box.cls[0])
#                     label = yolo.names.get(cls_id, str(cls_id)).lower()

#                     if target_label in label:
#                         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                         obj_x = (x1 + x2) / 2
#                         obj_y = (y1 + y2) / 2

#                         dx = (obj_x - center_x) / frame_w
#                         dy = (obj_y - center_y) / frame_h

#                         # Object center offset from image center
#                         offset_x = abs(dx)
#                         offset_y = abs(dy)

#                         print(f"Detected '{label}' | offset_x={offset_x:.3f}, offset_y={offset_y:.3f}")

#                         # If horizontally centered within 4–6% of width
#                         if 0 <= offset_x <= 0.1 and 0 <= offset_y <= 0.1:
#                             print("✅ Target object is centered! Capturing frame...")
#                             found_frame = frame.copy()
#                             raise StopIteration  # exit nested loops safely

#             # Optional: show live feed for debugging
#             # cv2.imshow("Live Feed", frame)
#             # if cv2.waitKey(1) & 0xFF == ord('q'):
#             #     break

#     except StopIteration:
#         pass
#     except KeyboardInterrupt:
#         print("🛑 Interrupted by user.")
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()

#     if found_frame is None:
#         msg = f"Target '{target_label}' not centered before stopping."
#         LOGGER.warning(msg)
#         return {"error": msg}

#     # Save the captured frame
#     captured_frame_path = Path("captured_target_frame_upscaled.jpg")
#     h, w = found_frame.shape[:2]
#     new_w = 768
#     scale = new_w / w
#     new_h = int(h * scale)
#     upscaled_frame = cv2.resize(found_frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
#     cv2.imwrite(str(captured_frame_path), upscaled_frame)

#     LOGGER.info(f"✅ Saved best centered frame: {captured_frame_path}")

#     # Continue the pipeline
#     return process_single_image_with_caption_and_llm(str(captured_frame_path), device, None, target_label)

# ---------------------------
# Utility: parse target label from user query
# ---------------------------
# def extract_target_label_from_query(user_query):
#     """
#     Attempt to extract target label from common spoken queries.
#     """
#     if not user_query:
#         return ""
#     q = user_query.strip().lower()
#     q = q.strip(string.punctuation + " ")
#     if " find " in f" {q} ":
#         idx = q.find(" find ")
#         remainder = q[idx + len(" find "):].strip()
#         for art in ("the ", "a ", "an "):
#             if remainder.startswith(art):
#                 remainder = remainder[len(art):]
#                 break
#         return remainder
#     tokens = q.split()
#     if tokens and tokens[0] == "find":
#         remainder = " ".join(tokens[1:]).strip()
#         return remainder
#     if q.startswith("help me find "):
#         remainder = q[len("help me find "):].strip()
#         return remainder
#     if len(tokens) >= 2:
#         return " ".join(tokens[-2:])
#     return tokens[-1] if tokens else ""


def extract_target_object(user_query):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(user_query)
    for chunk in doc.noun_chunks:
        # Filter for noun chunks with 'obj' or 'pobj' (object of verb or preposition)
        if any(tok.dep_ in ('dobj', 'pobj', 'attr', 'nsubj', 'ROOT') for tok in chunk):
            nouns = [tok for tok in chunk if tok.pos_ == 'NOUN']
            if nouns:
                # Return the last noun in the chunk, which is usually the target object
                return nouns[-1].lemma_
    # fallback: first noun in sentence
    for tok in doc:
        if tok.pos_ == 'NOUN':
            return tok.lemma_
    return None

import cv2
import time
from pathlib import Path

def capture_video_feed(video_path, duration=5, fps=30):
    cap = cv2.VideoCapture(VIDEO_PATH)  # Use 0 for default webcam

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return {"error": f"Cannot open video: {video_path}"}

    # Set video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))

    print(f"🎥 Recording video for {duration} seconds... Press Ctrl+C to stop early.")
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame read from webcam. Retrying...")
                continue

            # Write the frame to the video file
            out.write(frame)
            if time.time() - start_time >= duration:
                break

    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"✅ Video saved to {video_path}")
    return {"success": f"Video saved to {video_path}"}


def scene_description_pipeline(user_query):

    video_path = Path("captured_video_feed.mp4")
    capture_video_feed(video_path, duration=5)
    with open(video_path, "rb") as video_file:
        b64_str = base64.b64encode(video_file.read()).decode("utf-8")
        # Find the actual MIME type (e.g. 'video/mp4')
        data_uri = f"data:video/mp4;base64,{b64_str}"

    messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistive AI helping a blind user with a query which can be solved by looking at the detailed scene description of a given video "
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":
                    "Requirements for the scene description:\n"
                    "- Use given final video to give detailed and thorough scene description "
                    "Answer the below prompt using the scene description above"
                    f"query:{user_query}"
                    "Give the response in paragraph format and don't give it in numbering format"
                    },
                    {"type": "video_url", "video_url": {"url": data_uri}}
                ],
            },
        ]

    if perplexity_client is not None:
      try:
          LOGGER.info("🧠 Generating scene description using Perplexity Sonar API...")
          completion = perplexity_client.chat.completions.create(
              messages=messages,
              model="sonar"
          )
          scene_description = completion.choices[0].message.content.strip()
          LOGGER.info("✅ Perplexity Sonar response received.")
      except Exception as e:
          scene_description = "ERROR10001"
          LOGGER.error(f"⚠️ Perplexity API failed: {e}.")

    if scene_description != "ERROR10001":
        with open(OUTPUT_DIR / "scene_description.txt", "w", encoding="utf-8") as f:
            f.write(scene_description)
        print("📝 Final description saved to file.")
    else:
        LOGGER.error("❌ No description text could be generated by Perplexity")

    return scene_description

def process_general_query(user_query):
  messages = [
      {
          "role": "system",
          "content": (
              "You are an assistive AI helping a user with some general queries "
          ),
      },
      {
          "role": "user",
          "content": (f"{user_query}\n\n"
           "Give the response in paragraph format and don't give it in numbering format"
          ),
      },
  ]
  if perplexity_client is not None:
      try:
          LOGGER.info("🧠 Generating guidance using Perplexity Sonar API...")
          completion = perplexity_client.chat.completions.create(
              messages=messages,
              model="sonar"
          )
          query_response = completion.choices[0].message.content.strip()
          LOGGER.info("✅ Perplexity Sonar response received.")
      except Exception as e:
          query_response = "ERROR10002"
          LOGGER.warning(f"⚠️ Perplexity API failed: {e}.")

  if query_response != "ERROR10002":
    with open(OUTPUT_DIR / "general_response.txt", "w", encoding="utf-8") as f:
        f.write(query_response)
    print("📝 Final reponse saved to file.")
  else:
    LOGGER.error("❌ No guidance text could be generated by Perplexity")

  return query_response


def main_pipeline(device, intent, user_query):
    
    if intent == "scene-dependent":
        result = scene_description_pipeline(user_query)
        # Handle errors and TTS output
        if result == "ERROR10001":
            LOGGER.error("Pipeline error: Perplexity API failed")
            synthesize_speech(tts_model, tts_processor, "Pipeline error: Perplexity API failed", "./output/result.wav", device)
        elif result is None:
            msg = "Could not generate guidance for the requested target."
            LOGGER.error(msg)
            synthesize_speech(tts_model, tts_processor, msg, "./output/result.wav", device)
        else:
            print("\n---------------SCENE DESCRIPTION-----------------------\n")
            print(result)
            synthesize_speech(tts_model, tts_processor, result, "./output/result.wav", device)
    else:
        if intent == "path-finding":
            # Extract target label
            target_label = extract_target_object(user_query)
            target_label = target_label.strip().lower()
            if not target_label:
                LOGGER.error("Could not extract a target label from the query. Exiting.")
                exit(1)
            LOGGER.info(f"Extracted target label: '{target_label}'")
            result = process_video_and_select_frame(VIDEO_PATH, device, target_label, user_query)
            # Handle errors and TTS output
            if isinstance(result, dict) and "error" in result:
                LOGGER.error(f"Pipeline error: {result['error']}")
                synthesize_speech(tts_model, tts_processor, result['error'], "./output/result.wav", device)
            elif result is None:
                msg = "Could not generate guidance for the requested target."
                LOGGER.error(msg)
                synthesize_speech(tts_model, tts_processor, msg, "./output/result.wav", device)
            else:
                final_guidance = result.get("final_guidance") or "No final guidance generated."
                print("\n--- FINAL GUIDANCE ---\n")
                print(final_guidance)
                synthesize_speech(tts_model, tts_processor, final_guidance, "./output/result.wav", device)
        else:
            result = process_general_query(user_query)
            # Handle errors and TTS output
            if result == "ERROR10002":
                LOGGER.error("Pipeline error: Perplexity API failed")
                synthesize_speech(tts_model, tts_processor, "Pipeline error: Perplexity API failed", "./output/result.wav", device)
            elif result is None:
                msg = "Could not generate response"
                LOGGER.error(msg)
                synthesize_speech(tts_model, tts_processor, msg, "./output/result.wav", device)
            else:
                print("\n---------------GENERAL RESPONSE-----------------------\n")
                print(result)
                synthesize_speech(tts_model, tts_processor, result, "./output/result.wav", device)

    return result
# ---------------------------
# Main (STT -> pipeline -> TTS)
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    LOGGER.info(f"Using device: {device}")

    # Load STT & TTS
    asr_model, asr_processor, tts_model, tts_processor = load_stt_tts(device)

    print("Listening for query (from audio file)...")
    user_query = transcribe_audio(asr_model, asr_processor, AUDIO_PATH, device)
    if not user_query:
        LOGGER.error("No transcription obtained. Exiting.")
        exit(1)

    user_query = user_query.strip()
    user_query = user_query.rstrip(string.punctuation + " ")
    LOGGER.info(f"User query: {user_query}")

    query_model, query_tokenizer, id2label = load_bert(device)
    intent = classify_query(user_query, query_model, query_tokenizer, id2label, device)

    result = main_pipeline(device, intent, user_query)

    LOGGER.info("Done.")
