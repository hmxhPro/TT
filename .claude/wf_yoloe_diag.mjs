export const meta = {
  name: 'yoloe-train-detect-diagnosis',
  description: 'Adversarially verify root causes of YOLOE low-mAP + zero-detection, with empirical inference test, and produce the fix',
  phases: [{ title: 'Verify', detail: '4 independent agents: refute trainer hypothesis, empirical inference test, quantify small-object, design fix' }],
}

phase('Verify')

const ENV_PY = '/home/hmxh/miniconda3/envs/SOD/bin/python'
const BACKEND = '/home/hmxh/workspace/sodv3/SOD/backend'
const TRAINED = BACKEND + '/runs/train/汽车检测_v1_f190a05c/weights/best.pt'
const VAL_DIR = BACKEND + '/datasets/312b585e-7785-43a4-9f10-94456a894ae1/yolo/f190a05c-b7b6-4616-a4c4-0068327bde03/images/val'
const LABEL_DIR = BACKEND + '/datasets/312b585e-7785-43a4-9f10-94456a894ae1/yolo/f190a05c-b7b6-4616-a4c4-0068327bde03/labels/train'
const BASE = BACKEND + '/models/yolo/yoloe-11l-seg.pt'
const UL = '/home/hmxh/miniconda3/envs/SOD/lib/python3.10/site-packages/ultralytics'

const FACTS = [
  'GROUND-TRUTH FACTS (already verified by the lead — treat as given, do not re-derive):',
  '- ultralytics 8.4.58 in conda env "SOD" (python at ' + ENV_PY + '). YOLOE imports fine (no fallback to YOLO).',
  '- The project trains via YOLOWorld/train_yoloe.py: model = YOLOE("yoloe-11l-seg.pt"); model.train(data=..., epochs, imgsz=640, batch, device, workers, patience). It passes NO trainer= kwarg and no freeze.',
  '- Because the checkpoint is a -seg model, YOLOE.task_map routes task=segment -> trainer = ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer (bases: YOLOETrainer, SegmentationTrainer).',
  '- YOLOESegTrainer.get_model builds YOLOESegModel with nc=min(data_nc, 80) and a source comment: "this nc is the max number of different text samples in one image, not the actual nc". build_dataset (inherited from YOLOETrainer) calls build_yolo_dataset(..., multi_modal=(mode==train)). It does NOT call get_text_pe / set_classes / fuse.',
  '- The alternative trainer YOLOEPESegTrainer (and YOLOEPETrainer for detect) DOES, in get_model: del savpe; load weights; names=list(data names values); tpe=model.get_text_pe(names); model.set_classes(names, tpe); model.model[-1].fuse(model.pe); then sets cv3[*][2].requires_grad_(True) (linear-probe of the cls projection). This is the documented custom-data fine-tune path.',
  '- Training run 汽车检测_v1_f190a05c: task=segment, model=yoloe-11l-seg.pt, epochs=100 (ran 55), batch=8, imgsz=640, single class names={0: 汽车检测}. results.csv: mAP50(B) was 0.1398 at epoch1 and 0.124 at epoch55 (FLAT). train/cls loss 7.66 -> 4.82 (still huge; normal YOLO converges below 1). train/box 1.46 -> 1.31 (barely moved). Dataset: 4332 train / 1083 val images; labels are segmentation polygons (mostly 4-point rectangles); objects are tiny (relative widths ~0.016-0.05).',
  '- Inference: app/services/image_detector.py detect_with_model() loads best.pt via YOLOE() and predicts WITHOUT set_classes (uses baked-in model.names); default conf = IMAGE_DETECT_CONF = 0.25. The video path yoloe_detector.py always calls set_classes(prompt) (zero-shot) and uses the BASE weights.',
  'ultralytics source tree: ' + UL,
].join('\n')

const TRAINER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['confirmed', 'confidence', 'mechanism', 'refutation_attempt', 'evidence', 'official_recommendation'],
  properties: {
    confirmed: { type: 'boolean', description: 'true if using default YOLOESegTrainer instead of YOLOEPESegTrainer is genuinely a primary cause of cls-loss-stuck + flat mAP for a fixed single-class custom dataset' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    mechanism: { type: 'string', description: 'Precise mechanism citing ultralytics source: what the multi_modal=True path feeds the cls/text branch, and why a plain single-class YOLO dataset (no per-object grounding text) fails to supervise it.' },
    refutation_attempt: { type: 'string', description: 'Honest attempt to REFUTE: is there any way the default trainer DOES learn single-class? What does the multi_modal dataset do when the yaml only has names and no grounding texts?' },
    evidence: { type: 'array', items: { type: 'string' }, description: 'Concrete code/doc citations (file:func/line or doc URL + quote)' },
    official_recommendation: { type: 'string', description: 'What official Ultralytics YOLOE docs prescribe for fine-tuning a custom fixed-class dataset (trainer + freeze). Quote/cite.' },
  },
}

const INFER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ran_ok', 'trained_names', 'trained_task', 'trained_dets_conf001', 'trained_maxconf', 'trained_dets_conf025', 'base_setclasses_dets', 'base_setclasses_maxconf', 'observations', 'raw_notes'],
  properties: {
    ran_ok: { type: 'boolean' },
    trained_names: { type: 'string', description: 'repr of trained model.names' },
    trained_task: { type: 'string' },
    trained_dets_conf001: { type: 'integer', description: 'total detections across sampled val images from TRAINED best.pt at conf=0.001 (no set_classes)' },
    trained_maxconf: { type: 'number', description: 'max box confidence from TRAINED model across sampled images at conf=0.001 (-1 if none)' },
    trained_dets_conf025: { type: 'integer', description: 'total detections from TRAINED best.pt at conf=0.25' },
    base_setclasses_dets: { type: 'integer', description: 'total detections from BASE yoloe-11l-seg.pt after set_classes(["car"]) at conf=0.10 on same images' },
    base_setclasses_maxconf: { type: 'number' },
    observations: { type: 'array', items: { type: 'string' } },
    raw_notes: { type: 'string', description: 'Any errors, OOM, device fallbacks, the actual python you ran' },
  },
}

const SMALL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['median_rel_w', 'median_rel_h', 'p90_rel_w', 'objs_per_image', 'est_source_res', 'est_obj_px_at_640', 'severity', 'findings'],
  properties: {
    median_rel_w: { type: 'number' },
    median_rel_h: { type: 'number' },
    p90_rel_w: { type: 'number' },
    objs_per_image: { type: 'number', description: 'mean objects per labeled image' },
    est_source_res: { type: 'string', description: 'typical raw image resolution WxH from sampling' },
    est_obj_px_at_640: { type: 'string', description: 'estimated median object size in pixels after letterbox to imgsz=640' },
    severity: { type: 'string', enum: ['dominant', 'significant', 'minor'], description: 'how much small-object+imgsz640 alone caps mAP if the trainer bug were fixed' },
    findings: { type: 'array', items: { type: 'string' } },
  },
}

const FIX_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['corrected_run_training_snippet', 'key_changes', 'recommended_hyperparams', 'inference_guidance', 'caveats'],
  properties: {
    corrected_run_training_snippet: { type: 'string', description: 'Drop-in corrected model-build + train() portion of run_training() in train_yoloe.py using YOLOEPESegTrainer (seg) / YOLOEPETrainer (detect). Valid python for ultralytics 8.4.58, keeping existing kwargs.' },
    key_changes: { type: 'array', items: { type: 'string' } },
    recommended_hyperparams: { type: 'object', description: 'Overrides for single-class tiny-object dataset (imgsz, epochs, batch, lr0, optimizer, close_mosaic, mosaic, scale, imgsz~1280/tiling note) each with one-line why', additionalProperties: true },
    inference_guidance: { type: 'array', items: { type: 'string' }, description: 'Verify/use retrained weights: detect path, conf, SAHI; base vs model_id.' },
    caveats: { type: 'array', items: { type: 'string' } },
  },
}

const trainerPrompt = [
  'You are adversarially verifying a root-cause claim about YOLOE training in ultralytics 8.4.58 (conda env SOD, python ' + ENV_PY + ').',
  FACTS,
  '',
  'CLAIM TO TEST: A bare YOLOE("yoloe-11l-seg.pt").train(data=single_class_yolo_dataset) uses the multi-modal grounding trainer YOLOESegTrainer, which does NOT fuse text-prompt embeddings for the fixed class set, so the classification branch is not properly supervised. This is the PRIMARY reason train/cls loss stays huge (~5-7) and val mAP50 is flat (~0.12-0.14). The correct fine-tune path is trainer=YOLOEPESegTrainer (linear-probe: get_text_pe -> set_classes -> fuse -> train cv3 cls layers).',
  '',
  'CONFIRM or REFUTE rigorously. Default to skepticism.',
  '1) Read ultralytics source under ' + UL + ' : models/yolo/yoloe/train.py (YOLOETrainer.build_dataset, preprocess_batch), the multi_modal branch in data/build.py + data/dataset.py (YOLOMultiModalDataset, how it provides texts), and how YOLOEDetect / the cls head consume text PE (nn/modules/head.py YOLOEDetect, and YOLOEModel.loss). Determine what happens to text/cls supervision when the dataset yaml has only names and NO grounding json / no per-object text.',
  '2) Specifically: when multi_modal=True but the data is a normal detect/seg YOLO dataset, does YOLOMultiModalDataset synthesize texts from class names? If yes, can cls still learn, and why would mAP stay flat at ~0.12 with cls loss ~5? Reconcile with observed numbers.',
  '3) Fetch official Ultralytics YOLOE docs (try https://docs.ultralytics.com/models/yoloe/ and the train/fine-tune section) and report exactly which trainer + freeze strategy they prescribe for custom-dataset fine-tuning. Quote it.',
  '4) Verdict. If you cannot fully confirm, say so and explain residual uncertainty.',
  'Use ' + ENV_PY + ' for python introspection. Cite file:line.',
].join('\n')

const inferPrompt = [
  'You are the ONLY GPU agent. Empirically test why the TRAINED YOLOE model detects nothing. Use conda env SOD python at ' + ENV_PY + '.',
  FACTS,
  '',
  'Write a small python script to a temp file and run it with ' + ENV_PY + '. Requirements:',
  '- Set env YOLO_OFFLINE=1, ULTRALYTICS_OFFLINE=1 before importing ultralytics.',
  '- Load ONLY ONE model at a time; del it and torch.cuda.empty_cache() before loading the next (constrained ~GPU on a 15GB box). On CUDA OOM, fall back to device=cpu.',
  '- Pick 5 image files from ' + VAL_DIR + ' (sorted, first 5).',
  '- STEP A (trained): from ultralytics import YOLOE; m=YOLOE("' + TRAINED + '"); print m.task and m.names. Run m.predict(img, conf=0.001, verbose=False) on the 5 images WITHOUT set_classes. Sum total boxes; record single max box confidence across all 5. Then conf=0.25 and sum total boxes.',
  '- STEP B (base zero-shot): del trained; m2=YOLOE("' + BASE + '"); m2.set_classes(["car"]) then predict same 5 at conf=0.10 -> sum boxes + max conf. Also try set_classes(["汽车"]) and note if very different.',
  '- Print a clear machine-readable summary with all numbers at the end.',
  'Report REAL numbers, do not fabricate. On error, put the traceback in raw_notes. Keep runtime modest (CPU fallback on 5 imgs is fine).',
].join('\n')

const smallPrompt = [
  'Quantify the small-object / resolution problem to judge how much it ALONE limits mAP independent of the trainer bug. Use ' + ENV_PY + ' or shell tools.',
  FACTS,
  'Label dir: ' + LABEL_DIR,
  'Train images dir: ' + BACKEND + '/datasets/312b585e-7785-43a4-9f10-94456a894ae1/yolo/f190a05c-b7b6-4616-a4c4-0068327bde03/images/train',
  'Tasks:',
  '1) Parse polygon labels (class x1 y1 x2 y2 ... normalized). For each object compute bbox rel width/height from min/max polygon coords. Report median rel width/height, p90 rel width, mean objects per image.',
  '2) Sample ~10 training images, read pixel resolution via ' + ENV_PY + ' (PIL/cv2) or identify/file. Report typical source resolution.',
  '3) Compute median object size in pixels after letterbox to imgsz=640 (rel_size*640). State whether median objects fall below ~the detectable floor (objects under ~16-32px are very hard for stride 8/16/32 heads).',
  '4) Severity: is small-object+imgsz640 a dominant, significant, or minor cap on mAP assuming the trainer bug is fixed? Note training uses no slicing while inference has SAHI, and mosaic/scale aug further shrink objects.',
  'Be quantitative.',
].join('\n')

const fixPrompt = [
  'Produce the concrete FIX for YOLOWorld/train_yoloe.py so YOLOE actually learns a custom fixed-class (single-class, small-object) dataset, for ultralytics 8.4.58.',
  FACTS,
  'Read ' + BACKEND + '/YOLOWorld/train_yoloe.py (esp. _load_yoloe_model and run_training). Read PE trainers at ' + UL + '/models/yolo/yoloe/train.py (YOLOEPETrainer) and train_seg.py (YOLOEPESegTrainer); confirm passing them via model.train(..., trainer=YOLOEPESegTrainer). Confirm a -seg checkpoint must use the Seg PE trainer, and whether to set close_mosaic/freeze.',
  'Deliver:',
  '1) corrected_run_training_snippet: drop-in replacement for the model-build + model.train(...) portion of run_training(). Must: load YOLOE(model); pick YOLOEPESegTrainer when checkpoint/task is segment else YOLOEPETrainer; pass trainer=...; keep existing kwargs (data, epochs, imgsz, batch, device, project, name, workers, patience) and optional freeze/lr0; include imports; match file style and offline setup. Valid python.',
  '2) key_changes: bullets.',
  '3) recommended_hyperparams: for single-class tiny-object data on a 15GB box with low workers (imgsz, epochs, batch, lr0, optimizer, close_mosaic, mosaic, scale, plus note on raising imgsz ~1280 and/or tiling for training) each with one-line why.',
  '4) inference_guidance: confirm after retraining the IMAGE detect path (detect_with_model with model_id, NO set_classes, conf~0.25) is correct; YOLOE_BASE_MODEL stays base for zero-shot/video while trained weights are chosen per-request by model_id; what conf; whether to enable SAHI for tiny objects.',
  '5) caveats.',
].join('\n')

const results = await parallel([
  () => agent(trainerPrompt, { label: 'verify:trainer-rootcause', phase: 'Verify', schema: TRAINER_SCHEMA }),
  () => agent(inferPrompt, { label: 'verify:inference-empirical', phase: 'Verify', schema: INFER_SCHEMA }),
  () => agent(smallPrompt, { label: 'verify:smallobject-imgsz', phase: 'Verify', schema: SMALL_SCHEMA }),
  () => agent(fixPrompt, { label: 'design:fix', phase: 'Verify', schema: FIX_SCHEMA }),
])

return {
  trainer: results[0],
  inference: results[1],
  smallobject: results[2],
  fix: results[3],
}
