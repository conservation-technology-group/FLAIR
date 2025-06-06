import os
import gc
import time
import numpy as np
from collections import defaultdict
from utils import get_bounding_box, calculate_iou
import torch
import torchvision

def run_propagation(
    predictor,
    inference_state,
    frame_keep_boxes,
    frame_names,
    video_dir,
    image_shape,
    frame_gap,
    window_length,
    output_dir,
    bbox_buffer
):
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    curr_id = 0
    correct = []
    frames = []
    repropagate = []

    for a in range(len(frame_keep_boxes)):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nSecond: {a}")

        if curr_id == 0:
            predictor.reset_state(inference_state)
            for b in range(len(frame_keep_boxes[a])):
                curr_id += 1
                i, j, bbox = frame_keep_boxes[a][b]
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i * frame_gap,
                    obj_id=curr_id,
                    box=bbox,
                )
                correct.append(False)
                frames.append(i)
                repropagate.append(i + window_length)
            if curr_id != 0:
                _propagate(
                    predictor,
                    inference_state,
                    output_dir,
                    start=max(0, frame_gap * (a - window_length)),
                    max_frame=(frame_gap * window_length) + (frame_gap * a - max(0, frame_gap * (a - window_length))),
                    image_shape=image_shape,
                )

        else:
            prev_curr_id = curr_id
            predictor.reset_state(inference_state)

            for b in range(len(frame_keep_boxes[a])):
                i, j, bbox = frame_keep_boxes[a][b]
                filename = os.path.join(output_dir, f'frame_{i * frame_gap}.npy')
                mask_add = True

                if os.path.exists(filename):
                    frame_segments = np.load(filename, allow_pickle=True).item()
                    for key in sorted(frame_segments.keys(), key=int):
                        bounding_box = get_bounding_box(frame_segments[key])
                        if bounding_box is not None:
                            bounding_box = [
                                max(0, bounding_box[0] - bbox_buffer),
                                max(0, bounding_box[1] - bbox_buffer),
                                min(bounding_box[2] + bbox_buffer, image_shape[1]),
                                min(bounding_box[3] + bbox_buffer, image_shape[0])
                            ]
                            iou = calculate_iou(bbox, bounding_box)
                            if iou > 0.7:
                                print("Duplicate mask!")
                                mask_add = False
                                if i != frames[int(key) - 1]:
                                    correct[int(key) - 1] = True
                                break
                    del frame_segments

                if mask_add:
                    curr_id += 1
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=i * frame_gap,
                        obj_id=curr_id,
                        box=bbox,
                    )
                    correct.append(False)
                    frames.append(i)
                    repropagate.append(i + window_length)

            if prev_curr_id != curr_id:
                _propagate(
                    predictor,
                    inference_state,
                    output_dir,
                    start=max(0, frame_gap * (a - window_length)),
                    max_frame=(frame_gap * window_length) + (frame_gap * a - max(0, frame_gap * (a - window_length))),
                    image_shape=image_shape,
                )

        # Repropagate if necessary
        indices = [i for i, x in enumerate(repropagate) if x == a]
        should_reprop = False
        predictor.reset_state(inference_state)

        for index in indices:
            if correct[index]:
                filename = os.path.join(output_dir, f'frame_{(frame_gap * a) - 1}.npy')
                frame_segments_old = np.load(filename, allow_pickle=True).item()
                bounding_box = get_bounding_box(frame_segments_old.get(index + 1))
                del frame_segments_old

                if bounding_box is not None:
                    should_reprop = True
                    print("Re-propagating index", index, "at second", a)
                    bounding_box = [
                        max(0, bounding_box[0] - bbox_buffer),
                        max(0, bounding_box[1] - bbox_buffer),
                        min(bounding_box[2] + bbox_buffer, image_shape[1]),
                        min(bounding_box[3] + bbox_buffer, image_shape[0])
                    ]

                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=(frame_gap * a),
                        obj_id=index + 1,
                        box=bounding_box,
                    )
                    repropagate[index] += window_length

        if should_reprop:
            _propagate(
                predictor,
                inference_state,
                output_dir,
                start=frame_gap * a,
                max_frame=frame_gap * window_length,
                image_shape=image_shape,
            )

    # Filter final masks
    correct_indices = [i + 1 for i, is_correct in enumerate(correct) if is_correct]

    # Wrap the loop with tqdm for progress bar
    for i, fname in enumerate(tqdm(os.listdir(output_dir), desc="Filtering files")):
        fpath = os.path.join(output_dir, fname)
        data = np.load(fpath, allow_pickle=True).item()
        filtered = {k: v for k, v in data.items() if int(k) in correct_indices}
        np.save(fpath, filtered)

    print(f"Window {window_length} took {time.time() - start_time:.2f} seconds")


def _propagate(predictor, inference_state, output_dir, start, max_frame, image_shape):
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    all_results = defaultdict(dict)
    with torch.autocast("cuda", torch.bfloat16):
        with torch.inference_mode():
            iterator = predictor.propagate_in_video(
                inference_state,
                start_frame_idx=start,
                max_frame_num_to_track=max_frame
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
                frame_segments = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                all_results[out_frame_idx].update(frame_segments)
                del frame_segments, out_obj_ids, out_mask_logits

    for out_frame_idx, new_segments in all_results.items():
        file_path = os.path.join(output_dir, f'frame_{out_frame_idx}.npy')
        try:
            old_segments = np.load(file_path, allow_pickle=True).item()
            old_segments.update(new_segments)
            np.save(file_path, old_segments)
            del old_segments
        except FileNotFoundError:
            np.save(file_path, new_segments)
        del new_segments
