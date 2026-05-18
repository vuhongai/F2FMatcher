import os
import sys
import pickle
import torch

from f2fmatcher.config import load_config
from f2fmatcher.utils.seed import set_seed
from f2fmatcher.utils.io_utils import ensure_dirs
from f2fmatcher.io.czi_reader import import_resize_export_czi, resize_image, find_img_path
from f2fmatcher.segmentation.cellpose_seg import generate_VAE_inputs, segment_image, filter_ROIs
from f2fmatcher.vae.embed import generate_embedding_dir
from f2fmatcher.classifier.model import PairClassifier
from f2fmatcher.matching.matcher import match_fibers
from f2fmatcher.visualization.plot import save_FM_prediction


def run_single_pair(img1, img2, source_1, source_2, czi1, czi2, channel1, channel2,
                    cp_model_1, cp_model_2, param1, param2, obj1, obj2,
                    dir_output, export_images, save_step_prediction, n_processes, device,
                    skip_vae_inputs, skip_embeddings,
                    config):
    param_ref = ["fluorescence", "10X", 1.0]
    param_img1 = [param1, obj1, 1.0]
    param_img2 = [param2, obj2, 1.0]
    IHF1 = param1 == "fluorescence"
    IHF2 = param2 == "fluorescence"

    images_dir = f"{dir_output}/images_segmentation"
    dir_embedding = f"{dir_output}/VAE_embed"
    dir_npz = f"{dir_output}/npz_256"
    dir_prediction = f"{dir_output}/prediction_output"
    dir_cp_masks = f"{dir_output}/out_CP_masks"

    ensure_dirs([dir_output, images_dir, dir_embedding, dir_npz, dir_prediction, dir_cp_masks])

    pair_dir = f"{dir_prediction}/{img1}___vs___{img2}"
    os.makedirs(pair_dir, exist_ok=True)

    if os.path.exists(f"{pair_dir}/paired_labels.pkl"):
        print(f"Prediction for {img1} vs {img2} already exists, skipping.")
        return

    # 1. Image I/O
    if not os.path.exists(f"{images_dir}/{img1}.png"):
        if czi1:
            import_resize_export_czi(find_img_path(img1, source_1), IHF1, channel1, images_dir, param_img1, param_ref)
        else:
            resize_image(find_img_path(img1, source_1), images_dir, param_img1, param_ref)

    if not os.path.exists(f"{images_dir}/{img2}.png"):
        if czi2:
            import_resize_export_czi(find_img_path(img2, source_2), IHF2, channel2, images_dir, param_img2, param_ref)
        else:
            resize_image(find_img_path(img2, source_2), images_dir, param_img2, param_ref)

    # 2. VAE inputs (Cellpose segmentation + ROI cropping)
    seg_kwargs = {}
    cp_path = config.get("cellpose.model_path")
    if cp_path is not None:
        seg_kwargs["CP_model_path"] = cp_path

    if skip_vae_inputs:
        cp1 = segment_image(f"{images_dir}/{img1}.png", cp_model_1, savedir=dir_cp_masks, **seg_kwargs)
        cp2 = segment_image(f"{images_dir}/{img2}.png", cp_model_2, savedir=dir_cp_masks, **seg_kwargs)
    else:
        cp1 = generate_VAE_inputs(img1, images_dir, dir_npz, cp_model_1, dir_cp_masks,
                                  n_processes=n_processes, **seg_kwargs)
        cp2 = generate_VAE_inputs(img2, images_dir, dir_npz, cp_model_2, dir_cp_masks,
                                  n_processes=n_processes, **seg_kwargs)

    # 3. VAE embeddings
    if not skip_embeddings:
        vae_ckpt = config.get("vae.checkpoint")
        generate_embedding_dir(dir_npz, dir_embedding, vae_ckpt, device, img_names=[img1, img2])

    # 4. Prepare matching
    list_label_1 = sorted(int(f.split("_roi_")[1].split(".")[0]) for f in os.listdir(dir_npz) if img1 in f)
    list_label_2 = sorted(int(f.split("_roi_")[1].split(".")[0]) for f in os.listdir(dir_npz) if img2 in f)

    label2index = {
        "img1": {l: i for i, l in enumerate(list_label_1)},
        "img2": {l: i for i, l in enumerate(list_label_2)},
    }

    # 5. Run matching
    cls_ckpt = config.get("classifier.checkpoint")
    matching_cfg = config.get("matching", {})
    if isinstance(matching_cfg, dict):
        matching_cfg = type("obj", (object,), matching_cfg)

    matched_labels, scores, spatial_dist, cp_outputs = match_fibers(
        img1, list_label_1, cp1,
        img2, list_label_2, cp2,
        label2index, dir_embedding, cls_ckpt, device,
        list_k=matching_cfg.get("list_k", [3, 5, 7]) if hasattr(matching_cfg, "get") else matching_cfg.list_k,
        n_initial_guess=getattr(matching_cfg, "n_initial_guess", 80),
        n_pair_selected=getattr(matching_cfg, "n_pair_selected", 4),
        min_cls_logit_init=getattr(matching_cfg, "min_cls_logit_init", 0.75),
        distance_neighbors_ref=getattr(matching_cfg, "distance_neighbors_ref", 200),
        max_distance_affine=getattr(matching_cfg, "max_distance_affine", 150),
        max_cost_geo_neighbors_sides=getattr(matching_cfg, "max_cost_geo_neighbors_sides", 30),
        max_cost_geo_neighbors_angles=getattr(matching_cfg, "max_cost_geo_neighbors_angles", 0.15),
        min_cls_logit=getattr(matching_cfg, "min_cls_logit", 0.5),
        patience_label=getattr(matching_cfg, "patience_label", 5),
        n_neighbors_validation=getattr(matching_cfg, "n_neighbors_validation", 3),
        n_processes=n_processes,
        n_try_unannotated=getattr(matching_cfg, "n_try_unannotated", 1),
        patience_prediction_neighbors=getattr(matching_cfg, "patience_prediction_neighbors", 5),
        use_multiprocessing_for_local_prediction=getattr(matching_cfg, "use_multiprocessing", True),
        save_step_prediction=save_step_prediction,
        dir_save_prediction_output=pair_dir,
    )

    with open(f"{pair_dir}/paired_labels.pkl", "wb") as f:
        pickle.dump(matched_labels, f)

    # 6. Visualize
    if export_images:
        cp1_out, cp2_out = cp_outputs
        l1f = filter_ROIs(cp1, config.get("dataset.crop_size", 256))
        l2f = filter_ROIs(cp2, config.get("dataset.crop_size", 256))
        save_FM_prediction(img1, images_dir, 0, [cp1, l1f], matched_labels, pair_dir, IHF=IHF1)
        save_FM_prediction(img2, images_dir, 1, [cp2, l2f], matched_labels, pair_dir, IHF=IHF2)

    # cleanup
    for f in os.listdir(dir_embedding):
        if f.split("_roi_")[0] in [img1, img2]:
            os.remove(os.path.join(dir_embedding, f))
    for f in os.listdir(dir_npz):
        if f.split("_roi_")[0] in [img1, img2]:
            os.remove(os.path.join(dir_npz, f))

    print(f"Done: {len(matched_labels)} pairs matched.")

    return matched_labels


def run_pipeline(args):
    config = load_config(args.param_file) if args.param_file else load_config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.param_file:
        from f2fmatcher.utils.io_utils import import_module_from_path
        params = import_module_from_path(args.param_file)
        for img1, img2 in params.list_pair_images:
            run_single_pair(
                img1, img2,
                params.source_1, params.source_2,
                params.czi_img1, params.czi_img2,
                params.channel_index_img1, params.channel_index_img2,
                params.CP_model_name_1, params.CP_model_name_2,
                params.param_img1, params.param_img2, params.obj1, params.obj2,
                params.dir_save_output,
                params.export_output_images,
                params.save_step_prediction,
                getattr(params, "n_processes", 60),
                device,
                getattr(params, "skip_generate_VAE_inputs", False),
                getattr(params, "skip_generate_embeddings", False),
                config,
            )
    else:
        run_single_pair(
            args.img1, args.img2,
            args.source1, args.source2,
            args.czi1, args.czi2,
            args.channel1, args.channel2,
            args.cp_model_1, args.cp_model_2,
            args.param_img1, args.param_img2, args.obj1, args.obj2,
            args.output, args.export_images, args.save_step_prediction,
            args.n_processes, device,
            args.skip_vae_inputs, args.skip_embeddings,
            config,
        )
