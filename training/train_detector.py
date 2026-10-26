"""Train M1 or M2 for one leave-one-generator-out fold (owner: Workstream B).

    python -m training.train_detector --config configs/m1.yaml --fold 0

TODO: load_config -> set_seed -> ManifestDataset(train/val, fold) -> model -> AdamW ->
loop with BCEWithLogitsLoss -> keep best epoch by validation AUROC -> save checkpoint
dict {state_dict, config, epoch, val_metrics} -> predictions.csv on test ->
common.runlog.start_run / finish_run.
"""
