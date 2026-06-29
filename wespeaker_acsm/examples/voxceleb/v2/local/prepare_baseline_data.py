#!/usr/bin/env python3
# Copyright 2026
# Licensed under the Apache License, Version 2.0.

import argparse
from pathlib import Path

import numpy as np


DEFAULT_TRIALS = {
    "vox1_O_cleaned.kaldi": "/xmudata/pzj/vox-ca/vox1_O_cleaned.trial",
    "vox1_E_cleaned.kaldi": "/xmudata/pzj/vox-ca/vox1_E_cleaned.trial",
    "vox1_H_cleaned.kaldi": "/xmudata/pzj/vox-ca/vox1_H_cleaned.trial",
    "only_ca5.kaldi": "/xmudata/pzj/vox-ca/trials/only-CA5/trials",
    "only_ca10.kaldi": "/xmudata/pzj/vox-ca/trials/only-CA10/trials",
    "only_ca15.kaldi": "/xmudata/pzj/vox-ca/trials/only-CA15/trials",
    "only_ca20.kaldi": "/xmudata/pzj/vox-ca/trials/only-CA20/trials",
    "vox_ca5.kaldi": "/xmudata/pzj/vox-ca/trials/Vox-CA5/trials",
    "vox_ca10.kaldi": "/xmudata/pzj/vox-ca/trials/Vox-CA10/trials",
    "vox_ca15.kaldi": "/xmudata/pzj/vox-ca/trials/Vox-CA15/trials",
    "vox_ca20.kaldi": "/xmudata/pzj/vox-ca/trials/Vox-CA20/trials",
}


def read_wav_scp(path):
    table = {}
    with open(path, "r", encoding="utf8") as fin:
        for line in fin:
            if not line.strip():
                continue
            key, wav = line.rstrip("\n").split(maxsplit=1)
            table[key] = wav
    return table


def as_wespeaker_key(key):
    if "/" in key:
        return key if Path(key).suffix else key + ".wav"
    parts = key.split("-")
    if len(parts) < 3:
        return key if Path(key).suffix else key + ".wav"
    speaker = parts[0]
    utterance = parts[-1]
    video = "-".join(parts[1:-1])
    return f"{speaker}/{video}/{utterance}.wav"


def normalize_label(label):
    if label in {"1", "target", "true", "True"}:
        return "target"
    if label in {"0", "nontarget", "false", "False"}:
        return "nontarget"
    raise ValueError(f"Unsupported trial label: {label}")


def read_age_segments(path):
    if not path:
        return None
    loaded = np.load(path, allow_pickle=True)
    if getattr(loaded, "shape", None) == ():
        loaded = loaded.item()
    if not isinstance(loaded, dict):
        raise SystemExit(f"age_label_file must contain a dict: {path}")
    return set(loaded)


def age_segment_from_key(key):
    key = as_wespeaker_key(key)
    parts = key.split("/")
    if len(parts) < 2:
        return None
    return f"{parts[0]}-{parts[1]}"


def prepare_train(args, out_dir):
    wav_table = read_wav_scp(args.vox2_wav_scp)
    age_segments = read_age_segments(args.age_label_file)
    converted = []
    missing = []
    speakers = set()

    with open(args.train_label, "r", encoding="utf8") as fin:
        for line in fin:
            if not line.strip():
                continue
            utt, spk = line.split()[:2]
            key = as_wespeaker_key(utt)
            if age_segments is not None and age_segment_from_key(key) not in age_segments:
                continue
            if key not in wav_table:
                if len(missing) < 20:
                    missing.append((utt, key))
                continue
            converted.append((key, spk, wav_table[key]))
            speakers.add(spk)

    if missing:
        raise SystemExit(
            "Training keys failed to map to vox2 wav.scp; examples: "
            + repr(missing)
        )
    if not converted:
        raise SystemExit("No training rows were produced")

    train_dir = out_dir / args.train_dir_name
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "wav.scp").write_text(
        "".join(f"{key} {wav}\n" for key, _, wav in converted), encoding="utf8"
    )
    (train_dir / "utt2spk").write_text(
        "".join(f"{key} {spk}\n" for key, spk, _ in converted), encoding="utf8"
    )
    (train_dir / "spk2id").write_text(
        "".join(f"{spk} {idx}\n" for idx, spk in enumerate(sorted(speakers))),
        encoding="utf8",
    )
    (train_dir / "key.list").write_text(
        "".join(f"{key}\n" for key, _, _ in converted),
        encoding="utf8",
    )
    return len(converted), len(speakers)


def prepare_trials(args, out_dir):
    vox1_keys = set(read_wav_scp(args.vox1_wav_scp))
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for out_name, src in DEFAULT_TRIALS.items():
        src_path = Path(src)
        if not src_path.exists():
            summary.append((out_name, src, 0, "missing source"))
            continue
        rows = []
        missing = []
        with src_path.open("r", encoding="utf8") as fin:
            for line in fin:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 3:
                    raise SystemExit(
                        f"Expected 3 columns in trial {src}: {line.strip()}"
                    )
                enroll = as_wespeaker_key(parts[0])
                test = as_wespeaker_key(parts[1])
                label = normalize_label(parts[2])
                if enroll not in vox1_keys or test not in vox1_keys:
                    if len(missing) < 20:
                        missing.append((parts[0], enroll, parts[1], test))
                    continue
                rows.append((enroll, test, label))
        if missing:
            raise SystemExit(
                f"Trial {src} has keys missing from vox1 wav.scp; examples: {missing}"
            )
        (trials_dir / out_name).write_text(
            "".join(f"{a} {b} {c}\n" for a, b, c in rows), encoding="utf8"
        )
        summary.append((out_name, src, len(rows), "ok"))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare baseline data views.")
    parser.add_argument("--out_dir", default="data/baseline")
    parser.add_argument("--train_dir_name", default="vox2_train_voxca")
    parser.add_argument("--train_label", default="/data/vox2_dev/utt2spk")
    parser.add_argument("--age_label_file",
                        default="/xmudata/pzj/vox-ca/vox2dev/segment2age.npy")
    parser.add_argument("--vox2_wav_scp", default="/data/vox2_dev/wav.scp")
    parser.add_argument("--vox1_wav_scp", default="/data/vox1/wav.scp")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_utts, train_spks = prepare_train(args, out_dir)
    trial_summary = prepare_trials(args, out_dir)

    lines = [
        f"train_utts {train_utts}",
        f"train_speakers {train_spks}",
        f"age_labels_used {str(bool(args.age_label_file)).lower()}",
        "trial\tsource\trows\tstatus",
    ]
    lines.extend(
        f"{name}\t{src}\t{rows}\t{status}"
        for name, src, rows, status in trial_summary
    )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
