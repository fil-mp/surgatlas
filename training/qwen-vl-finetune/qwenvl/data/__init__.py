import re


# Set this to the directory containing the SurgAtlas annotation JSONL files
# downloaded from https://huggingface.co/datasets/filbel/SurgAtlas.
SURGATLAS_ANNOTATION_DIR = "PATH_TO_SURGATLAS_ANNOTATIONS"

SURG_SEGMENT_CAPTIONS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/captions_train.jsonl",
    "data_path": "",
}

SURG_STEP_DESCRIPTIONS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/steps_train.jsonl",
    "data_path": "",
}

SURG_OCR_PHASE_DESCRIPTIONS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/ocr_phases_train.jsonl",
    "data_path": "",
}

SURG_VIDEO_SUMMARIES = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/summaries.jsonl",
    "data_path": "",
}

SURG_TITLE_DESCRIPTIONS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/metadata.jsonl",
    "data_path": "",
}

SURGATLAS_QA = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/train_vqa.jsonl",
    "data_path": "",
}

SURG_OCR_PHASE_VQA = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/ocr_phases_vqa_train.jsonl",
    "data_path": "",
}

SURG_SEGMENT_CAPTIONS_OPEN = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/captions_open_train.jsonl",
    "data_path": "",
}

SURG_STEP_DESCRIPTIONS_OPEN = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/steps_open_train.jsonl",
    "data_path": "",
}

SURG_OCR_PHASE_DESCRIPTIONS_OPEN = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/ocr_phases_open_train.jsonl",
    "data_path": "",
}

SURG_VIDEO_SUMMARIES_OPEN = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/summaries_open.jsonl",
    "data_path": "",
}

SURG_TITLE_DESCRIPTIONS_OPEN = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/metadata_open.jsonl",
    "data_path": "",
}

SURGATLAS_QA_OPEN = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/train_vqa_open.jsonl",
    "data_path": "",
}

SURG_OCR_PHASE_VQA_OPEN = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/ocr_phases_vqa_open_train.jsonl",
    "data_path": "",
}

SURG_SEGMENT_CAPTIONS_MIS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/captions_mis_train.jsonl",
    "data_path": "",
}

SURG_STEP_DESCRIPTIONS_MIS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/steps_mis_train.jsonl",
    "data_path": "",
}

SURG_OCR_PHASE_DESCRIPTIONS_MIS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/ocr_phases_mis_train.jsonl",
    "data_path": "",
}

SURG_VIDEO_SUMMARIES_MIS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/summaries_mis.jsonl",
    "data_path": "",
}

SURG_TITLE_DESCRIPTIONS_MIS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/metadata_mis.jsonl",
    "data_path": "",
}

SURGATLAS_QA_MIS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/train_vqa_mis.jsonl",
    "data_path": "",
}

SURG_OCR_PHASE_VQA_MIS = {
    "annotation_path": f"{SURGATLAS_ANNOTATION_DIR}/ocr_phases_vqa_mis_train.jsonl",
    "data_path": "",
}

data_dict = {
    "surg_segment_captions": SURG_SEGMENT_CAPTIONS,
    "surg_step_descriptions": SURG_STEP_DESCRIPTIONS,
    "surg_ocr_phases": SURG_OCR_PHASE_DESCRIPTIONS,
    "surg_video_summaries": SURG_VIDEO_SUMMARIES,
    "surg_title_descriptions": SURG_TITLE_DESCRIPTIONS,
    "surgatlas_qa": SURGATLAS_QA,
    "ocr_phases_vqa": SURG_OCR_PHASE_VQA,
    "surg_segment_captions_open": SURG_SEGMENT_CAPTIONS_OPEN,
    "surg_step_descriptions_open": SURG_STEP_DESCRIPTIONS_OPEN,
    "surg_ocr_phases_open": SURG_OCR_PHASE_DESCRIPTIONS_OPEN,
    "surg_video_summaries_open": SURG_VIDEO_SUMMARIES_OPEN,
    "surg_title_descriptions_open": SURG_TITLE_DESCRIPTIONS_OPEN,
    "surgatlas_qa_open": SURGATLAS_QA_OPEN,
    "ocr_phases_vqa_open": SURG_OCR_PHASE_VQA_OPEN,
    "surg_segment_captions_mis": SURG_SEGMENT_CAPTIONS_MIS,
    "surg_step_descriptions_mis": SURG_STEP_DESCRIPTIONS_MIS,
    "surg_ocr_phases_mis": SURG_OCR_PHASE_DESCRIPTIONS_MIS,
    "surg_video_summaries_mis": SURG_VIDEO_SUMMARIES_MIS,
    "surg_title_descriptions_mis": SURG_TITLE_DESCRIPTIONS_MIS,
    "surgatlas_qa_mis": SURGATLAS_QA_MIS,
    "ocr_phases_vqa_mis": SURG_OCR_PHASE_VQA_MIS,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["surg_segment_captions"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
