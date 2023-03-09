# qt coyotes data labeled

This repository contains all of the labeled coyote data (images and metadata)
and scripts for managing and preprocessing data for use in machine learning.

## Installation

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## DVC

A DVC remote is setup with Google Drive, to pull data use `dvc pull`.

## Format

Will convert all metadata to
[COCO Camera Traps](https://github.com/microsoft/CameraTraps/tree/main/data_management)
format with additional fields.
