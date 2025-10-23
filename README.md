# Image extractor from video

## Prerequisite

Python installed

And install dependencies by `requirements.txt`

```[bash]
pip install -r requirements.txt
```

## Usage

Run script with `--help` flag to see available options

```[bash]
python extractor.py --help
```

Example

```[bash]
python extractor.py videos -o results -fb -sf 2 -v
```

This command executes the script in verbose mode,
reading all video files from the `videos` directory,
extracting an image every 2 frames,
and saving the non-blurry images to the `results` directory.
