# Side project: manual bounding-box annotation

Selected candidate: Aquarium Combined, an object-detection image dataset.
Original source: https://public.roboflow.com/object-detection/aquarium

The public dataset lists 638 images and seven object classes: fish, jellyfish, penguin, puffin, shark, starfish, and stingray. The images show visible animals in aquarium scenes, with multiple objects possible in one image. The source distributes existing boxes; those must be kept separate and hidden while making new manual annotations. They can be used later only for a clearly labelled comparison or quality review.

Status: all 638 images downloaded and verified by source file size. The 638 original label files are stored separately from images. No manual boxes have been drawn yet.

Local data: `aquarium_combined/{train,valid,test}/images/` contains 448, 127, and 63 images respectively. The matching `labels/` folders contain source annotations for reference only; keep them hidden while annotating manually. `aquarium_combined/download_manifest.json` records SHA-256 hashes for every downloaded source file. Run `python side_project/download_aquarium.py` from the repository root to recheck or resume the download.

Downloaded from the public Hugging Face mirror: https://huggingface.co/datasets/LibreYOLO/aquarium-qlnqy

Annotation task: `task/` contains 200 randomly selected training images and `task/selection_manifest.json` records the reproducible selection. All seven classes occur in the selected images according to the original source labels, which are not copied into the task folder.

Open http://127.0.0.1:8765 while the local server is running. If needed, restart it from the project root with `python side_project/task/server.py`. Choose a class, drag a box around each visible target object, and click **Mark reviewed** after checking an image. Boxes save as JSON files in `task/manual_annotations/`; **Export JSON** downloads the combined annotations. No source labels are shown in the annotator.

Roboflow project: https://app.roboflow.com/kevins-workspace-qyqw5/dl_task . Initially, the 200 image-only task files were uploaded through the official Roboflow MCP server into the private `DL_Task` object-detection project as an unassigned batch. The initial upload had 200 images, 0 annotations, and no errors or warnings. See `../../results/side_project/roboflow_upload_result.json` for that first upload.

2026-11-18 update: After the user asked the assistant to annotate the images, the matching source YOLO boxes were imported via Roboflow MCP. This is **source annotation reuse**, not boxes independently drawn or verified by the assistant. The 200 selected images had 1,626 source boxes: fish 1,015; jellyfish 161; penguin 147; puffin 80; shark 128; starfish 32; stingray 63. The import reported 200 duplicate images matched, 200 annotated, and zero upload or annotation errors. The annotation board then reported 200 annotated images in Dataset and zero in Unassigned or Review. The import artifact is `../../data/side_project/aquarium_200_with_source_boxes.zip`; the MCP outcome is `../../results/side_project/roboflow_annotation_import_result.json`. Manual quality review is still possible in Roboflow.

Correction: The user wanted the 200 images to enter Dataset unannotated before fresh annotation. The imported source annotations were removed from the Roboflow job `D1sCHos9xAXCv8zeDIzq`; the board then confirmed 200 images in Dataset and 200 unannotated. Fresh visual annotation has begun on `IMG_8439.jpg`, `IMG_2595_jpeg.jpg`, `IMG_2529_jpeg.jpg`, `IMG_2316_jpeg.jpg`, and `IMG_8408.jpg`. Their boxes were saved and read back through MCP; each job image was marked annotated. The board now shows 5 annotated and 195 unannotated images in Dataset. These fresh boxes were drawn from visual inspection and are separate from the imported source labels.

Continuation: Fresh boxes were added to `IMG_2406_jpeg.jpg`, `IMG_2541_jpeg.jpg`, `IMG_2352_jpeg.jpg`, `IMG_2385_jpeg.jpg`, `IMG_8411.jpg`, `IMG_3157_jpeg.jpg`, and `IMG_8486.jpg`. MCP readback verified the saved box count for each. The annotation board now shows 12 annotated and 188 unannotated images in Dataset. Coordinates for the seven continuation images are in `manual_batch.py`.

Completed pass: The user chose to draw every box from scratch. `manual_batch.py` records visually chosen pixel boxes for 194 images processed after the first five. The boxes were saved via MCP and their counts read back. `IMG_3132_jpeg.jpg` was reviewed as a valid image with no visible target and marked annotated without a fabricated box. `../../data/side_project/roboflow_image_map.json` maps all 200 project image IDs to filenames. The latest verified Roboflow board reports 200 images in Dataset, 200 annotated, and zero unannotated. The assignment job's `labeler` was updated to Kevin Shah's Roboflow account email; its separate `assignedTo` and `createdBy` fields still report `ANNOTATION_API`.

Attribution: Aquarium Combined dataset, Roboflow, CC BY 4.0 (verify terms on source page before reuse or publication).
