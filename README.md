# GeoFM

## Download Checkpoints for GroundingDINO, RemoteCLIP and RemoteSAM
```
mkdir -p ./weights/bert-base-uncased
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P ./weights/
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -P ./weights/

wget -q https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt -P ./weights/
wget -q https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-L-14.pt -P ./weights/

wget -q https://huggingface.co/1e12Leon/RemoteSAM/resolve/main/RemoteSAMv1.pt -P ./weights/

wget -q https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json -P ./weights/bert-base-uncased/
wget -q https://huggingface.co/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin -P ./weights/bert-base-uncased/
wget -q https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json -P ./weights/bert-base-uncased/
wget -q https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json -P ./weights/bert-base-uncased/
wget -q https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt -P ./weights/bert-base-uncased/
```

## Download RemoteSAM GitHub Repository
```
git clone https://github.com/1e12Leon/RemoteSAM.git
```

## Data
Data has to be downloaded and unzipped into the folder `/data`
### BBD
BBD data was originally provided here: https://doi.org/10.14459/2023mp1709451

The preprocessed files, which are tiled and cleaned to not contain empty images can be found here: https://figshare.com/s/83c2cead0d8fcc65dc6c.

### Surface Water
The surface water dataset was originally proposed here: https://doi.org/10.11588/data/AAKAF9. The preprocessing script used is in `src/prepare_surface_water.py`.

The preprocessed files which are tiled, restricted to three visible bands and preprocessed can be found here: https://figshare.com/s/a359b71447c18d02b9db.

## Environment
For all experiments except RemoteSAM, we can build the following Docker image:
```
docker build -t python-geofm .
```
Run and access the docker container
```
docker run --gpus device=1 --name geofm-container -d -it --mount type=bind,source="$(pwd)",target=/GEO python-geofm
docker cp geofm-container:/GEOtmp/GroundingDINO ./GroundingDINO
docker exec -it geofm-container bash
```

For the RemoteSAM experiments, create an environment as described in RemoteSAM/README.md. 
It might be necessary to edit the last library to spacy==3.7.4
A pandas and importlib-resources installation are required and missing from RemoteSAM/requirements.txt

## Models
Run the evaluation of the models like specified in the following. You can change the SAM model size with `--model_name` with possible model sizes base, large, and huge. You must use `--dataset water1k` to test on the Surface Water Dataset.

1. Evaluate SAM with prompts from ground-truth masks
    ```
    python3 src/sam_gt_prompt.py --dataset bbd1k
    ```

    The following prompt types are supported:
    1. `--prompt bb` : bounding-boxes

    2. `--prompt center_pt` : center point of mask geometries

    3. `--prompt multiple_pts` : multiple points randomly generated within each mask geometry (`--nr_pts` to specify the number of points in each gemoetry)

    4. `--prompt foreground_background_pts` : background and foreground set of randomly generated points (`--nr_pts` to specify the number of points in each of the foreground and background)

2. Evaluate SAM with bounding box prompt generated from text prompts run through GroundingDINO:
    ```
    python3 src/sam_dino_prompt.py --dataset bbd1k --text_prompt building
    ```

    Multiple text prompts can be passed as such `--text_prompt "river . stream . lake ."`

    You can change the GroundingDINO model size with `--dino_model_name` with possible model sizes SwinT and SwinB. The arguments `--box_threshold` and `--text_threshold` control GroundingDINO objectness and vision-language similarity.

3. Evaluate SAM Automatic classified with CLIP, RemoteCLIP and a trained few-shot adapter for CLIP (Tip-Adapter):
    ```
    python3 src/sam_automatic_label.py --dataset bbd1k --label buildings
    ```

    SAM Automatic can be parameterized with `--points_per_side`, which specifies how many points are sampled uniformly along each side of the image. This results in a total of points_per_side**2 sampled points across the image.

    The used CLIP model can be changed using the `--clip_model_name` flag. For RemoteCLIP, choose ViT-B-32 or ViT-L-14, while ViT-bigG-14 if for the original CLIP model without fine-tuning on geospatial data. `--clip_threshold` specifies the minimum CLIP confidence required to retain SAM-generated masks classified as the target label.

    To train a few-shot adapter for CLIP and then evaluate it with SAM Automatic: First create a folder of images with at least the number of shots to be used from the foreground (buildings or water surfaces) and the background. These should be under the folders data/bbd1k_cache/foreground and data/bbd1k_cache/background, respectively. Such images can be for example extracted frrom the dataset with the help of the ground truth masks.
    ```
    python3 src/train_tip_adapter_f.py --cache_dir data/bbd1k_cache/ --label buildings
    python3 src/sam_automatic_label.py --dataset bbd1k --label buildings --few-shot --cache_dir data/bbd1k_cache/
    ```

    Training parameters like `--shots`, `--lr`, `--augment_epoch`, `--train_epoch`, `--beta` and `--alpha` can be adapted as necessary.

4. Evaluate RemoteSAM with text prompts:
    ```
    python3 src/remote_sam_text_prompt.py --dataset bbd1k --text_prompt building
    ```

    Multiple text prompts can be passed as such `--text_prompt river stream lake`
