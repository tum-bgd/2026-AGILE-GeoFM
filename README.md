# GeoFM

## Models
### Download Checkpoints for GroundingDINO and RemoteClip
```
mkdir ./weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P ./weights/
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -P ./weights/
wget -q https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt -P ./weights/
wget -q https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-L-14.pt -P ./weights/
```


## Data
Data has to be downloaded and saved into a folder `/data`
### BBD
BBD data was originally provided here: https://doi.org/10.14459/2023mp1709451

The preprocessed files, which are tiled and cleaned to not contain empty images can be found here: https://figshare.com/s/83c2cead0d8fcc65dc6c

### Surface Water
The surface water dataset was originally proposed here: https://doi.org/10.11588/data/AAKAF9

The preprocessed files which are tiled, restricted to three visible bands and preprocessed can be found here: https://figshare.com/s/a359b71447c18d02b9db

The preprocessing script is in `src/prepare_surface_water.py`.


## Docker Environment
Build the docker image
```
docker build -t python-geofm .
```
Run and access the docker container
```
docker run --gpus device=1 --name geofm-container -d -it --mount type=bind,source="$(pwd)",target=/GEO python-geofm
docker cp geofm-container:/GEOtmp/GroundingDINO ./GroundingDINO
docker exec -it geofm-container bash
```

Run the evaluation of the models like specified in the following section

## Models
1. Evaluate SAM with prompts from ground-truth masks
    ```
    python3 src/sam_prompt.py --dataset bbd1k
    ```
    1. `--prompt bb` : bounding-boxes

    2. `--prompt center_pt` : center point of mask geometries

    3. `--prompt multiple_pts` : multiple points randomly generated within each mask geometry (`--nr_pts` to specify the number of points in each gemoetry)

    4. `--prompt foreground_background_pts` : background and foreground set of randomly generated points (`--nr_pts` to specify the number of points in each of the foreground and background)

2. Evaluate SAM with bounding box prompt generated from text prompts run through GroundingDINO:
    ```
    python3 src/sam_dino_prompt.py --dataset bbd1k --text_prompt building
    ```

3. Evaluate classified automatic SAM for the wanted label:
    ```
    python3 src/sam_automatic_label.py --dataset bbd1k --label building
    ```
