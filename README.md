# GeoFM

## Docker Environment
Build the docker file
```
docker build --target python-geofm -t python-geofm .
```
Run and access the docker container
```
docker run --gpus device=1 -d -it --mount type=bind,source="$(pwd)",target=/GEO python-geofm
docker exec -it <docker-container-id> bash
```
Start a tmux session and attach to it
```
tmux
tmux attach -t 0
```
Run the evaluation of the models like specified in the following section

Detach from the session and let the code run in the background
```
Ctrl+b d
```

## Models
1. Evaluate SAM with prompts from ground-truth masks
    ```
    python3 src/sam_prompt.py
    ```
    1. `--prompt bb` : bounding-boxes

    2. `--prompt center_pt` : center point of mask geometries

    3. `--prompt multiple_pts` : multiple points randomly generated within each mask geometry (`--nr_pts` to specify the number of points in each gemoetry)

    4. `--prompt foreground_background_pts` : background and foreground set of randomly generated points (`--nr_pts` to specify the number of points in each of the foreground and background)
