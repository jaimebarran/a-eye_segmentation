# Useful commands

## Docker

- Build docker image

    `docker build --no-cache -t jaimebarran/fw_gear_aeye .`

- Push docker image

    `docker push jaimebarran/fw_gear_aeye`

- Run docker image interactively (local image)

    `docker run -it --rm docker.io/jaimebarran/fw_gear_aeye /bin/bash`

- Run docker image interactively (remote image)

    `docker run -it --rm --entrypoint=/bin/bash jaimebarran/fw_gear_aeye:0.0.1`

- List images

    `docker images` or `docker image ls`

- List containers

    `docker ps -a` or `docker container ls -a`

- Remove image

    `docker rmi <image_id>`

- Remove container

    `docker rm <container_id>`

- Prune images

    `docker image prune`

- Prune containers

    `docker container prune`

## Flywheel

- Log in

    `fw login {FW_API_KEY}`

- Test local gear

    `fw gear local --nifti=input/nifti/I_Kopf_t1_mpr_tra_iso_p2.nii.gz --measurement="auto-detect" --debug=true`

- Upload gear

    `fw gear upload`

### Flywheel beta

- Log in

    `fw-beta login --api-key=${FW_API_KEY}`

- Validate the manifest

    `fw-beta gear --validate manifest.json`

- Build the gear - this runs `docker build` and then extracts the `ENV` inside the Docker container and adds it to the `manifest.json`

    `fw-beta gear build`

- Create the gear `config.json` (needed to later run the gear locally)

    `fw-beta gear config --create`

- List the configuration options for this specific gear

    `fw-beta gear config --show`

- Configure the local job (specify the input and config options)

    `fw-beta gear config -i nifti=input/nifti/<filename> -c debug=True`

- Run the gear locally

    `fw-beta gear run --rm`

## nnUNet

- Test to see if it is installed

    `nnUNet_predict -h`

- Run nnUNet locally

    ```bash
    nnUNet_predict -i "/home/jaimebarranco/Downloads/nnUNet_inference/input" -o "/home/jaimebarranco/Downloads/nnUNet_inference/output" -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task313_Eye
    ```

- Run nnUNet through Docker pushed image without trained model

    ```bash
    sudo docker run --rm --gpus device=0 --shm-size=10gb --entrypoint=/bin/bash -v /home/jaimebarranco/Desktop/nnUNet:/opt/nnunet_resources -e nnUNet_preprocessed=/opt/nnunet_resources/nnUNet_preprocessed -e nnUNet_raw_data_base=/opt/nnunet_resources/nnUNet_raw_data_base -e RESULTS_FOLDER=/opt/nnunet_resources/nnUNet_trained_models -v /home/jaimebarranco/Desktop/nnUNet_inference:/tmp jaimebarran/fw_gear_aeye:0.0.0 -c "nnUNet_predict -i /tmp/input -o /tmp/output -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task313_Eye"
    ```

- Run nnUNet through Docker pushed image with trained model

    ```bash
    sudo docker run --rm --gpus device=0 --shm-size=10gb -v /home/jaimebarranco/Desktop/nnUNet_inference:/tmp jaimebarran/fw_gear_aeye_test:latest nnUNet_predict -i /tmp/input -o /tmp/output -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task313_Eye
    ```

- Run nnUNet through Docker pushed image with trained model changing the entrypoint

    ```bash
    sudo docker run --rm --gpus device=all --shm-size=10gb --entrypoint=/bin/bash -v /home/jaimebarranco/Desktop/nnUNet_inference:/tmp jaimebarran/fw_gear_aeye:0.0.0 -c "nnUNet_predict -i /tmp/input -o /tmp/output -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task313_Eye"
    ```

### CDI

- Last command but with CDI

    ```bash
    sudo docker run --rm --runtime=nvidia --shm-size=10gb --entrypoint=/bin/bash -e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=0 -v /home/jaimebarranco/Downloads/nnUNet_inference:/tmp jaimebarran/fw_gear_aeye:0.0.0 -c "nnUNet_predict -i /tmp/input -o /tmp/output -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task313_Eye"
    ```

- Last command but with CDI set in etc/nvidia-container-runtime/config.toml > nvidia-container-runtime > mode = "cdi" instead of "auto"

    ```bash
    sudo docker run --rm --runtime=nvidia --shm-size=10gb --entrypoint=/bin/bash -e NVIDIA_VISIBLE_DEVICES=0 -v /home/jaimebarranco/Downloads/nnUNet_inference:/tmp jaimebarran/fw_gear_aeye:0.0.0 -c "nnUNet_predict -i /tmp/input -o /tmp/output -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task313_Eye"
    ```

#### Podman

- CDI with Podman (kind of "docker run")

    ```bash
    podman run --rm --device nvidia.com/gpu=0 --security-opt=label=disable -v /home/jaimebarranco/Downloads/nnUNet_inference:/tmp jaimebarran/fw_gear_aeye_test:latest nnUNet_predict -i /tmp/input -o /tmp/output -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task313_Eye
    ```

- Export environment variables (required)

  - nnUNet_raw

    `export nnUNet_raw="/home/jaimebarranco/Desktop/nnUNet/nnUNet_raw_data"`

  - nnUNet_preprocessed

    `export nnUNet_preprocessed="/home/jaimebarranco/Desktop/nnUNet/nnUNet_preprocessed"`

  - nnUNet_results

    `export nnUNet_results="/home/jaimebarranco/Desktop/nnUNet/nnUNet_trained_models"`

## File permissions

- Change file permissions

    `sudo chmod -R a+rw /mnt/sda1/Repos/flywheel/aeye_flywheel/output/input/`

- Change file owner

    `sudo chown jaimebarranco output/input/I_Kopf_t1_mpr_tra_iso_p2.nii.gz`
