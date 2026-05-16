#!/bin/bash
export DOCKERTAG="-v0.0.1"

docker build -t pangyuteng/dlstp:3duxnet${DOCKERTAG} 3duxnet/docker
docker push pangyuteng/dlstp:3duxnet${DOCKERTAG}

docker build -t pangyuteng/dlstp:nnunet${DOCKERTAG} nnUnet/docker
docker push pangyuteng/dlstp:nnunet${DOCKERTAG}

docker build -t pangyuteng/dlstp:umamba${DOCKERTAG} umamba/docker
docker push pangyuteng/dlstp:umamba${DOCKERTAG}

docker build -t pangyuteng/dlstp:unetr${DOCKERTAG} unetr/docker
docker push pangyuteng/dlstp:unetr${DOCKERTAG}
