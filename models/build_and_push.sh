#!/bin/bash
export DOCKERTAG="-v0.0.1"

docker build -t pangyuteng/dlstp:3duxnet${DOCKERTAG} -f 3duxnet/docker/Dockerfile 3duxnet/docker
docker push pangyuteng/dlstp:3duxnet${DOCKERTAG}

docker build -t pangyuteng/dlstp:nnunet${DOCKERTAG} -f nnUnet/docker/Dockerfile nnUnet/docker
docker push pangyuteng/dlstp:nnunet${DOCKERTAG}

docker build -t pangyuteng/dlstp:umamba${DOCKERTAG} -f umamba/docker/Dockerfile umamba/docker
docker push pangyuteng/dlstp:umamba${DOCKERTAG}

docker build -t pangyuteng/dlstp:unetr${DOCKERTAG} -f unetr/docker/Dockerfile unetr/docker
docker push pangyuteng/dlstp:unetr${DOCKERTAG}

docker build -t pangyuteng/dlstp:3duxnet-report${DOCKERTAG} -f 3duxnet/docker/Dockerfile.report 3duxnet/docker
docker push pangyuteng/dlstp:3duxnet-report${DOCKERTAG}

docker build -t pangyuteng/dlstp:nnunet-report${DOCKERTAG} -f nnUnet/docker/Dockerfile.report nnUnet/docker
docker push pangyuteng/dlstp:nnunet-report${DOCKERTAG}

docker build -t pangyuteng/dlstp:umamba-report${DOCKERTAG} -f umamba/docker/Dockerfile.report umamba/docker
docker push pangyuteng/dlstp:umamba-report${DOCKERTAG}

docker build -t pangyuteng/dlstp:unetr-report${DOCKERTAG} -f unetr/docker/Dockerfile.report unetr/docker
docker push pangyuteng/dlstp:unetr-report${DOCKERTAG}

