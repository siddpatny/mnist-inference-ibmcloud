# mnist-inference-ibmcloud
Cloud ML Assignment - 4. Deploy MNIST on IBM Cloud Kubernetes Cluster and get Inference

# Set up Environment
1) Load Vagrant VM (ubuntu:latest) with docker installed and IBM Cloud CLI Installed and ssh into it: 
 vagrant up, vagrant ssh

#Create Docker Image
2) Update params of the model in the docker file, python code
3) Create docker image with new tag: - docker build -t mnist-inference:tag .
4) Push to Docker Hub: docker login --username=username , docker push mnist-inference:tag
5) (Optional) Run Train or Inference locally
	docker run -p 8001:8001 -it mnist-inference:tag
	curl -X POST -F "http://localhost:8001/train?batch-size=32&epochs=5&lr=0.3"
	curl -X POST -F image=@test.jpg "http://localhost:8001/inference"


#Set Up Kubernetes Cluster
6) Go to kubernetes dashboard on your IBM portal and create the deployment using yaml file (sepcifying the correct docker image)
   Login to IBM CLoud: ibmcloud login
7) Check status - kubectl get pods

(Optional - Update Image)
kubectl set image deployment/mnist-deployment mnist-inference=mnist-inference:tag

8) kubectl expose deployment mnist-deployment --port 8001 --target-port 8001 --type=NodePort
9) Get Enpoint and Port: kubectl describe svc mnist-deployment

#Run App
10) Run Train
	curl -X POST -F "http://{Cluster IP}:{Node Port}/train?batch-size=32&epochs=5&lr=0.3"
11) Run Inference
	curl -X POST -F image=@test.jpg "http://{Cluster IP}:{Node Port}/inference"
12) View Inference on Terminal 