# mnist-inference-ibmcloud
Cloud ML Assignment - 4. Deploy MNIST on IBM Cloud Kubernetes Cluster and get Inference

# Set up Environment
1) Load Vagrant VM (ubuntu:latest) with docker installed and IBM Cloud CLI Installed and ssh into it: 
 vagrant up/reload --provisions, vagrant ssh

# Create Docker Image
Update params of the model in the docker file, python code
Create docker image with new tag: - docker build -t mnist-inference:tag .
Push to Docker Hub: 
	docker login --username=username 
	docker tag {Image ID} username/mnist-inference:tag  
	docker push username/mnist-inference:tag
(Optional) Run Train or Inference locally
	docker run -p 8001:8001 -it mnist-inference:tag
	curl -X POST "http://localhost:8001/train?batch-size=32&epochs=5&lr=0.3"
	curl -X POST -F image=@test.jpg "http://localhost:8001/inference"


# Set up Minikube Cluster
Install mnikube - curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 \
  && chmod +x minikube
  sudo mkdir -p /usr/local/bin/
  sudo install minikube /usr/local/bin/

Create deployment of replica sets with yaml file (Check Docker Image)
 kubectl create -f mnist-inference.yaml (use mnist_train.yaml to train)


# Set Up Kubernetes Cluster
Go to kubernetes dashboard on your IBM portal and create the deployment using yaml file (sepcifying the correct docker image)
   Login to IBM CLoud: ibmcloud login -a cloud.ibm.com -r us-south -g Default \n
   Connect to ibm-cluster by setting context: ibmcloud ks cluster config --cluster bqriu25d0hvamd79p3bg \n
Check status - kubectl get pods \n

(Optional - Update Image)
kubectl set image deployment/mnist-deployment mnist-inference=mnist-inference:tag

# Create Service and Expose Endpoint
kubectl expose deployment mnist-deployment --port 8001 --target-port 8001 --type=NodePort \n
minikube service mnist-deployment \n
Get Port: kubectl describe svc mnist-deployment \n
Get Enpoint (Worker's Public IP): ibmcloud ks worker ls --cluster {cluster_name}

# Run App
Run Train
curl -X POST "http://{Cluster IP}:{Node Port}/train?batch-size=32&epochs=5&lr=0.03"

# Run Inference
curl -X POST -F image=@test.jpg "http://{Endpoint}:{Node Port}/inference"

View Inference on Terminal 

# View Cluster Config
ibmcloud ks clusters
cd ~/.bluemix/plugins/container-service/clusters/{Cluster ID}
