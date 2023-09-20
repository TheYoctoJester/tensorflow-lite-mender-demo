# Tensorflow lite Mender Artifact demo

[TensorFlow](https://www.tensorflow.org/) is a powerful and versatile open-source machine learning framework developed by Google. It has emerged as a cornerstone in the world of artificial intelligence and deep learning, offering a robust ecosystem for building, training, and deploying machine learning models. TensorFlow's flexibility and scalability make it a popular choice among researchers and developers, enabling them to tackle a wide range of AI tasks, from image recognition and natural language processing to reinforcement learning and more. Whether you're a seasoned AI practitioner or just getting started, TensorFlow provides the tools and resources needed to bring your AI projects to life.

[TensorFlow Lite](https://www.tensorflow.org/lite), an optimized version of TensorFlow, is specifically tailored for edge devices and mobile platforms. It empowers developers to deploy machine learning models on resource-constrained hardware, extending TensorFlow's capabilities to the edge and enabling AI-powered applications on devices like smartphones, IoT devices, and more.

In the following demonstration, we'll walk you through the steps to deploy a TensorFlow Lite model update on our x86 Debian-based edge device using Mender, showcasing the seamless integration of machine learning at the edge with robust OTA capabilities.

## Prerequisites


- Python3
- `mender-artifact` installed and on `$PATH`, see https://docs.mender.io/downloads#mender-artifact
- a Mender-enabled device, connected to a Hosted Mender account, see step 0 below

## Demo

For the remainder of this demo, we will assume that you have prepared a device running Debian (or a derivative thereof), preferably on an x86 or ARM-based platform.

### Step 0: prepare the device

As a base device, use a device running the Mender client and able to install packages from [pypi.org](https://pypi.org). The recommended setup for this demo is a virtual machine running on VirtualBox or a comparable solution. Follow the guide in the [Mender documentation](https://docs.mender.io/client-installation/install-with-debian-package#install-mender-using-the-debian-package) for the installation of the Mender client. In the following, we refer to the configured device type as "debian-tensorflow".

Install the dependencies:

```
$ sudo apt install python3 python3-pip
$ sudo pip install pillow tensorflow
```

Install the Update Module on the device:

Option A: the device is directly accessible via filesystem or SSH

Copy the `tflilte-demo` Update Module, (located at `mender/module/tflite-d3mo`) to `/usr/share/mender/modules/v3`, and make sure it is executable (`chmod a+x`)

Option B: the device is not directly accessible

Prepare the `tflite-demo` Update Module as a Mender Artifact for deploying it through Mender.
```
$ cd mender/module
$ ./tflite-um-artifact-gen --artifact-name tflite-demo-um --device-type debian-tensorflow
$ cd ../..
```
Upload and the deploy the Artifact through Hosted Mender to the device.

### Step 1: training

We will train a model based on the [Tensorflow Image classification example](https://www.tensorflow.org/tutorials/images/classification). This dataset provides a number of flower images, and trains a model for classification to the labels:
- `daisy`
- `dandelion`
- `roses`
- `sunflowers`
- `tulips`

The `training.py` script is closely based on the tutorial, and generates two artifacts:
- `model.tflite`: the trained model
- `class_names.json`: the list of labels for inference

To do so, run the script:
```
$ python3 training.py
```

The resulting artifacts are placed in the `artifacts` directory by default:
```
$ ls -alh artifacts
total 31184
drwxr-xr-x@  4 josef  staff   128B Sep 20 10:36 .
drwxr-xr-x@ 12 josef  staff   384B Sep 20 10:33 ..
-rw-r--r--@  1 josef  staff    55B Sep 20 10:33 class_names.json
-rw-r--r--@  1 josef  staff    15M Sep 20 10:36 model.tflite
```

### Step 1b (optional): local inference

If you would like to see the generated model in action, you can run the `inference.py` script locally. It is the exact same script as we will deploy through Mender in a second.

To test the training artifacts, run:
```
$ python3 inference.py
```

An example output could be:
```
using artifacts directory artifacts
classification will use the labels ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
The image belongs to class sunflowers with 99.76% probability
```

### Step 2: create artifact and upload Artifact
```
./mender/module-artifact-gen/tflite-demo-artifact-gen -n tflite -t debian-tensorflow
```
This generates the Mender Artifact file `tflite.mender`. Upload this Artifact to your Hosted Mender tenant, either through the web interface or using `mender-cli`.

### Step 3: deploy artifact and verify

Create a deployment for the the demo device using the `tflite` release. The recommended way is
1. select the device to show details
2. hover over the "Device actions" menu, then choose "Create deployment for this device"
3. in the "Select a Release" box, pick `tflite`
4. click "Create Deployment"

To verify the successful inference after deployment, open a terminal session on the demo device (you can also use the Troubleshoot-AddOn remote terminal for that), and run:
```
sudo journalctl -u mender-client | grep tflite
```

An example output might be:
```
Sep 19 17:18:43 jh-bullseye-mender mender[595]: time="2023-09-19T17:18:43+02:00" level=info msg="Validating the Update Info: https://hosted-mender-artifacts.s3.amazonaws.com/626a6e59bb8795f14e4f3337/ee0a6026-6eb1-48b0-bd4e-b5894a31f8b7?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQWI25QR6NDALMYE2%2F20230919%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230919T151843Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22tflite.mender%22&response-content-type=application%2Fvnd.mender-artifact&x-id=GetObject&X-Amz-Signature=e98104aba29ccf4cb7825383e82a1594838545fd750b3560d313fd6ff3e78a67 [name: tflite; devices: [jh-bullseye-mender]]"
Sep 19 17:18:49 jh-bullseye-mender mender[595]: time="2023-09-19T17:18:49+02:00" level=info msg="Output (stdout) from command \"/usr/share/mender/modules/v3/tflite-demo\": using artifacts directory /var/lib/mender/modules/v3/payloads/0000/tree/files/"
Sep 19 17:18:49 jh-bullseye-mender mender[595]: time="2023-09-19T17:18:49+02:00" level=info msg="Output (stdout) from command \"/usr/share/mender/modules/v3/tflite-demo\": classification will use the labels ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']"
Sep 19 17:18:49 jh-bullseye-mender mender[595]: time="2023-09-19T17:18:49+02:00" level=info msg="Output (stdout) from command \"/usr/share/mender/modules/v3/tflite-demo\": The image belongs to class sunflowers with 98.87% probability\n
```

Looking at the log, you can see that the Update Module successfully ran inference on the device, using the model that was previously trained on the development host. The timestamps show that the whole process took no longer than 6 seconds in this example, further highlighting the asymmetry between training and inference.

## Conclusion

AI/ML offers incredible possibilities to provide advanced functionality in edge devices. A key characteristic is the highly asymmetric computation requirements between the training of a model, and using resulting model for inference. Given the resource contraints on connected edge devices, the vast majority of training (colloquially, the "learning") is done on high performance servers in the cloud, generating the models to be deployed. An example technology here is Tensorflow Lite, which was designed as a library to use Tensorflow-based models on mobile and edge devices.
The distribution and rapid updates of such models at scale will be a core requirement for the next generation of devices using AI. Mender provides a robust and highly customizable solution to this challenge. By using a Update Module as shown in the demo, you can manage and deploy your models to your fleet, including advanced versioning schemes and custom hooks to interact with the application software.