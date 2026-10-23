[ Skip to last reply ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539/20) [ Skip to top ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539/1)
[ Skip to main content ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539#main-container)
#  [ Cannot passthrough GPU to Kubernetes pod on the Jetson AGX Orin dev kit ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539)
[Robotics & Edge Computing](https://forums.developer.nvidia.com/c/robotics-edge-computing/55)[Jetson Systems](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/70)[Jetson AGX Orin](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-agx-orin/486)


You have selected **0** posts.
[ select all ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539)
[ cancel selecting ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539)
1.4k views  7 likes  11 links 
read  7 min 
[ Mar 2025  ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539/1 "Jump to the first post")
1 / 17 
Mar 2025 
[ Mar 25 ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539/20)
##  post by shahizat on Mar 27, 2025 
shahizat 
[ Mar 27 '25 ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539 "Post date")
Hello everyone,
Previously, I was able to passthrough the GPU using K3s and MicroK8s on my Nvidia Jetson AGX Orin dev kit, but I can no longer do so. After upgrading to a newer version of JetPack, it stopped working. I updated the NVIDIA Container Toolkit, but it did not help.
My JetPack version:

```
Package: nvidia-jetpack
Source: nvidia-jetpack (6.2)
Version: 6.2+b77
Architecture: arm64
Maintainer: NVIDIA Corporation

```

Running containers with GPU support works as expected in Docker, but not with containerd within k3s or microk8s.
Thanks in advance for your help.
  * [ Does Jetson Orin + k8s-device-plugin + MPS work? ](https://forums.developer.nvidia.com/t/does-jetson-orin-k8s-device-plugin-mps-work/344019/5)


1.4k views  7 likes  11 links 
read  7 min 
##  post by AastaLLL on Mar 31, 2025 
Moderator 
[ Mar 31 '25 ](https://forums.developer.nvidia.com/t/cannot-passthrough-gpu-to-kubernetes-pod-on-the-jetson-agx-orin-dev-kit/328539/3 "Post date")
Hi,
Could you try the commands in the below link:
Thanks.
##  post by shahizat on Mar 31, 2025 
##  post by AastaLLL on Apr 2, 2025 
##  post by shahizat on Apr 2, 2025 
##  post by whitesscott on Apr 2, 2025 
##  post by shahizat on Apr 5, 2025 
##  post by whitesscott on Apr 6, 2025 
##  post by shahizat on Apr 6, 2025 
##  post by whitesscott on Apr 7, 2025 
##  post by AastaLLL on Apr 7, 2025 
##  post by shahizat on Apr 7, 2025 
##  post by whitesscott on Apr 9, 2025 
9 days later 
##  post by shahizat on Apr 18, 2025 
##  post by whitesscott on Apr 20, 2025 
14 days later 
##  Closed on May 4, 2025 
11 months later 
##  post by AastaLLL on Mar 25 
Reply
###  Related topics   
Topic list, column headers with buttons are sortable.  
|  Topic   |  Replies   |  Views   |  Activity   |  
| --- | --- | --- | --- |  
|  [JetPack 6.3 containerd and kubernetes](https://forums.developer.nvidia.com/t/jetpack-6-3-containerd-and-kubernetes/296483) [Jetson AGX Orin](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-agx-orin/486)

 |  1.8k  |  
|  [Installation of single node microK8s on the NVIDIA Jetson AGX Orin Developer Kit](https://forums.developer.nvidia.com/t/installation-of-single-node-microk8s-on-the-nvidia-jetson-agx-orin-developer-kit/304171) [Jetson AGX Orin](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-agx-orin/486)

 |  
|  [Jetson Orin Nano Dev Board Pods Stuck in ContainersCreating State](https://forums.developer.nvidia.com/t/jetson-orin-nano-dev-board-pods-stuck-in-containerscreating-state/298883) [Jetson Orin Nano](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-orin-nano/632)

 |  
|  [Fail to install K3S and customize kernel on Jetson AGX ORIN](https://forums.developer.nvidia.com/t/fail-to-install-k3s-and-customize-kernel-on-jetson-agx-orin/320515) [Jetson AGX Orin](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-agx-orin/486)

 |  
|  [How to integrate two AGX Orins GPU resources in a Kubernetes (K8s) cluster for running a LLM inference?](https://forums.developer.nvidia.com/t/how-to-integrate-two-agx-orins-gpu-resources-in-a-kubernetes-k8s-cluster-for-running-a-llm-inference/311711) [Jetson AGX Orin](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-agx-orin/486)

 |  
|  [Cann’t use gpu resources in containerd with k8s and orin nano](https://forums.developer.nvidia.com/t/cannt-use-gpu-resources-in-containerd-with-k8s-and-orin-nano/302164) [Jetson Orin Nano](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-orin-nano/632)

 |  
|  [K3s on Jetson Nano 4GB with Jetson MATE | (NVIDIA Container Toolkit) upgrade](https://forums.developer.nvidia.com/t/k3s-on-jetson-nano-4gb-with-jetson-mate-nvidia-container-toolkit-upgrade/264799) [Jetson Nano](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-nano/76)

 |  4.1k  |  
|  [Jetson Orin Nano](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-orin-nano/632)

 |  1.5k  |  
|  [Choosing K3s vs MicroK8s for Nvidia Orin (+ Pi’s)](https://forums.developer.nvidia.com/t/choosing-k3s-vs-microk8s-for-nvidia-orin-pis/314377) [Jetson Orin Nano](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-orin-nano/632)

 |  
|  [How can I use nvidia gpu in kubernetes pod?](https://forums.developer.nvidia.com/t/how-can-i-use-nvidia-gpu-in-kubernetes-pod/224316) [Jetson Xavier NX](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-xavier-nx/258)

 |  9.8k  |  
NVIDIA and our third-party partners use cookies and other tools to collect and record information you provide as well as information about your interactions with our websites for performance improvement, analytics, and to assist in our marketing efforts. By clicking "Accept All", you consent to our use of cookies and other tools as described in our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/). You can manage your cookie settings by clicking on "Manage Settings". Please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information on our privacy practices.
We have detected the Global Privacy Control (GPC) signal and recorded your rejection of all optional cookies on this site for this browser. You can manage your cookie settings by clicking on "Manage Settings". Please see our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/) for more information and visit our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for details on our privacy practices.
Manage Settings
Reject Optional Accept All
Cookie Settings
We and our third-party partners (including social media, advertising, and analytics partners) use cookies and other tracking technologies to collect, store, monitor, and process certain information about you when you visit our website. The information collected might relate to you, your preferences, or your device. We use that information to make the site work and with your consent to analyze performance and traffic on our website, to provide a more personalized web experience, and assist in our marketing efforts.Click on the different category headings below to find out more and change the settings according to your preference. You cannot opt out of Required Cookies as they are deployed to ensure the proper functioning of our website (such as prompting the cookie banner and remembering your settings, etc.). By clicking "Save and Accept" at the bottom, you consent to the use of cookies and other tools as described in our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/) in accordance with your settings. For more information about our privacy practices, please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/).
Required Cookies
Always Active
These cookies enable core functionality such as security, network management, and accessibility. These cookies are required for the site to function and cannot be turned off.
Cookies Details
Performance Cookies
Performance Cookies
These cookies are used to provide quantitative measures of our website visitors, such as the number of times you visit, time on page, your mouse movements, scrolling, clicks and keystroke activity on the websites; other browsing, search, or product research behavior; and what brought you to our site. These cookies may store a unique ID so that our system will remember you when you return. Information collected with these cookies is used to measure and find ways to improve website performance.
Cookies Details
Personalization Cookies
Personalization Cookies
These cookies collect data about how you have interacted with our website to help us improve your web experience, such as which pages you have visited. These cookies may store a unique ID so that our system will remember you when you return. They may be set by us or by third party providers whose services we have added to our pages. These cookies enable us to provide enhanced website functionality and personalization as well as make the marketing messages we send to you more relevant to your interests. If you do not allow these cookies, then some or all of these services may not function properly.
Cookies Details
Advertising Cookies
Advertising Cookies
These cookies record your visit to our websites, the pages you have visited and the links you have followed to influence the advertisements that you see on other websites. These cookies and the information they collect may be managed by other companies, including our advertising partners, and may be used to build a profile of your interests and show you relevant advertising on other sites. We and our advertising partners will use this information to make our websites and the advertising displayed on it, more relevant to your interests.
Cookies Details
Back Button
Cookie List
Search Icon
Filter Icon
Clear
  * checkbox label label


Apply Cancel
Consent Leg.Interest
checkbox label label
checkbox label label
checkbox label label
Decline All Save and Accept
