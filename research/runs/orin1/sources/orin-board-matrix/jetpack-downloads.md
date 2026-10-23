* [Downloads](https://developer.nvidia.com/downloads)
* [Support](https://developer.nvidia.com/embedded/community/support-resources)
* [JetPack](https://developer.nvidia.com/embedded/develop/software)
# NVIDIA JetPack SDK Downloads and Notes  

## JetPack 7.2.1 with Jetson Linux 39.2.1
Access release notes, packages, and resources for NVIDIA JetPack 7.2.1 with Jetson Linux 39.2.1, CUDA 13.2.2, TensorRT 10.16.2, and Jetson Agentic AI resources, including Jetson agent skills and agentic workflow tools.  

Download the JetPack ISO and Yocto Images for developer kits here. By downloading and using the software, you agree to fully comply with the terms and conditions of the[NVIDIA Software License Agreement](https://developer.download.nvidia.com/compute/mathdx/License.txt).
[Download JetPack ISO Image](https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/iso/jetsoninstaller-r39.2.1-2026-08-07-18-30-47-arm64.iso)[Download SDK Manager](https://developer.nvidia.com/sdk-manager)
[Yocto for Jetson AGX Thor  
](https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/yocto/demo-image-reference-jetson-agx-thor-devkit.rootfs.tegraflash-tar.zst)[Yocto for Jetson AGX Orin  
](https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/yocto/demo-image-reference-p3737-0000-p3701-0005.rootfs.tegraflash-tar.zst)[Yocto for Jetson Orin Nano  
](https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/yocto/demo-image-reference-jetson-orin-nano-devkit-nvme.rootfs.tegraflash-tar.zst)
NOTE: See the[Jetson AGX Thor Developer Kit Quick Start Guide](https://docs.nvidia.com/jetson/agx-thor-devkit-4fed1671/user-guide/latest/quick_start.html)for flashing steps and the[Jetson Linux Developer Guide](http://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/)(Flashing Support section) for all flashing methods for developer kits and production modules**.** See the[Quick Start Guide with Yocto](https://www.jetson-ai-lab.com/tutorials/yocto-on-jetson/)for flashing steps on the dev-kits.
NVIDIA JetPack 7.2.1 with Jetson Linux 39.2.1. This section includes highlights of what's new, supported Jetson hardware platforms, and a complete list of included library and SDK versions.
NOTE: For full details, refer to[Jetson Linux release notes](https://docs.nvidia.com/jetson/archives/r39.2.1/ReleaseNotes/Jetson_Linux_Release_Notes_r39.2.1.pdf).
Jetson Linux 39.2.1 is the latest Jetson Linux release that supports NVIDIA JetPack 7.2.1. The following are the highlights for this release:
Support for Jetson T3000 emulation
ISO now flashes the Jetson Orin Nano Developer Kit with Super Mode flashing configuration by default.
Support for PCN211461 and PCN211462.
**BLOG POST** : Learn more about the key features in this release in the supporting blog post[here](https://developer.nvidia.com/blog/nvidia-jetpack-7-2-1-adds-agentic-video-skills-and-t3000-emulation/).
With JetPack 7 series, Jetson software aligns with the Server Base System Architecture (SBSA), positioning Jetson Thor alongside industry-standard ARM server design. SBSA standardizes critical hardware and firmware interfaces, delivering stronger OS support, simpler software portability, and smoother enterprise integration. Building on this foundation, Jetson Thor now supports a unified NVIDIA® CUDA® 13.2.2 installation across all Arm targets, streamlining development, reducing fragmentation, and ensuring consistency from server-class systems to Jetson Thor.
There is no more SD Card image for Jetson Orin Nano Developer Kit. Use the unified ISO image to flash the developer kit using a USB stick. Follow the instructions on[Getting Started Guide](https://docs.nvidia.com/jetson/agx-thor-devkit-4fed1671/user-guide/latest/quick_start.html).[](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/FlashingSupport.html)
If you’re re-installing JetPack 7.2.1 using ISO on an already installed system, please carefully follow the instructions in the[Getting Started Guide](https://docs.nvidia.com/jetson/agx-thor-devkit-4fed1671/user-guide/latest/quick_start.html).[](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/FlashingSupport.html)
The ISO image-based flashing now uses Super flashing configuration on Jetson Orin Nano Developer Kit by default.[](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/FlashingSupport.html)
### Supported Hardware
Jetson AGX Thor Developer Kit  
Jetson T5000  
Jetson T4000  
Jetson Orin Family
### Supported Libraries and SDK versions
| Jetson Linux  | L4T Ubuntu 24.04  | Other Supported Linux Distros  | Canonical Ubuntu 24.04 for Jetson  | Not supported  | Supported SDKs and Tools  | Jetson Platform Services
### Jetson Linux Downloads and Documents  

NVIDIA Jetson Linux Driver Package is the board support package for Jetson. It includes Linux Kernel, UEFI bootloader, NVIDIA drivers, flashing utilities, sample filesystem based on Ubuntu, and more for the Jetson platform.
#### Jetson Linux 39.2.1 (8/11/2026)
 | [Jetson Linux API Reference](https://developer.nvidia.com/downloads/embedded/L4T/r39_Release_v2.1/release/Jetson_Multimedia_API_R39.2.1_aarch64.tbz2)  | [OE4T for Jetson](https://github.com/oe4t)  | [Jetson AGX Thor Dev-Kit Image SBOM](https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/yocto/demo-image-reference-jetson-agx-thor-devkit-sbom.tar.gz)  | [Jetson AGX Orin Dev-Kit Image SBOM](https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/yocto/demo-image-reference-p3737-0000-p3701-0005-sbom.tar.gz)  | [Jetson Orin Nano Dev-Kit Image SBOM](https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/yocto/demo-image-reference-jetson-orin-nano-devkit-nvme-sbom.tar.gz)  | [CUDA Tools](https://developer.nvidia.com/cuda-toolkit)  | [Jetson Device-side Skills](https://github.com/NVIDIA-AI-IOT/jetson-device-skills)  | [Jetson BSP Skills](https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills)  | [Jetson Linux Release Notes](https://docs.nvidia.com/jetson/archives/r39.2.1/ReleaseNotes/Jetson_Linux_Release_Notes_r39.2.1.pdf)  | [Jetson Linux Developer Guide](http://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/)  | [Jetson AGX Thor Developer Kit User Guide](https://docs.nvidia.com/jetson/agx-thor-devkit-4fed1671/user-guide/latest/quick_start.html)  | Supported Software  | [Jetpack Customer IQ Migration Tools](https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/jetson_customer_iq_migration_tools_r39.2.1_jp_7.2.1_ga.zip)  | JetPack 7.2.1/Jetson Linux 39.2.1  | Jetpack Customer IQ Migration Tools for Jetpack 7.2.1 with the hotfix for camera For older versions of JetPack or Jetson Linux, please visit the[JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive)or[Jetson Linux Archive](https://developer.nvidia.com/embedded/jetson-linux-archive).
