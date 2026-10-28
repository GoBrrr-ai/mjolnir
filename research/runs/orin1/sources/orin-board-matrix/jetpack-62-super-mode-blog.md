English 中文
# NVIDIA JetPack 6.2 Brings Super Mode to NVIDIA Jetson Orin Nano and Jetson Orin NX Modules
Jan 16, 2025 
By [Shashank Maheshwari](https://developer.nvidia.com/blog/author/shmaheshwari/ "Posts by Shashank Maheshwari") and [Chen Su](https://developer.nvidia.com/blog/author/chesu/ "Posts by Chen Su")
Like 


## AI-Generated Summary
  * [Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) and [Jetson Orin NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) production modules gain up to 2x higher generative AI model performance with JetPack 6.2 Super Mode.
  * New reference power modes include 25W and uncapped MAXN SUPER for Orin Nano, and 40W plus MAXN SUPER for Orin NX, unlocking higher GPU, DLA, and CPU clock frequencies.
  * Benchmarks show LLMs such as Llama 3.1 8B and Qwen 2.5 7B achieve 1.1x to 1.6x token-per-second gains, while VLMs and ViTs also see measurable throughput improvements across all module variants.
  * An updated Power Estimator Tool and Thermal Design Guide help developers create custom power profiles that balance performance with thermal constraints.
  * JetPack 5.1.5 will extend Super Mode support to developers still on the JetPack 5 release line, and the Jetson Orin product lifecycle has been extended through 2032.


### Next Steps
  * Flash JetPack 6.2 using the [JetPack SDK](https://developer.nvidia.com/embedded/jetpack) to enable Super Mode on your device.
  * Explore generative AI tutorials on the [Jetson AI Lab](https://www.jetson-ai-lab.com/) for hands-on deployment examples.
  * Download updated datasheets and design guides from the [Jetson Download Center](https://developer.nvidia.com/embedded/downloads) for hardware integration.


Powered by NVIDIA Nemotron. AI-generated content may summarize information incompletely. Verify important information. [Learn more](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/)
The introduction of the [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) sparked a new age of [generative AI](https://www.nvidia.com/en-us/glossary/generative-ai/) for small edge devices. The new [Super Mode](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/) delivered an unprecedented generative AI performance boost of up to 1.7x on the developer kit, making it the most affordable generative AI supercomputer. 
JetPack 6.2 is now available to support Super Mode for [Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) and [Jetson Orin NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) production modules, delivering up to 2x higher generative AI model performance. Now you can unlock greater value and lower total cost of ownership for new and existing robotics and edge AI applications.
This post discusses the details of Super Mode, including new power modes, benchmarks for popular generative AI models on the Jetson Orin Nano and Orin NX modules, updates to the documentation, and insights into NPN partners supporting Super Mode.
## New reference power modes on the Jetson Orin Nano and Jetson Orin NX series 
JetPack 6.2 enables the power boost on the Jetson Orin Nano and Jetson Orin NX series by unlocking higher frequencies on the GPU, DLA memory, and CPU clocks.  
| **MODULE**  |  **EXISTING REFERENCE POWER MODES****(available with existing flashing configs)**  |  **NEW POWER MODES****(available only with new flashing configs)**  |  
| --- | --- | --- |  
| NVIDIA Jetson Orin Nano 4GB  | 7W, 10W  |  **10W, 25W,****MAXN** **SUPER**  |  
| NVIDIA Jetson Orin Nano 8GB  | 7W, 15W  |  **15W, 25W,****MAXN** **SUPER**  |  
| NVIDIA Jetson Orin NX 8GB  | 10W, 15W, 20W,MAXN  |  **10W, 15W, 20W, 40W,****MAXN** **SUPER**  |  
| NVIDIA Jetson Orin NX 16GB  | 10W, 15W, 25W,MAXN  |  **10W, 15W, 25W, 40W,****MAXN** **SUPER**  |  
_Table 1. New reference power modes on the Jetson Orin Nano and Jetson Orin NX modules_
Jetson Orin Nano modules now have a 25W mode and a new uncapped MAXN SUPER mode. Similarly, Jetson Orin NX modules can now use a new higher 40W reference power mode as well as an uncapped MAXN SUPER mode.
The MAXN SUPER is an uncapped power mode that enables the highest number of cores and clock frequency for CPU, GPU, DLA, PVA, and SOC engines. If the total module power exceeds the thermal design power (TDP) budget in this mode, the module is throttled to lower frequency, which delivers lower performance while staying within the thermal budget. 
We strongly recommend building your own custom power mode to find the right balance between power consumption or thermal stability and performance for your application and needs.
Table 2 compares the detailed specifications of Jetson Orin Nano 4GB and 8GB and Jetson Orin NX 8GB and 16GB in their original and Super Mode.  
| **ORIN NANO 4GB​**  |  **ORIN NANO 4GB****(SUPER) ​**  | **ORIN NANO 8GB​**  |  **ORIN NANO 8GB ​****(SUPER) ​**  | **ORIN NX 8GB​**  |  **ORIN NX 8GB​****(SUPER)​**  | **ORIN NX 16GB​**  |  **ORIN NX 16GB​****(SUPER)​**  |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
|  **PEAK AI PERF** ​**INT8** ​  | 20 TOPS (Sparse)​10 TOPS (Dense)​  |  **34 TOPS (Sparse)** ​**17 TOPS (Dense)** ​  | 40 TOPS (Sparse)​20 TOPS (Dense)​  |  **67 TOPS (Sparse)** ​**33 TOPS (Dense)** ​  | 70 TOPS (Sparse)​35 TOPS (Dense)​  |  **117 TOPS (Sparse)** ​**58 TOPS (Dense)** ​  | 100 TOPS (Sparse)​50 TOPS (Dense)​  |  **157 TOPS (Sparse)** ​**78 TOPS (Dense)** ​  |  
| **NVIDIA AMPERE GPU**  | 512 CUDA Cores ​16 Tensor Cores​625 MHz 20/10 INT8 TOPs (S/D)5 FP16 TFLOPs  | 512 CUDA Cores ​16 Tensor Cores ​**1020 MHz****34/17 INT8 TOPs (S/D)****8.5 FP16 TFLOPs**  | 1024 CUDA Cores​32 Tensor Cores​625 MHz 40/20 INT8 TOPs (S/D)10 FP16 TFLOPs  | 1024 CUDA Cores​32 Tensor Cores​**1020 MHz****67/33 INT8 TOPs (S/D)****17 FP16 TFLOPs**  | 1024 CUDA Cores​32 Tensor Cores765 MHz 50/25 INT8 TOPs (S/D) ​13 FP16 TFLOPs  | 1024 CUDA Cores​32 Tensor Cores​**1173 MHz****77/38 INT8 TOPs (S/D)****19 FP16 TFLOPs**  | 1024 CUDA Cores​32 Tensor Cores​918 MHz60/30 INT8 TOPs (S/D) 15 FP16 TFLOPs  | 1024 CUDA Cores​32 Tensor Cores​**1173 MHz****77/38 INT8 TOPs (S/D)****19 FP16 TFLOPs**  |  
|  **CPU** ​  | 6X A78​1.5 GHz​  | 6X A78​**1.7 GHz** ​  | 6X A78​1.5 GHz​  | 6X A78​**1.7 GHz** ​  | 6X A78​2.0 GHz​  | 6X A78​2.0 GHz​  | 8X A78​2.0 GHz​  | 8X A78​2.0 GHz​  |  
|  **DLA (S/D)** ​  | NA​  | NA​  | NA​  | NA​  | 20/10 INT8 TOPs​  |  **40/20 INT8 TOPs** ​  | 40/20 INT8 TOPs​  |  **80/40 INT8 TOPs** ​  |  
|  **DRAM BW** ​  | 34 GB/s​  |  **51 GB/s** ​  | 68 GB/s​  |  **102 GB/s** ​  | 102 GB/s​  | 102 GB/s​  | 102 GB/s​  | 102 GB/s​  |  
|  **REFERENCE MODULE POWER** ​  | 7W |10W​  | 7W |**25W** |​**MAXN SUPER**  | 7W |15W​  |  **15W** |**25W** | **MAXN SUPER​**  | 10W |15W |20W | MAXN​  | 10W |15W |20W |**40W** | **MAXN SUPER​**  | 10W |15W |25W​ | MAXN  | 10W |15W |25W |**40W** ​ | **MAXN SUPER**  |  
_Table 2. Original specs for Jetson Orin Nano and Jetson Orin NX and the specs in Super Mode_
While using the new power modes, ensure that your product’s existing or new thermal design can accommodate the new specifications with the power modes. For more information, see the updated [Thermal Design Guide](https://developer.nvidia.com/downloads/jetson-orin-nx-orin-nano-series-thermal-design-guide).
## Updated Power Estimator Tool 
The Power Estimator Tool is a powerful tool provided by NVIDIA to create custom power profiles and `nvpmodel` configuration files by modifying system parameters such as the cores, maximum frequency and load levels on the GPU, CPU, DLA, and so on. The tool provides an estimated power consumption with various settings and can be used to create optimal parameters settings to get the desired balance between performance and power consumption. 
We have updated the Power Estimator Tool with Super Mode. We strongly recommend that you use the Power Estimator Tool and verify in practice before deploying with high-performance applications.
## Boost performance on popular generative AI models 
With the introduction of Super Mode with JetPack 6.2, the Jetson Orin Nano and Jetson Orin NX modules deliver up to a 2x inference performance boost. We benchmarked the most popular large language models (LLMs), vision language models (VLMs) and vision transformers (ViTs). 
### Large language models 
The following chart and tables show the Super Mode performance benchmark for popular LLMs such as Llama3.1 8B, Qwen2.5 7B, and Gemma2 2B.
_Figure 1. Performance improvements for LLMs using Super Mode_
DNR means that memory on the module was not sufficient to run the specific model. Model performance will be influenced by throttling behavior.
In the following tables,LLM generation performance (tokens per second) was measured with INT4 quantization using MLC API.
Table 3 shows the LLM performance gain on Jetson Orin Nano 4GB with JetPack 6.2.  
| **Model**  | **Orin Nano 8GB (original)**  | **Orin Nano 8GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| Gemma 2 2B  | 11.40  | 18.60  | 1.64  |  
| SmolLM2 1.7B  | 23.00  | 35.80  | 1.56  |  
_Table 3. Benchmark performance in tokens/sec for popular LLMs on Jetson Orin Nano 4GB_
Table 4 shows the LLM performance gain on Jetson Orin Nano 8GB with JetPack 6.2.  
| **Model**  | **Orin Nano 8GB (original)**  | **Orin Nano 8GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| Llama 3.1 8B  | 14.00  | 19.10  | 1.37  |  
| Llama 3.2 3B  | 27.70  | 43.10  | 1.55  |  
| Qwen 2.5 7B  | 14.20  | 21.80  | 1.53  |  
| Gemma 2 2B  | 21.5  | 35.0  | 1.63  |  
| Gemma 2 9B  | 7.20  | 9.20  | 1.28  |  
| Phi-3.5 3.8B  | 24.70  | 38.10  | 1.54  |  
| SmolLM2 1.7B  | 41.00  | 64.50  | 1.57  |  
_Table 4. Benchmark performance in tokens/sec for popular LLMs on Jetson Orin Nano 8GB_
Table 5 shows the LLM performance gain on Jetson Orin NX 8GB with JetPack 6.2.  
| **Model**  | **Orin NX 8GB (original)**  | **Orin NX 8GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| Llama 3.1 8B  | 15.90  | 23.10  | 1.46  |  
| Llama 3.2 3B  | 34.50  | 46.50  | 1.35  |  
| Qwen 2.5 7B  | 17.10  | 23.80  | 1.39  |  
| Gemma 2 2B  | 26.60  | 39.30  | 1.48  |  
| Gemma 2 9B  | 8.80  | 13.38  | 1.52  |  
| Phi-3.5 3.8B  | 30.80  | 41.30  | 1.34  |  
| SmolLM2 1.7B  | 51.50  | 69.80  | 1.35  |  
_Table 5. Benchmark performance in tokens/sec for popular LLMs on Jetson Orin NX 8GB_
Table 6 shows the LLM performance gain on Jetson Orin NX 16GB with JetPack 6.2.  
| **Model**  | **Orin NX 16GB (original)**  | **Orin NX 16GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| Llama 3.1 8B  | 20.50  | 22.80  | 1.11  |  
| Llama 3.2 3B  | 40.40  | 45.80  | 1.13  |  
| Qwen 2.5 7B  | 20.80  | 23.50  | 1.13  |  
| Gemma 2 2B  | 31.60  | 39.00  | 1.23  |  
| Gemma 2 9B  | 10.56  | 13.26  | 1.26  |  
| Phi-3.5 3.8B  | 35.90  | 40.90  | 1.14  |  
| SmolLM2 1.7B  | 59.50  | 68.80  | 1.16  |  
_Table 6. Benchmark performance in tokens/sec for popular LLMs on Jetson Orin NX and Orin Nano modules_
### Vision language models 
The following chart and tables show the Super Mode performance benchmark for popular VLMs such as VILA1.5 8B, LLAVA1.6 7B, and Qwen2 VL 2B.
_Figure 2. Performance improvements of VLMs when run using Super Mode_
DNR means that memory on the module was not sufficient to run the specific model. Model performance will be influenced by throttling behavior.
Table 7 shows the VLM performance gain on Jetson Orin Nano 4GB with JetPack 6.2.  
| **Model**  | **Orin Nano 4GB (original)**  | **Orin Nano 4GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| PaliGemma2 3B  | 7.2  | 11.2  | 1.56  |  
_Table 7. Benchmark performance in tokens/sec for popular VLMs on Jetson Orin Nano 4GB_
Table 8 shows the VLM performance gain on Jetson Orin Nano 8GB with JetPack 6.2.  
| **Model**  | **Orin NX 16GB (original)**  | **Orin NX 16GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| VILA 1.5 3B  | 0.7  | 1.1  | 1.51  |  
| VILA 1.5 8B  | 0.6  | 0.8  | 1.45  |  
| LLAVA 1.6 7B  | 0.4  | 0.6  | 1.38  |  
| Qwen2 VL 2B  | 2.8  | 4.4  | 1.57  |  
| InternVL2.5 4B  | 2.5  | 5.1  | 2.04  |  
| PaliGemma2 3B  | 13.7  | 21.6  | 1.58  |  
| SmolVLM 2B  | 8.1  | 12.9  | 1.59  |  
_Table 8. Benchmark performance in tokens/sec for popular VLMs on Jetson Orin Nano 8GB_
Table 9 shows the VLM performance gain on Jetson Orin NX 8GB with JetPack 6.2.  
| **Model**  | **Orin NX 16GB (original)**  | **Orin NX 16GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| VILA 1.5 3B  | 0.8  | 1  | 1.25  |  
| VILA 1.5 8B  | 0.7  | 1.04  | 1.50  |  
| LLAVA 1.6 7B  | 0.5  | 1.2  | 2.54  |  
| Qwen2 VL 2B  | 3.4  | 4.8  | 1.41  |  
| InternVL2.5 4B  | 3  | 4.1  | 1.37  |  
| PaliGemma2 3B  | 17.1  | 23.9  | 1.40  |  
| SmolVLM 2B  | 9.7  | 14.4  | 1.48  |  
_Table 9. Benchmark performance in tokens/sec for popular VLMs on Jetson Orin NX 16GB_
Table 10 shows the VLM performance gain on Jetson Orin NX 16GB with JetPack 6.2.  
| **Model**  | **Orin NX 16GB (original)**  | **Orin NX 16GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| VILA 1.5 3B  | 1  | 1.3  | 1.23  |  
| VILA 1.5 8B  | 0.8  | 1  | 1.25  |  
| LLAVA 1.6 7B  | 0.6  | 0.7  | 1.07  |  
| Qwen2 VL 2B  | 4  | 4.8  | 1.20  |  
| InternVL2.5 4B  | 2.8  | 4.4  | 1.57  |  
| PaliGemma2 3B  | 20  | 23.8  | 1.19  |  
| SmolVLM 2B  | 11.7  | 14.3  | 1.22  |  
_Table 10. Benchmark performance in tokens/sec for popular VLMs on Jetson Orin NX and Orin Nano modules_
All VILA and LLAVA models were run with INT4 precision using MLC while the rest of the models were run in FP4 precision with Hugging Face Transformers.
### Vision transformers 
The following chart and tables show the Super Mode performance benchmark for popular ViTs such as CLIP, DINO, and SAM2.
_Figure 3. Performance improvements of ViTs when run using Super Mode_
DNR means that memory on the module was not sufficient to run the specific model. Model performance will be influenced by throttling behavior.
Table 11 shows the ViT performance gain on Jetson Orin Nano 4GB with JetPack 6.2.  
| **Model**  | **Orin Nano 4GB (original)**  | **Orin Nano 4GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| clip-vit-base-patch32  | 126.8  | 189.5  | 1.49  |  
| clip-vit-base-patch16  | 63.2  | 112.4  | 1.78  |  
| DINOv2-base-patch14  | 49.3  | 79.3  | 1.61  |  
| SAM2 base  | 2.5  | 3.8  | 1.54  |  
| vit-base-patch16-224  | 62.4  | 103.3  | 1.66  |  
_Table 11. Benchmark performance in tokens/sec for popular ViTs on Jetson Orin Nano 4GB_
Table 12 shows the ViT performance gain on Jetson Orin Nano 8GB with JetPack 6.2.  
| **Model**  | **Orin Nano 8GB (original)**  | **Orin Nano 8GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| clip-vit-base-patch32  | 196  | 314  | 1.60  |  
| clip-vit-base-patch16  | 95  | 161  | 1.69  |  
| DINOv2-base-patch14  | 75  | 126  | 1.68  |  
| SAM2 base  | 4.4  | 6.3  | 1.43  |  
| Grounding DINO  | 4.1  | 6.2  | 1.52  |  
| vit-base-patch16-224  | 98  | 158  | 1.61  |  
| vit-base-patch32-224  | 171  | 273  | 1.60  |  
_Table 12. Benchmark performance in tokens/sec for popular ViTs on Jetson Orin Nano 8GB_
Table 13 shows the ViT performance gain on Jetson Orin NX 8GB with JetPack 6.2.  
| **Model**  | **Orin NX 8GB (original)**  | **Orin NX 8GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| clip-vit-base-patch32  | 234.0  | 361.1  | 1.54  |  
| clip-vit-base-patch16  | 101.7  | 204.3  | 2.01  |  
| DINOv2-base-patch14  | 81.4  | 160.3  | 1.97  |  
| SAM2 base  | 3.9  | 7.4  | 1.92  |  
| Grounding DINO  | 4.2  | 7.4  | 1.75  |  
| vit-base-patch16-224  | 98.6  | 192.5  | 1.95  |  
| vit-base-patch32-224  | 193.1  | 313.5  | 1.62  |  
_Table 13. Benchmark performance in tokens/sec for popular ViTs on Jetson Orin NX 8GB_
Table 14 shows the ViT performance gain on Jetson Orin NX 16GB with JetPack 6.2.  
| **Model**  | **Orin NX 16GB (original)**  | **Orin NX 16GB (Super Mode)**  | **Perf Gain (x)**  |  
| --- | --- | --- | --- |  
| clip-vit-base-patch32  | 323.2  | 356.7  | 1.10  |  
| clip-vit-base-patch16  | 163.5  | 193.6  | 1.18  |  
| DINOv2-base-patch14  | 127.5  | 159.8  | 1.25  |  
| SAM2 base  | 6.2  | 7.3  | 1.18  |  
| Grounding DINO  | 6.2  | 7.2  | 1.16  |  
| vit-base-patch16-224  | 158.6  | 190.2  | 1.20  |  
| vit-base-patch32-224  | 281.2  | 309.5  | 1.10  |  
_Table 14. Benchmark performance in frames/sec for popular ViTs on Jetson Orin NX 16GB_
All ViT models were run with FP16 precision using NVIDIA TensorRT and measurements are in FPS.
## Getting started on NVIDIA Jetson Orin Nano and Jetson Orin NX with JetPack 6.2 
The NVIDIA Jetson ecosystem provides various ways for you to flash the developer kit and production modules with the JetPack image. 
To install JetPack 6.2 on the Jetson Orin Nano Developer Kit or the modules, use one of the following methods: 
  * **SD-Card image** from the [JetPack SDK](https://developer.nvidia.com/embedded/jetpack) page.
  * **SDK Manager** for flashing per the [download and install instructions](https://developer.nvidia.com/sdk-manager).
  * **Flashing script** for bootloader, kernel, and root file system, per [Flashing Support](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/FlashingSupport.html#flashing-support). 


## New flashing configuration 
The new power modes are only available with the new flashing configuration. The default flashing configuration has not changed. To enable the new power modes, you must use the new flashing configuration while flashing. 
Here’s the new flashing configuration to be used with flashing:  
| `jetson-orin-nano-devkit-super.conf` |  
| --- |  
After flashing or updating to JetPack 6.2, run the following command to start the newly available Super Mode.
MAXN SUPER mode on Jetson Orin Nano Modules:  
|  `sudo` `nvpmodel -m 2`  |  
| --- |  
MAXN SUPER mode on Jetson Orin NX Modules:  
|  `sudo` `nvpmodel -m 0`  |  
| --- |  
You can also select the MAXN SUPER and other power modes from the power mode menu at the top-right corner of the page.
_Figure 4. Power mode selection menu_ s
## Jetson AI Lab 
The [Jetson AI Lab](https://www.jetson-ai-lab.com/) is the NVIDIA hub for exploring and experimenting with generative AI technologies optimized for edge devices. It supports developers and provides a collaborative community with nearly 50 tutorials, prebuilt containers, and resources for deploying on-device LLMs, SLMs, VLMs, diffusion policies, and speech models using optimized inferencing infrastructures. 
By simplifying access to cutting-edge AI tools, the lab empowers developers of all levels to innovate and deploy generative AI locally, advancing open-source edge AI and robot learning.
Dive into generative AI with ease using these easy-to-follow tutorials for your developer kit powered by JetPack 6.2:
  * [Ollama with Open Web UI](https://www.jetson-ai-lab.com/tutorial_ollama.html): Introductory LLM application example
  * [NanoOWL](https://www.jetson-ai-lab.com/vit/tutorial_nanoowl.html): Interactive demo with cutting-edge ViT model
  * [LeRobot](https://www.jetson-ai-lab.com/lerobot.html): Physical AI example with HuggingFace LeRobot


## Updated documentation: Datasheets and design guides 
With the newer performance boost, the following resources have been updated and can be downloaded from the [Jetson Download Center](https://developer.nvidia.com/embedded/downloads): 
  * [Jetson Orin NX Module datasheet](https://developer.nvidia.com/downloads/jetson-orin-nx-module-series-data-sheet)
  * [Jetson Orin Nano Module datasheet](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_ds)
  * [Jetson Orin NX Series and Jetson Orin Nano Series Design Guide](https://developer.nvidia.com/downloads/jetson-orin-nx-series-nano-series-design-guide)


## Jetson ecosystem partners ready for Super Mode 
To support customer deployments, the Jetson ecosystem partners have enhanced their solutions to support this boosted performance.  
| **Category**  | **Jetson ecosystem partners**  |  
| --- | --- |  
| ISV solutions  |  [DeepEdge](https://www.lluminaai.com/post/advanced-ai-understanding-at-the-edge-implementing-vision-language-models-on-jetson-nano)[Edge Impulse](https://www.youtube.com/watch?v=ntNAbfKm_Xs)[RidgeRun](https://developer.ridgerun.com/wiki/index.php/Exploring_NVIDIA_Jetson_Orin_Nano_Super_Mode_performance_using_Generative_AI)[Ultralytics](https://www.ultralytics.com/blog/ultralytics-yolo11-on-nvidia-jetson-orin-nano-super-fast-and-efficient)  |  
| Hardware system partners  |  [AAEON](https://www.aaeon.com/en/product/list/aaeon-nvidia-ai-solutions)[Advantech](https://www.advantech.com/en/resources/news/advantech-mic-ai-computer-system-powered-by-nvidia-jetson-orin-nano-series-supports-super-mode-for-exceptional-generative-ai-and-edge-ai-performance)[Aetina](https://www.aetina.com/about-news-detail.php?i=1048)[AIMobile](https://www.aimobile.com.tw/blog/supermode)[ASUSTek](https://iot.asus.com/resources/news/jetson-edge-ai-super-mode/)[Axiomtek](https://www.axiomtek.com/Default.aspx?MenuId=News&FunctionId=NewsView&ItemId=16431)[Connect Tech](https://connecttech.com/jetson-orin-nx-nano-super-mode/)[Seeed Studio](https://www.seeedstudio.com/blog/2024/12/20/introducing-the-nvidia-jetson-orin-nano-super-developer-kit-redefining-generative-ai-for-small-edge-devices/?srsltid=AfmBOooewxjCDj8sH5CrXlrzBgoRALfwz0MIdeHIa1NV0yCCEszDdl17)[Syslogic](https://www.syslogic.com/all-products?technology=NVIDIA%C2%AE+Jetson+Orin%E2%84%A2+Nano)[Vecow](https://www.vecow.com/dispPageBox/vecow/VecowCP.aspx?ddsPageID=SOLUTIONS_NVIDIA)[Yuan High-Tech](https://www.yuan.com.tw/newscontent/161)  |  
| Thermal solutions  |  [Advanced Thermal Solutions](https://www.qats.com/eShop.aspx?q=Device%20Specific%20-%20NVIDIA)[Frore Systems](https://www.froresystems.com/products/airjet-r-pak)  |  
## NVIDIA Jetson Orin lifecycle and roadmap 
Due to the growing customer demand for Jetson Orin, NVIDIA [recently announced](https://developer.nvidia.com/embedded/lifecycle) the extension of the product lifecycle of Jetson Orin through 2032. With this performance boost, the Jetson Orin Nano and Orin NX series are the ideal platforms for both current and future models. 
The upcoming [JetPack 5.1.5 ](https://developer.nvidia.com/embedded/develop/roadmap)will also enable Super Mode for the Jetson Orin NX and Jetson Orin Nano modules. Developers and customers who develop with JetPack 5 will benefit from the performance boost. 
_Figure 5. JetPack software roadmap_
## Boost your application performance with JetPack 6.2 
JetPack 6.2 is a groundbreaking release. It delivers an astonishing 2x boost in inference performance on existing Jetson modules, without any added cost. This upgrade is a must-have for Jetson developers and customers looking to supercharge their applications. Upgrade to JetPack 6.2 today and unleash the full potential of your Jetson platform.
_Stay up to date by subscribing to our_[newsletter](https://www.nvidia.com/en-us/industries/robotics/robotics-stay-informed/) _, and follow NVIDIA Robotics on_[LinkedIn](https://www.linkedin.com/showcase/nvidiarobotics), [Instagram](https://www.instagram.com/nvidiarobotics/) _,_[X](https://x.com/NVIDIARobotics) _, and_[Facebook](https://www.facebook.com/NVIDIARobotics) _. For more information, explore our_[documentation](https://docs.nvidia.com/) _or join the Robotics community on our_[developer forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/55) _,_[Discord](https://discord.gg/w9VvuYdq) _, and_[YouTube](https://www.youtube.com/@NVIDIADeveloper/) _channels._
Like 
## Tags
[Agentic AI / Generative AI](https://developer.nvidia.com/blog/category/generative-ai/) | [Robotics](https://developer.nvidia.com/blog/category/robotics/) | [Manufacturing](https://developer.nvidia.com/blog/recent-posts/?industry=Manufacturing) | [JetPack](https://developer.nvidia.com/blog/recent-posts/?products=JetPack) | [Jetson Orin](https://developer.nvidia.com/blog/recent-posts/?products=Jetson+Orin) | [Intermediate Technical](https://developer.nvidia.com/blog/recent-posts/?learning_levels=Intermediate+Technical) | [Benchmark](https://developer.nvidia.com/blog/recent-posts/?content_types=Benchmark) | [featured](https://developer.nvidia.com/blog/tag/featured/) | [LLMs](https://developer.nvidia.com/blog/tag/large-language-models/) | [ViTs](https://developer.nvidia.com/blog/tag/vits/) | [VLMs](https://developer.nvidia.com/blog/tag/vlms/)
##  About the Authors 
**About Shashank Maheshwari** Shashank Maheshwari is the product manager for Jetson and IGX software at NVIDIA. He holds an MBA from Duke University and a bachelor’s degree in Electronics from BITS Pilani. 
[ View all posts by Shashank Maheshwari ](https://developer.nvidia.com/blog/author/shmaheshwari/)
**About Chen Su** Chen Su is a senior technical product marketing manager at NVIDIA. He is passionate about collaborating with cross-functional teams and growing businesses fueled by IoT, AI, robotics, and edge computing. Bringing clarity and creativity to cutting-edge technology and tools, he drove dynamic messaging and engagements to help global developers and partners achieve faster time to value and outcomes. Prior to that, Chen had 10+ yrs work experience in R D, product management, and pre-sales consulting in industrial automation and enterprise software industries. Chen has a M.Sc. in engineering from the University of Pennsylvania and an MBA from the Berkeley Haas School of Business. 
[ View all posts by Chen Su ](https://developer.nvidia.com/blog/author/chesu/)
## Comments
Close Previous Next
NVIDIA and our third-party partners use cookies and other tools to collect and record information you provide as well as information about your interactions with our websites for performance improvement, analytics, and to assist in our marketing efforts. By clicking "Accept All", you consent to our use of cookies and other tools as described in our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/). You can manage your cookie settings by clicking on "Manage Settings". Please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information on our privacy practices.
We have detected the Global Privacy Control (GPC) signal and recorded your rejection of all optional cookies on this site for this browser. You can manage your cookie settings by clicking on "Manage Settings". Please see our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/) for more information and visit our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for details on our privacy practices.
Manage Settings
Reject Optional Accept All
Cookie Settings
We and our third-party partners (including social media, advertising, and analytics partners) use cookies and other tracking technologies to collect, store, monitor, and process certain information about you when you visit our website. The information collected might relate to you, your preferences, or your device. We use that information to make the site work and with your consent to analyze performance and traffic on our website, to provide a more personalized web experience, and assist in our marketing efforts.Click on the different category headings below to find out more and change the settings according to your preference. You cannot opt out of Required Cookies as they are deployed to ensure the proper functioning of our website (such as prompting the cookie banner and remembering your settings, etc.). By clicking "Save and Accept" at the bottom, you consent to the use of cookies and other tools as described in our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/) in accordance with your settings. For more information about our privacy practices, please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/). [More information](https://cookiepedia.co.uk/giving-consent-to-cookies)
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
