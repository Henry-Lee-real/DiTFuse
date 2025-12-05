# Environment

DiTFuse is built on top of the **OmniGen** framework.  
Before running this repository, please follow the official environment setup instructions provided by OmniGen:

- OmniGen Repository: [`VectorSpaceLab/OmniGen`](https://github.com/VectorSpaceLab/OmniGen)

Once the OmniGen environment and dependencies are properly installed, you can directly run the scripts provided in this project.

hint: we keep the requirement.txt in our repo.

# Model Weights

DiTFuse uses OmniGen as the backbone model and loads task-specific LoRA weights for semantic and instruction-aware fusion.

- **Base Model**  
  We adopt the pretrained Diffusion Transformer released by OmniGen:  
  👉 [`Shitao/OmniGen-v1`](https://huggingface.co/Shitao/OmniGen-v1)

- **Fine-tuned Weights (DiTFuse LoRA)**  
  Our semantic-enhanced and instruction-controllable LoRA weights are available at:  
  👉 [`lijiayangCS/DiTFuse`](https://huggingface.co/lijiayangCS/DiTFuse)

To reproduce the results in the paper, please load **OmniGen-v1** as the base checkpoint and merge the **DiTFuse LoRA** weights into it.
