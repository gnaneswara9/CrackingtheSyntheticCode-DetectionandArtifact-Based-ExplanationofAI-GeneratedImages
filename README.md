# **Cracking the Synthetic Code: Detection and Artifact-Based Explanation of AI-Generated Images**

## **Overview**  
This project focuses on detecting AI-generated images and identifying their distinguishing artifacts. By leveraging deep learning and signal processing techniques, we develop a robust pipeline that ensures both high detection accuracy and interpretability.  

## **Motivation**  
With the rapid advancement of AI-generated content, distinguishing real images from synthetic ones has become increasingly challenging. Many detection models function as "black boxes," making it difficult to trust their decisions. This project bridges the gap between accuracy and interpretability by integrating explainable AI techniques.  

## **Key Features**  
- **AI-Generated Image Detection:** Uses noise spectrum analysis and deep learning models.  
- **Artifact-Based Explanations:** Integrates a Vision-Language Model (MiniCPM) to provide human-understandable explanations.  
- **Fourier Spectrum Analysis:** Leverages frequency domain features to enhance detection accuracy.  
- **Denoising Autoencoder (DAE):** Highlights generative artifacts through noise reconstruction.  
- **CNN Classifier:** Trained on spectral features to distinguish between real and AI-generated images.  

## **Methodology**  
1. **Data Preprocessing:**  
   - Uses the CIFAKE dataset (32×32 RGB images).  
   - Upscales images using **SRGAN** to retain generative noise patterns.  
   - Denoises images with a **Denoising Autoencoder (DAE)** trained on real images.  

2. **Noise Spectrum Analysis:**  
   - Computes noise by subtracting DAE-reconstructed images from original images.  
   - Transforms noise into the **Fourier domain** to extract high-frequency patterns indicative of AI generation.  

3. **Classification:**  
   - Trains a **CNN classifier** on Fourier-transformed noise features.  
   - Uses **binary classification** to distinguish between real and fake images.  

4. **Artifact Detection & Explanation:**  
   - Employs the **MiniCPM Vision-Language Model** to explain detection decisions.  
   - Identifies key artifacts (e.g., inconsistent textures, unnatural shading).  

## **Results**  
| Metric  | Training Set | Validation Set |
|---------|-------------|---------------|
| Accuracy | 91.74% | 88.05% |
| F1-Score | 90.74% | 87.68% |
| Recall | 91.86% | 87.05% |

## **Challenges & Future Work**  
- **Minimal noise artifacts**: Advanced models generate near-perfect images, requiring improved detection techniques.  
- **Explainability**: Enhancing the interpretability of results with better artifact localization.  
- **Dataset Expansion**: Increasing diversity to improve model generalization.  
- **Real-Time Applications**: Optimizing inference time for deployment in media verification and fraud detection.  

## **Applications**  
- **Fake news detection**  
- **E-commerce product authenticity verification**  
- **Digital identity verification**  
- **AI-generated content monitoring**  

## **Installation & Usage**  
### **Requirements**  
- Python 3.8+  
- PyTorch  
- OpenCV  
- TensorFlow/Keras  
- Transformers  
- NumPy & SciPy  

### **Installation**  
```bash
git clone https://github.com/yourusername/ai-image-detection.git
cd ai-image-detection
pip install -r requirements.txt
