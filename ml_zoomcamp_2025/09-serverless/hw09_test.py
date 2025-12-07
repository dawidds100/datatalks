#!/usr/bin/env python3
"""
Quick test script for Homework 9 questions 1-4
Run this to verify your setup and get answers
"""

import sys
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

# Try to import onnxruntime
try:
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except ImportError:
    print("⚠️  onnxruntime not found. Install with: pip install onnxruntime")
    ONNX_AVAILABLE = False


def download_image(url):
    """Download image from URL"""
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    """Prepare image by converting to RGB and resizing"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(img_array):
    """Apply ImageNet normalization"""
    x = img_array / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std
    return x


# ============================================================================
# QUESTION 1: Inspect Model
# ============================================================================

def answer_q1(model_path):
    """Q1: What's the name of the output node?"""
    if not ONNX_AVAILABLE:
        return None
    
    try:
        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print("\n" + "="*60)
        print("QUESTION 1: Output Node Name")
        print("="*60)
        print(f"Input node: {input_name}")
        print(f"Output node: {output_name}")
        print(f"\n✓ Q1 ANSWER: {output_name}")
        print("  (likely 'sigmoid' for binary classification)")
        
        return output_name
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# ============================================================================
# QUESTION 2: Target Image Size
# ============================================================================

def answer_q2():
    """Q2: What should be the target size for the image?"""
    print("\n" + "="*60)
    print("QUESTION 2: Target Image Size")
    print("="*60)
    print("From homework 8 (deep learning), the model was trained with")
    print("128x128 pixel images.")
    print("\n✓ Q2 ANSWER: 128x128")
    print("  Options were: 64x64, 128x128, 200x200, 256x256")
    return 128


# ============================================================================
# QUESTION 3: Preprocessed Pixel Value
# ============================================================================

def answer_q3(image_url):
    """Q3: First pixel R channel value after preprocessing"""
    try:
        print("\n" + "="*60)
        print("QUESTION 3: First Pixel R Value After Preprocessing")
        print("="*60)
        
        # Download and prepare
        img = download_image(image_url)
        img_prepared = prepare_image(img, target_size=(128, 128))
        
        # Convert to array
        img_array = np.array(img_prepared)
        print(f"Original image shape: {img_array.shape}")
        print(f"First pixel RGB: {img_array[0, 0, :]}")
        
        # Preprocess
        img_normalized = preprocess_image(img_array)
        
        # Get first pixel R channel
        r_value = img_normalized[0, 0, 0]
        
        print(f"\nPreprocessing steps:")
        print(f"  1. Original R value: {img_array[0, 0, 0]}")
        print(f"  2. Normalized [0,1]: {img_array[0, 0, 0] / 255.0:.4f}")
        print(f"  3. ImageNet norm: {r_value:.4f}")
        print(f"\n✓ Q3 ANSWER: {r_value:.3f}")
        print("  Options were: -10.73, -1.073, 1.073, 10.73")
        
        return r_value
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# ============================================================================
# QUESTION 4: Model Output
# ============================================================================

def answer_q4(model_path, image_url):
    """Q4: What's the output of the model?"""
    if not ONNX_AVAILABLE:
        return None
    
    try:
        print("\n" + "="*60)
        print("QUESTION 4: Model Output")
        print("="*60)
        
        # Load model
        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"Model loaded: {output_name}")
        
        # Download and preprocess image
        img = download_image(image_url)
        img_prepared = prepare_image(img, target_size=(128, 128))
        img_array = np.array(img_prepared).astype('float32')
        img_normalized = preprocess_image(img_array)
        
        # Prepare for model
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        
        print(f"Input shape: {img_input.shape}")
        print(f"Input range: [{img_input.min():.3f}, {img_input.max():.3f}]")
        
        # Run inference
        output = session.run([output_name], {input_name: img_input})[0]
        
        # Extract prediction
        if output.shape[0] == 1 and len(output.shape) > 1:
            prediction = float(output[0][0])
        else:
            prediction = float(output[0])
        
        print(f"Output shape: {output.shape}")
        print(f"\n✓ Q4 ANSWER: {prediction:.2f}")
        print("  Options were: 0.09, 0.49, 0.69, 0.89")
        
        return prediction
    except Exception as e:
        print(f"Error running inference: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Configuration
    model_path = "hair_classifier_v1.onnx"
    image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    
    print("\n" + "="*60)
    print("HOMEWORK 9 - Quick Test Script")
    print("Hair Classifier Model Deployment")
    print("="*60)
    
    # Q1: Model inspection
    if ONNX_AVAILABLE:
        answer_q1(model_path)
    else:
        print("\nQ1: Cannot test without onnxruntime")
    
    # Q2: Target size
    answer_q2()
    
    # Q3: Pixel value
    try:
        answer_q3(image_url)
    except Exception as e:
        print(f"\nQ3: Error - {e}")
        print("(Check internet connection for image download)")
    
    # Q4: Model output
    if ONNX_AVAILABLE:
        try:
            answer_q4(model_path, image_url)
        except Exception as e:
            print(f"\nQ4: Error - {e}")
    else:
        print("\nQ4: Cannot test without onnxruntime")
    
    print("\n" + "="*60)
    print("SUMMARY OF EXPECTED ANSWERS")
    print("="*60)
    print("Q1: sigmoid")
    print("Q2: 128x128")
    print("Q3: -1.073")
    print("Q4: 0.69")
    print("Q5: 608 Mb")
    print("Q6: 0.10")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
