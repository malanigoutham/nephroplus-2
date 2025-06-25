#!/usr/bin/env python3
"""
Medical Report OCR Parser with Ollama DeepSeek
Extract text from medical reports and convert to JSON using local Ollama
"""

import os
import cv2
import numpy as np
import json
import requests
from pathlib import Path
import glob
from datetime import datetime
import pytesseract
from tqdm import tqdm
import traceback
import logging # Import logging module
import re # Import re for regex

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# If DEBUG_MODE is True, set level to DEBUG, otherwise INFO
# This allows granular control without changing every print statement
if os.getenv("DEBUG_MODE", "False").lower() == "true" or True: # Set to True for always-on debug in this example
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("Debug mode is ENABLED via configuration.")
else:
    logging.info("Debug mode is DISABLED.")

# ====== CONFIGURATION VARIABLES ======
INPUT_FOLDER = "./input_images"  # Folder containing medical report images
OUTPUT_FOLDER = "./output"       # Where to save results
MAX_IMAGES = 0                   # 0 = process all images, else limit to this number
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2:3b"     # Model to use
# Path to Tesseract executable (e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe" on Windows)
TESSERACT_CMD_PATH = "" # Leave empty if Tesseract is in PATH, otherwise specify.
DEBUG_MODE = True                # Enable detailed debugging output (also controlled by logging level)
# =====================================

# Set Tesseract command path if specified
if TESSERACT_CMD_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
    logging.info(f"Using Tesseract executable from: {TESSERACT_CMD_PATH}")
else:
    logging.info("Tesseract executable path not explicitly set. Assuming it's in system PATH.")


class MedicalReportOCR:
    def __init__(self, ollama_url: str = OLLAMA_BASE_URL, model_name: str = OLLAMA_MODEL):
        """Initialize OCR processor and Ollama client"""
        self.ollama_url = ollama_url
        self.model_name = model_name

        # Test Ollama connection
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5) # Added timeout for connection test
            if response.status_code == 200:
                logging.info(f"✅ Connected to Ollama at {ollama_url}")

                # Check if model is available
                models = [model['name'] for model in response.json().get('models', [])]
                if model_name in models:
                    logging.info(f"✅ Model {model_name} is available")
                else:
                    logging.warning(f"⚠️  Model {model_name} not found. Available models: {models}")
                    logging.warning(f"    Run: ollama pull {model_name}")
            else:
                logging.error(f"❌ Failed to connect to Ollama at {ollama_url}. Status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logging.critical(f"❌ Ollama connection error: Could not connect to {ollama_url}")
            logging.critical("    Make sure Ollama is running: 'ollama serve'")
            raise # Re-raise to stop execution if Ollama isn't available
        except Exception as e:
            logging.critical(f"❌ Ollama connection error: {e}")
            logging.critical("    Make sure Ollama is running: 'ollama serve'")
            raise # Re-raise to stop execution

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Read image
            # Using str(image_path) for compatibility with cv2.imread which might not like Path objects directly
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}. Check path and file integrity.")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply denoising
            # h, templateWindowSize, searchWindowSize, filterStrength
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21) # Default parameters, fine-tuned can be better

            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(sharpened)

            logging.debug(f"    🖼️  Image preprocessing completed successfully for {image_path.name}")

            return enhanced
        except Exception as e:
            logging.error(f"    ❌ Image preprocessing error for {image_path}: {e}")
            raise # Re-raise the exception to be caught by process_image

    def extract_text_tesseract(self, image_path: Path) -> tuple[str, list[dict]]:
        """Extract text using Tesseract"""
        try:
            logging.debug(f"    🔍 Starting Tesseract OCR extraction for {image_path.name}...")

            processed_img = self.preprocess_image(image_path)

            # Get text with confidence
            # You can add config for Tesseract like lang='eng', --psm 3 (page segmentation mode)
            # Example: config = '--psm 3 --oem 3'
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)

            extracted_texts = []
            for i in range(len(data['text'])):
                # Filter out empty strings and low confidence text
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                if text and confidence > 30: # Confidence threshold of 30
                    extracted_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'box': { # Add bounding box info for potential future use
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })

            # Combine all text, maintaining original order
            full_text = ' '.join([item['text'] for item in extracted_texts])

            logging.debug(f"    📝 OCR extracted {len(extracted_texts)} text blocks for {image_path.name}")
            logging.debug(f"    📏 Full text length: {len(full_text)} characters")
            if full_text:
                preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
                logging.debug(f"    👀 Text preview: {preview}")
            else:
                logging.warning(f"    ⚠️  No significant text extracted for {image_path.name}")

            return full_text, extracted_texts

        except pytesseract.TesseractNotFoundError:
            logging.critical("❌ Tesseract executable not found. Please install Tesseract OCR and configure its path.")
            logging.critical("   See https://tesseract-ocr.github.io/tessdoc/Installation.html")
            return "", []
        except Exception as e:
            logging.error(f"    ❌ OCR extraction failed for {image_path.name}: {e}")
            if DEBUG_MODE:
                logging.debug(f"    🔧 Full error traceback for {image_path.name}:", exc_info=True) # Use exc_info=True
            return "", []

    def generate_json_with_ollama(self, extracted_text: str, image_filename: str) -> dict:
        """Use Ollama to convert extracted text to structured JSON"""

        logging.debug(f"    🤖 Starting Ollama processing for {image_filename}...")
        logging.debug(f"    📊 Input text length: {len(extracted_text)} characters")

        # Truncate text if too long to avoid token limits
        # The specific max_tokens for llama3.2:3b might be 4096 or 8192 depending on context window
        # Using a conservative limit to ensure it fits and leaves room for output
        max_text_length = 6000 # Adjusted slightly more conservative
        if len(extracted_text) > max_text_length:
            extracted_text = extracted_text[:max_text_length] + "\n[TEXT TRUNCATED DUE TO LENGTH]"
            logging.warning(f"    ✂️  Text truncated to {max_text_length} characters for {image_filename}")

        # --- Few-shot prompting example (conceptual) ---
        # If you had a set of good input-output pairs, you'd add them here.
        # This is commented out to avoid changing the LLM's behavior and thus the output in this specific request.
        # Adding actual examples would make the model more robust to varied inputs.
        few_shot_examples = """
        Example 1:
        Extracted Text: \"Patient: John Doe, Age: 45, Test: Blood Sugar, Result: 120 mg/dL, Ref: 70-100 mg/dL\"
        JSON Output:
        {
          "patient_info": {"name": "John Doe", "age": "45"},
          "test_results": [{"test_name": "Blood Sugar", "result_value": "120", "unit": "mg/dL", "reference_range": "70-100", "status": "Abnormal"}]
        }

        Example 2:
        Extracted Text: \"Hospital: City Hospital, Test: CBC, WBC: 8.5 (4.0-10.0), RBC: 4.8 (4.5-5.5)\"
        JSON Output:
        {
          "hospital_info": {"hospital_name": "City Hospital"},
          "test_results": [
            {"test_name": "WBC", "result_value": "8.5", "unit": "K/uL", "reference_range": "4.0-10.0", "status": "Normal"},
            {"test_name": "RBC", "result_value": "4.8", "unit": "M/uL", "reference_range": "4.5-5.5", "status": "Normal"}
          ]
        }
        """ # End of conceptual few-shot examples

        prompt = f"""You are an expert medical report parser. I have extracted text from a medical report image using OCR. Please analyze this text and convert it into a well-structured JSON format.

{extracted_text}

Please create a comprehensive JSON structure that includes:

1.  **hospital_info**: Hospital name, address, phone, website, etc.
2.  **patient_info**: Patient details like name, age, gender, ID, etc.
3.  **doctor_info**: Referring doctor, consultant, pathologist, etc.
4.  **report_info**: Report type, dates (collection, report), sample info, etc.
5.  **test_results**: Array of all tests with:
    -   test_name
    -   result_value
    -   reference_range
    -   unit
    -   status (normal/abnormal if determinable)
6.  **additional_info**: Any notes, interpretations, or other relevant information

Guidelines:
-   Extract ALL available information from the text.
-   If a field is not found, include it with `null` value.
-   For test results, try to identify patterns like "TEST_NAME VALUE RANGE UNIT".
-   Preserve exact values and ranges as found in the text.
-   Clean up obvious OCR errors where possible.
-   Make the JSON as comprehensive and accurate as possible.
-   Ensure all dates are parsed into `YYYY-MM-DD` format and times are `HH:MM:SS` or `HH:MM` where possible.
-   Infer 'status' (Normal/Abnormal) by comparing 'result_value' with 'reference_range'.

Return ONLY the JSON structure, no additional text or explanations.
"""
        # If you were to enable few-shot, you'd insert it like this:
        # prompt = f"..."
        # if few_shot_examples:
        #     prompt = few_shot_examples + "\n" + prompt


        try:
            logging.debug(f"    📡 Sending request to Ollama API for {image_filename}...")
            logging.debug(f"    🔗 URL: {self.ollama_url}/api/generate")
            logging.debug(f"    🏷️  Model: {self.model_name}")

            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 4096, # Explicitly set context window if known for model
                    "max_tokens": 2048 # Ensure this is sufficient for the expected JSON output size
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_data,
                timeout=120
            )

            logging.debug(f"    📥 Ollama response status for {image_filename}: {response.status_code}")
            logging.debug(f"    📏 Response content length for {image_filename}: {len(response.text)} characters")

            if response.status_code != 200:
                error_msg = f'Ollama API error: HTTP {response.status_code}'
                logging.error(f"    ❌ {error_msg} for {image_filename}")
                if DEBUG_MODE:
                    logging.debug(f"    📄 Response headers: {dict(response.headers)}")
                    logging.debug(f"    📄 Response content: {response.text[:500]}...")

                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': response.text,
                    'status_code': response.status_code
                }

            # Parse the response
            try:
                result = response.json()
                logging.debug(f"    ✅ Successfully parsed Ollama response JSON for {image_filename}")
                logging.debug(f"    🔑 Response keys: {list(result.keys())}")
            except json.JSONDecodeError as e:
                error_msg = f'Failed to parse Ollama response as JSON: {str(e)}'
                logging.error(f"    ❌ {error_msg} for {image_filename}")
                if DEBUG_MODE:
                    logging.debug(f"    📄 Raw response: {response.text[:1000]}...")

                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': response.text
                }

            json_text = result.get('response', '').strip()

            logging.debug(f"    📏 JSON text length from Ollama for {image_filename}: {len(json_text)} characters")
            if not json_text:
                logging.warning(f"    ⚠️  WARNING: Empty response from Ollama for {image_filename}!")
                logging.debug(f"    🔍 Full Ollama result for {image_filename}: {result}")
                return {
                    'success': False,
                    'error': 'Empty response from Ollama',
                    'raw_response': json_text,
                    'full_ollama_result': result
                }
            else:
                preview = json_text[:300] + "..." if len(json_text) > 300 else json_text
                logging.debug(f"    👀 JSON preview for {image_filename}: {preview}")

            # Clean up the response to get just the JSON
            original_json_text = json_text

            # Regex to find JSON block, more robust
            json_match = re.search(r'(\{.*\}|\s*\[.*\]\s*)', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0).strip()
                if DEBUG_MODE and json_text != original_json_text:
                    logging.debug(f"    🧹 Extracted JSON from response using regex for {image_filename}")
            else:
                # Fallback to old cleaning if regex fails or for simple cases
                if json_text.startswith('```json'):
                    json_text = json_text[7:]
                    if DEBUG_MODE: logging.debug(f"    🧹 Removed ```json prefix for {image_filename}")
                elif json_text.startswith('```'):
                    json_text = json_text[3:]
                    if DEBUG_MODE: logging.debug(f"    🧹 Removed ``` prefix for {image_filename}")

                if json_text.endswith('```'):
                    json_text = json_text[:-3]
                    if DEBUG_MODE: logging.debug(f"    🧹 Removed ``` suffix for {image_filename}")

                json_text = json_text.strip()
                if DEBUG_MODE and json_text != original_json_text and not json_match:
                     logging.debug(f"    🧹 Cleaned JSON text length: {len(json_text)} characters for {image_filename}")


            # Parse and return JSON
            try:
                parsed_json = json.loads(json_text)
                logging.debug(f"    ✅ Successfully parsed JSON structure for {image_filename}")
                logging.debug(f"    🔑 JSON keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'} for {image_filename}")
            except json.JSONDecodeError as e:
                error_msg = f'JSON parsing error: {str(e)}'
                logging.error(f"    ❌ {error_msg} for {image_filename}")
                if DEBUG_MODE:
                    logging.debug(f"    📄 JSON text that failed to parse for {image_filename}: {json_text[:500]}...")
                    logging.debug(f"    🔧 JSON error position for {image_filename}: line {e.lineno}, column {e.colno}")

                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': json_text,
                    'original_response': original_json_text,
                    'json_error_details': {
                        'line': e.lineno,
                        'column': e.colno,
                        'message': e.msg
                    }
                }

            # Add metadata
            parsed_json['_metadata'] = {
                'source_image': image_filename,
                'extraction_method': 'tesseract_ollama_deepseek',
                'processing_timestamp': datetime.now().isoformat(),
                'model_used': self.model_name
            }

            logging.debug(f"    🎉 JSON processing completed successfully for {image_filename}!")

            return {
                'success': True,
                'json_data': parsed_json,
                'raw_response': json_text
            }

        except requests.exceptions.RequestException as e:
            error_msg = f'Ollama request error: {str(e)}'
            logging.error(f"    ❌ {error_msg} for {image_filename}")
            if DEBUG_MODE:
                logging.debug(f"    🔧 Full error traceback for {image_filename}:", exc_info=True)

            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
        except Exception as e:
            error_msg = f'Unexpected error in Ollama processing for {image_filename}: {str(e)}'
            logging.error(f"    ❌ {error_msg}")
            if DEBUG_MODE:
                logging.debug(f"    🔧 Full error traceback for {image_filename}:", exc_info=True)

            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }

    def process_image(self, image_path: Path) -> dict:
        """Process a single medical report image"""
        image_filename = image_path.name
        logging.info(f"📄 Processing: {image_filename}")

        try:
            # Extract text using Tesseract
            extracted_text, extraction_details = self.extract_text_tesseract(image_path)

            if not extracted_text.strip():
                error_msg = 'No text extracted from image'
                logging.warning(f"    ❌ {error_msg} for {image_filename}")

                return {
                    'success': False,
                    'image_path': str(image_path), # Convert Path to string for dict
                    'image_filename': image_filename,
                    'error': error_msg
                }

            logging.info(f"    📝 Extracted {len(extraction_details)} text blocks from {image_filename}")

            # Generate structured JSON using Ollama
            ollama_result = self.generate_json_with_ollama(extracted_text, image_filename)

            if ollama_result['success']:
                logging.debug(f"    ✅ Successfully generated JSON structure for {image_filename}")

                return {
                    'success': True,
                    'image_path': str(image_path),
                    'image_filename': image_filename,
                    'extracted_text': extracted_text,
                    'extraction_details': extraction_details,
                    'structured_json': ollama_result['json_data'],
                    'ollama_raw_response': ollama_result['raw_response']
                }
            else:
                logging.error(f"    ❌ Failed to generate JSON for {image_filename}: {ollama_result['error']}")

                return {
                    'success': False,
                    'image_path': str(image_path),
                    'image_filename': image_filename,
                    'error': ollama_result['error'],
                    'extracted_text': extracted_text,
                    'ollama_raw_response': ollama_result.get('raw_response'),
                    'ollama_error_details': ollama_result
                }

        except Exception as e:
            error_msg = f'Overall processing error for {image_filename}: {str(e)}'
            logging.error(f"    ❌ {error_msg}")
            if DEBUG_MODE:
                logging.debug(f"    🔧 Full error traceback for {image_filename}:", exc_info=True)

            return {
                'success': False,
                'image_path': str(image_path),
                'image_filename': image_filename,
                'error': error_msg
            }

def get_image_files(folder_path: Path) -> list[Path]:
    """Get all image files from the specified folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []

    for ext in image_extensions:
        # Use Path.glob for more robust path handling
        image_files.extend(folder_path.glob(ext))
        image_files.extend(folder_path.glob(ext.upper()))

    return sorted(image_files)

def save_raw_text(extracted_text: str, image_filename: str, text_output_dir: Path) -> tuple[bool, str]:
    """Save raw extracted text to a .txt file"""
    base_name = Path(image_filename).stem # Use Pathlib for stem
    txt_filename = f"{base_name}.txt"
    txt_filepath = text_output_dir / txt_filename # Use Pathlib for joining

    try:
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        return True, txt_filename
    except Exception as e:
        logging.error(f"    ❌ Error saving raw text to {txt_filepath}: {e}")
        return False, str(e)

def main():
    """Main processing function"""

    # Convert string paths to Path objects for better handling
    input_folder_path = Path(INPUT_FOLDER)
    output_folder_path = Path(OUTPUT_FOLDER)

    # Validate input folder
    if not input_folder_path.exists():
        logging.critical(f"❌ Input folder not found: {input_folder_path}")
        logging.critical(f"    Create the folder and add medical report images")
        return

    # Create output directories
    json_output_dir = output_folder_path / "json"
    text_output_dir = output_folder_path / "text"
    debug_output_dir = output_folder_path / "debug"

    json_output_dir.mkdir(parents=True, exist_ok=True)
    text_output_dir.mkdir(parents=True, exist_ok=True)
    if DEBUG_MODE:
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"📂 Debug output directory: {debug_output_dir}")

    logging.info(f"📂 JSON output directory: {json_output_dir}")
    logging.info(f"📂 Text output directory: {text_output_dir}")

    # Get image files
    image_files = get_image_files(input_folder_path)

    if not image_files:
        logging.critical(f"❌ No image files found in: {input_folder_path}")
        logging.critical("    Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        return

    # Limit number of images if specified
    if MAX_IMAGES > 0:
        image_files = image_files[:MAX_IMAGES]
        logging.info(f"📊 Processing limited to {MAX_IMAGES} images")

    logging.info(f"📊 Found {len(image_files)} image(s) to process")

    # Initialize OCR processor
    try:
        ocr_processor = MedicalReportOCR()
    except Exception as e:
        logging.critical(f"❌ Failed to initialize OCR processor: {str(e)}")
        if DEBUG_MODE:
            logging.debug("🔧 Full error traceback:", exc_info=True)
        return

    # Process each image
    successful_count = 0
    failed_count = 0
    text_saved_count = 0

    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        logging.info(f"\n{'='*50}\nProcessing {i+1}/{len(image_files)}\n{'='*50}")

        # Process image
        result = ocr_processor.process_image(image_path)

        # Generate output filenames
        base_name = image_path.stem # Path object .stem
        json_filename = f"{base_name}_extracted.json"
        json_filepath = json_output_dir / json_filename # Pathlib for joining

        # Save raw extracted text regardless of JSON processing success
        if 'extracted_text' in result and result['extracted_text'].strip():
            text_success, text_result_info = save_raw_text(
                result['extracted_text'],
                result['image_filename'], # Pass original filename string
                text_output_dir
            )

            if text_success:
                logging.info(f"    📝 Raw text saved: {text_result_info}")
                text_saved_count += 1
            else:
                logging.error(f"    ❌ Failed to save raw text: {text_result_info}")

        if result['success']:
            # Save structured JSON
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(result['structured_json'], f, indent=2, ensure_ascii=False)

                logging.info(f"    ✅ Successfully saved: {json_filename}")

                # Display summary of extracted data
                json_data = result['structured_json']
                hospital_name = json_data.get('hospital_info', {}).get('hospital_name', 'N/A')
                patient_name = json_data.get('patient_info', {}).get('name', 'N/A')
                test_results = json_data.get('test_results', [])
                test_count = len(test_results) if isinstance(test_results, list) else 0

                logging.info(f"    📋 Hospital: {hospital_name}")
                logging.info(f"    👤 Patient: {patient_name}")
                logging.info(f"    🧪 Tests found: {test_count}")

                successful_count += 1

            except Exception as e:
                logging.error(f"    ❌ Failed to save JSON for {json_filename}: {str(e)}")
                if DEBUG_MODE:
                    logging.debug("🔧 Full error traceback:", exc_info=True)
                failed_count += 1
        else:
            # Save detailed error information
            error_data = {
                'error': result['error'],
                'image_path': result['image_path'],
                'extracted_text': result.get('extracted_text', ''),
                'timestamp': datetime.now().isoformat(),
                'debug_info': result.get('ollama_error_details', {})
            }

            error_filename = f"{base_name}_error.json"
            error_filepath = json_output_dir / error_filename

            try:
                with open(error_filepath, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, indent=2, ensure_ascii=False)
                logging.info(f"    💾 Error details saved to: {error_filename}")
            except Exception as e:
                logging.error(f"    ❌ Failed to save error details for {error_filename}: {str(e)}")

            # Save debug information if enabled
            if DEBUG_MODE and 'ollama_error_details' in result:
                debug_filename = f"{base_name}_debug.json"
                debug_filepath = debug_output_dir / debug_filename

                try:
                    with open(debug_filepath, 'w', encoding='utf-8') as f:
                        json.dump(result['ollama_error_details'], f, indent=2, ensure_ascii=False)
                    logging.info(f"    🔧 Debug info saved to: {debug_filename}")
                except Exception as e:
                    logging.error(f"    ❌ Failed to save debug info for {debug_filename}: {str(e)}")

            logging.error(f"    ❌ Failed to process {image_filename}: {result['error']}")
            failed_count += 1

    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info(f"🎉 PROCESSING COMPLETE!")
    logging.info(f"{'='*60}")
    logging.info(f"✅ Successfully processed: {successful_count} images")
    logging.info(f"❌ Failed to process: {failed_count} images")
    logging.info(f"📝 Raw text files saved: {text_saved_count}")
    logging.info(f"📂 Output folder: {OUTPUT_FOLDER}")

    if DEBUG_MODE:
        logging.info(f"🔧 Debug mode was enabled - check debug folder for detailed logs")

    return successful_count, failed_count, text_saved_count

if __name__ == "__main__":
    logging.info("🚀 Starting Medical Report OCR Processing with Ollama DeepSeek")
    logging.info(f"📂 Input folder: {INPUT_FOLDER}")
    logging.info(f"🔢 Max images: {'All' if MAX_IMAGES == 0 else MAX_IMAGES}")
    logging.info(f"🤖 Using: Tesseract OCR + Ollama {OLLAMA_MODEL}")
    logging.info(f"🔧 Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")

    try:
        successful, failed, text_saved = main()
    except KeyboardInterrupt:
        logging.info("\n⏸️  Processing interrupted by user")
    except Exception as e:
        logging.critical(f"\n❌ Unexpected error: {e}")
        if DEBUG_MODE:
            logging.debug("🔧 Full error traceback:", exc_info=True)
