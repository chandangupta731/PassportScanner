# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os
import json
import re
from paddleocr import PaddleOCR
import google.generativeai as genai
import traceback # Added for better error logging

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# IMPORTANT: Use environment variables for API keys in production!
API_KEY = "AIzaSyA6-YdUVWyqV9uqEFApTS8q0lysHL9qV1s" # Replace or set environment variable

GEMINI_MODEL_NAME = 'gemini-1.5-pro-latest'

# Placeholders and Not Found Markers
PASSPORT_REDACTION_PLACEHOLDER = "[PASSPORT NUMBER REDACTED]"
AADHAAR_REDACTION_PLACEHOLDER = "[AADHAAR NUMBER REDACTED]"
ID_NUMBER_NOT_FOUND = "NA" # Generic "Not Found"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    """Check if uploaded file extension is allowed."""
    print(f"Checking if file {filename} is allowed.")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_paddle(image_path, lang='en'):
    """Perform OCR using PaddleOCR."""
    print(f"Extracting text from image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist.")
        return None
    try:
        # Initialize PaddleOCR - Consider initializing once globally if performance is critical
        # Adjust languages if needed, e.g., 'en+hi' for English and Hindi
        # Using detect=True might help with mixed language detection if needed
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)
        result = ocr.ocr(image_path, cls=True)
        extracted_text = ""
        if result and result[0]:
            # Extract text from PaddleOCR result format
            txts = [line[1][0] for line in result[0] if line and len(line) == 2 and isinstance(line[1], tuple) and len(line[1]) == 2]
            extracted_text = "\n".join(txts)

        if not extracted_text:
             print("Warning: OCR completed but no text was extracted.")
             return None

        print(f"OCR extracted text:\n---\n{extracted_text}\n---")
        return extracted_text.strip()
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        print(traceback.format_exc()) # Print full traceback for OCR errors
        return None

def detect_document_type(text_content):
    """Detect if the text likely comes from a Passport or Aadhaar card."""
    if not text_content:
        print("Cannot detect document type: No text content provided.")
        return "unknown"

    print("Detecting document type...")
    text_lower = text_content.lower()

    # Keywords for Passport
    passport_keywords = ["passport", "republic of india", "issuing authority", "date of expiry", "place of issue", "surname", "given names", "passeport"]

    # Keywords for Aadhaar - Including common Hindi terms
    aadhaar_keywords = ["aadhaar", "uidai", "unique identification authority", "enrollment no", "enrollment number", "government of india", "आधार", "भारत सरकार", "जन्म का वर्ष", "लिंग", "address:", "pincode"] # Year of Birth, Gender

    # Check for Machine Readable Zone (MRZ) - strong indicator for Passport
    mrz_pattern = re.compile(r'P<[A-Z0-9<]{39}\n[A-Z0-9<]{39}')
    if mrz_pattern.search(text_content):
         print("Detected type: Passport (MRZ found)")
         return "passport"

    passport_score = sum(keyword in text_lower for keyword in passport_keywords)
    aadhaar_score = sum(keyword in text_lower for keyword in aadhaar_keywords)

    # Check for 12-digit number format typical of Aadhaar
    if re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text_content):
         aadhaar_score += 2 # Give higher weight if Aadhaar number format is found

    print(f"Detection scores - Passport: {passport_score}, Aadhaar: {aadhaar_score}")

    # Refined detection logic
    if passport_score > aadhaar_score and passport_score >= 2: # Require at least 2 keywords for passport
        print("Detected type: Passport")
        return "passport"
    elif aadhaar_score > passport_score and aadhaar_score >= 2: # Require at least 2 keywords/indicators for aadhaar
        print("Detected type: Aadhaar")
        return "aadhaar"
    elif aadhaar_score > 0 and passport_score == 0: # If only aadhaar keywords found
        print("Detected type: Aadhaar (fallback)")
        return "aadhaar"
    elif passport_score > 0 and aadhaar_score == 0: # If only passport keywords found
         print("Detected type: Passport (fallback)")
         return "passport"
    else:
        print("Could not reliably determine document type based on keywords.")
        return "unknown"


def find_and_redact_passport_no(text_content):
    """Find and redact passport number (e.g., A1234567)."""
    print("Searching for passport number in extracted text.")
    # Common Indian passport format: Letter followed by 7 digits. Case-insensitive search.
    passport_pattern = re.compile(r'\b([A-Z]\d{7})\b', re.IGNORECASE)
    found_passport_no = None
    redacted_text = text_content

    # Sort matches by position to get the most likely one first (often near labels)
    matches = sorted(passport_pattern.finditer(text_content), key=lambda m: m.start())

    if matches:
        # Try finding near labels first
        lines = text_content.split('\n')
        labels = ["passport no", "passport number", "passeport n", "पासपोर्ट नं"]
        found_near_label = False
        for i, line in enumerate(lines):
            if any(label in line.lower() for label in labels):
                search_indices = list(range(max(0, i-1), min(i+3, len(lines)))) # Search current line, 1 before, 2 after
                line_texts = {idx: lines[idx] for idx in search_indices}
                # Check matches found within the proximity of the label
                for match in matches:
                    match_line_index = text_content[:match.start()].count('\n')
                    if match_line_index in line_texts:
                        found_passport_no = match.group(1).upper() # Standardize to uppercase
                        redacted_text = text_content[:match.start()] + PASSPORT_REDACTION_PLACEHOLDER + text_content[match.end():]
                        print(f"Found passport number near label: {found_passport_no}")
                        found_near_label = True
                        break # Stop after first find near a label
            if found_near_label:
                break

        # If not found near label, take the first overall match
        if not found_near_label:
            first_match = matches[0]
            found_passport_no = first_match.group(1).upper()
            redacted_text = text_content[:first_match.start()] + PASSPORT_REDACTION_PLACEHOLDER + text_content[first_match.end():]
            print(f"Found passport number via direct pattern (first match): {found_passport_no}")

    if not found_passport_no:
         print("Passport number not found.")

    return found_passport_no, redacted_text

def find_and_redact_aadhaar_no(text_content):
    """Find and redact Aadhaar number (12 digits, possibly with spaces)."""
    print("Searching for Aadhaar number in extracted text.")
    # Aadhaar pattern: 12 digits, potentially separated by spaces (e.g., xxxx xxxx xxxx or xxxxxxxxxxxx)
    # Ensures it's exactly 12 digits, handling spaces
    aadhaar_pattern = re.compile(r'\b(\d{4}\s?\d{4}\s?\d{4})\b')
    found_aadhaar_no = None
    redacted_text = text_content

    match = aadhaar_pattern.search(text_content)
    if match:
        # Store the number without spaces for consistency, but redact the original format found
        original_format = match.group(1)
        found_aadhaar_no = "".join(original_format.split()) # Remove spaces for storage/potential validation
        # Use re.sub for reliable replacement
        redacted_text = aadhaar_pattern.sub(AADHAAR_REDACTION_PLACEHOLDER, text_content, count=1)
        print(f"Found Aadhaar number: {found_aadhaar_no} (Original format: {original_format})")
    else:
        print("Aadhaar number not found.")
        # Could add label-based search here if needed

    return found_aadhaar_no, redacted_text


def analyze_text_with_gemini(text_content, api_key, model_name, document_type, redaction_placeholder):
    """Use Gemini to extract structured data based on document type."""
    print(f"Analyzing text for document type: {document_type} with Gemini API.")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
         print("Error: Gemini API Key is missing or is a placeholder.")
         return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # --- Construct the prompt dynamically ---
        if document_type == "passport":
            prompt = f"""Analyze the following text extracted from a PASSPORT image using OCR. The OCR text might contain errors or be poorly formatted. The Passport Number has been identified separately and replaced with '{redaction_placeholder}'.

Please identify and extract the REMAINING key information:
- Type (e.g., P)
- Country Code (e.g., IND)
- Surname
- Given Names
- Nationality
- Sex / Gender (e.g., M/F)
- Date of Birth (Format: DD/MM/YYYY)
- Place of Birth
- Place of Issue
- Date of Issue (Format: DD/MM/YYYY)
- Date of Expiry (Format: DD/MM/YYYY)
# The line asking for 'Any other relevant fields' has been removed below to avoid '[object Object]' issues.

Correct obvious OCR errors based on context for these fields. Format the extracted information STRICTLY as a JSON object. Use null or an empty string "" for missing/unclear values. Do NOT include the Passport Number field in your JSON output (it's already redacted).

Potentially Redacted OCR Text:
"{text_content}"

Structured Passport Information (JSON object only, excluding Passport No.):"""

        elif document_type == "aadhaar":
            prompt = f"""Analyze the following text extracted from an AADHAAR CARD image using OCR. The OCR text might contain errors or be poorly formatted. The Aadhaar Number has been identified separately and replaced with '{redaction_placeholder}'.

Please identify and extract the REMAINING key information:
- Name
- Date of Birth OR Year of Birth (Format: DD/MM/YYYY or YYYY)
- Sex / Gender (e.g., Male/Female/Transgender or M/F/T)
- Address (Extract as completely as possible, including Pincode if available)
- Date of Issue (if available, often near the QR code or bottom)

Correct obvious OCR errors based on context for these fields. Format the extracted information STRICTLY as a JSON object. Use null or an empty string "" for missing/unclear values. Do NOT include the Aadhaar Number field in your JSON output (it's already redacted).

Potentially Redacted OCR Text:
"{text_content}"

Structured Aadhaar Information (JSON object only, excluding Aadhaar No.):"""
        else: # Unknown document type
             print("Cannot analyze text: Unknown document type.")
             return None # Or return an error message structure

        print(f"\n--- Gemini Prompt (Document Type: {document_type}) ---\n{prompt}\n---------------------\n")

        # Set safety settings to minimum if needed, otherwise use defaults
        # safety_settings=[
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        # ]
        # response = model.generate_content(prompt, safety_settings=safety_settings)
        response = model.generate_content(prompt) # Use default safety settings

        # Check if the response was blocked or has no text
        if not response.parts:
             print("Gemini response was empty or blocked.")
             # Log the prompt feedback if available
             if response.prompt_feedback:
                 print(f"Prompt Feedback: {response.prompt_feedback}")
             return None

        print(f"Gemini raw response: {response.text}")
        return response.text
    except Exception as e:
        print(f"Error during Gemini analysis: {e}")
        print(traceback.format_exc()) # Print full traceback for Gemini errors
        return None

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Render homepage."""
    print("Rendering homepage.")
    # Pass allowed extensions to template if needed
    # return render_template('index.html', allowed_ext=ALLOWED_EXTENSIONS)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload, OCR, detection, redaction, and analysis."""
    print("\n--- New Upload Request ---")
    if 'file' not in request.files:
        print("Error: No file part in request.")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        print("Error: No file selected.")
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
         print(f"Error: Invalid file format: {file.filename}")
         return jsonify({"error": f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    print(f"Processing file: {file.filename}")
    file_path = None # Initialize file_path
    try:
        # Secure the filename before saving
        from werkzeug.utils import secure_filename
        safe_filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # 1. Perform OCR
        # Consider adding language hints if possible ('hi' or 'en+hi' for Aadhaar)
        full_ocr_text = extract_text_paddle(file_path, lang='en') # Using 'en' as default

        if not full_ocr_text:
            print("Error: OCR failed to extract text.")
            # No need to remove file here, finally block will handle it
            return jsonify({"error": "OCR failed to extract text from the image."}), 500

        # 2. Detect Document Type
        doc_type = detect_document_type(full_ocr_text)

        redacted_text = full_ocr_text
        found_id_number = None
        id_field_name = "Document ID" # Generic default
        redaction_placeholder = "[REDACTED]" # Generic default

        # 3. Redact based on detected type
        if doc_type == "passport":
            found_id_number, redacted_text = find_and_redact_passport_no(full_ocr_text)
            id_field_name = "Passport No."
            redaction_placeholder = PASSPORT_REDACTION_PLACEHOLDER
        elif doc_type == "aadhaar":
            found_id_number, redacted_text = find_and_redact_aadhaar_no(full_ocr_text)
            id_field_name = "Aadhaar No."
            redaction_placeholder = AADHAAR_REDACTION_PLACEHOLDER
        else:
             # If type is unknown, still try a generic extraction? Or return error.
             # For now, return error as detection is expected.
             print("Error: Could not determine document type. Aborting analysis.")
             return jsonify({"error": "Could not determine document type (Passport or Aadhaar). Please upload a clearer image or a supported document."}), 400

        # 4. Analyze with Gemini using the appropriate prompt
        gemini_raw_output = analyze_text_with_gemini(
            redacted_text,
            API_KEY,
            GEMINI_MODEL_NAME,
            doc_type, # Pass the detected type
            redaction_placeholder # Pass the correct placeholder
        )

        # 5. Construct Final Result
        final_result_dict = {
             "Detected Document Type": doc_type.capitalize(),
             id_field_name: found_id_number if found_id_number else ID_NUMBER_NOT_FOUND
             }


        if gemini_raw_output:
            # Attempt to parse the JSON response from Gemini
            cleaned_response = gemini_raw_output.strip()
            # Remove potential Markdown code block fences (```json ... ``` or ``` ... ```)
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[len("```json"):].strip()
            elif cleaned_response.startswith("```"):
                 cleaned_response = cleaned_response[len("```"):].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-len("```")].strip()

            # Sometimes Gemini might add introductory text before the JSON
            json_start_index = cleaned_response.find('{')
            json_end_index = cleaned_response.rfind('}')
            if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                potential_json = cleaned_response[json_start_index:json_end_index+1]
                print(f"Attempting to parse JSON part: {potential_json}")
                try:
                    gemini_data_dict = json.loads(potential_json)
                    if isinstance(gemini_data_dict, dict):
                         # Merge Gemini data, ensuring not to overwrite the ID field we set
                        for key, value in gemini_data_dict.items():
                             # Check case-insensitively if Gemini tried to return the ID number again
                            if key.lower().replace(" ", "").replace(".","").replace("_","") != id_field_name.lower().replace(" ", "").replace(".","").replace("_",""):
                                final_result_dict[key] = value
                            else:
                                print(f"Skipping field '{key}' from Gemini output as it matches the redacted ID field '{id_field_name}'.")
                        final_result_dict["gemini_analysis_status"] = "Success" # Mark as success if parsed
                    else:
                         print(f"Warning: Gemini response parsed but was not a JSON dictionary (Type: {type(gemini_data_dict)}).")
                         final_result_dict["gemini_analysis_raw"] = gemini_raw_output # Include raw if not dict
                         final_result_dict["gemini_analysis_status"] = "Failed: Response not a JSON dictionary"

                except json.JSONDecodeError as e:
                    print(f"Error parsing Gemini JSON response: {e}")
                    print(f"Problematic response text slice: '{potential_json}'")
                    final_result_dict["gemini_analysis_raw"] = gemini_raw_output # Include raw response on error
                    final_result_dict["gemini_analysis_status"] = "Failed: Could not parse JSON response"
                except Exception as e:
                     print(f"Unexpected error processing Gemini response: {e}")
                     final_result_dict["gemini_analysis_raw"] = gemini_raw_output
                     final_result_dict["gemini_analysis_status"] = f"Failed: {e}"
            else:
                 print("Error: Could not find valid JSON structure '{...}' in Gemini response.")
                 final_result_dict["gemini_analysis_raw"] = gemini_raw_output
                 final_result_dict["gemini_analysis_status"] = "Failed: No valid JSON object found"
        else:
            final_result_dict["gemini_analysis_status"] = f"Failed: No response or error during Gemini call for {doc_type}"

        print(f"Final extracted data: {json.dumps(final_result_dict, indent=2)}")

        # Return the result
        return jsonify(final_result_dict)

    except Exception as e:
        print(f"An unexpected error occurred during the upload process: {e}")
        print(traceback.format_exc()) # Log the full traceback for debugging server-side
        return jsonify({"error": "An internal server error occurred. Please check server logs."}), 500

    finally:
        # Clean up the uploaded file after processing, regardless of success or failure
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
            except OSError as e:
                print(f"Error removing file {file_path}: {e}")


# --- Start Server ---
if __name__ == '__main__':
    print("Starting Flask server...")
    # Set debug=False for production environments
    # Use host='0.0.0.0' to make it accessible on your network (be careful with security)
    # Consider using a production WSGI server like Gunicorn or Waitress instead of app.run for deployment
    app.run(debug=True, host='0.0.0.0', port=5000) # Default port is 5000
