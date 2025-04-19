from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import easyocr
import os
from paddleocr import PaddleOCR

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Add use_gpu=True if CUDA is available

# Function to format date
def format_date(date_str):
    try:
        year = int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        year += 2000 if year < 50 else 1900
        return f"{day:02d}/{month:02d}/{year}"
    except:
        return ""

# Function to parse MRZ lines into structured data
def parse_mrz_to_dict(line1, line2):
    print(line1)
    print(line2)

    if len(line1) != 44 or len(line2) != 44:
        return {"error": "Invalid MRZ line lengths."}

    # Names
    names = line1[5:].split("<<")
    last_name = names[0].replace('<', ' ').strip()
    first_name = names[1].replace('<', ' ').strip() if len(names) > 1 else ""

    return {
        "Document Type": line1[0:2].replace('<', '').strip(),
        "Issuing Country": line1[2:5],
        "Last Name": last_name,
        "First Name": first_name,
        "Passport Number": line2[0:9],
        "Nationality": line2[10:13],
        "Date of Birth": format_date(line2[13:19]),
        "Sex": line2[20],
        "Date of Expiry": format_date(line2[21:27]),
    }

# Function to extract Date of Issue based on keywords
def extract_doi_from_keywords(lines, keywords):
    doi = ""
    for i, line in enumerate(lines):
        stripped_line = line.strip().lower()
        if any(keyword in stripped_line for keyword in keywords):
            if i + 1 < len(lines):
                doi = lines[i + 1].strip()
    return doi

# Function to extract Place of Issue based on keywords
def extract_poi_from_keywords(lines, keywords):
    poi = ""
    for i, line in enumerate(lines):
        stripped_line = line.strip().lower()
        if any(keyword in stripped_line for keyword in keywords):
            if i + 1 < len(lines):
                poi = lines[i + 1].strip()
    return poi

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Handling file upload
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Debug print to confirm the file save location
    print("Saved file to:", path)

    # Read the image with OpenCV
    image = cv2.imread(path)
    if image is None:
        return jsonify({"error": "Failed to load image with OpenCV"}), 400

    # Crop the MRZ area (bottom 30% of the image)
    height = image.shape[0]
    mrz_area = image[int(height * 0.7):, :]
    temp_crop = os.path.join(UPLOAD_FOLDER, "cropped.jpg")
    cv2.imwrite(temp_crop, mrz_area)

    # Use EasyOCR to read the MRZ text
    reader = easyocr.Reader(['en'], gpu=False)
    mrz_results = reader.readtext(temp_crop, detail=0)
    print(mrz_results)
    mrz_lines = [line.replace(" ", "").replace('\n', '') for line in mrz_results if line.count('<') > 5 and len(line) >= 30]

    # Check if MRZ lines are detected
    if len(mrz_lines) >= 2:
        print(mrz_lines)
        mrz_data = parse_mrz_to_dict(mrz_lines[0], mrz_lines[1])
    else:
        mrz_data = {"error": "MRZ not detected"}

    # OCR with PaddleOCR for Date of Issue and Place of Issue
    results = ocr.ocr(temp_crop, cls=True)
    lines = [line[1][0] for block in results for line in block]

    # Define keywords for Date of Issue (DOI)
    keywords_doi = ["date", "Date", "issue", "Issue", "date issue", "Date Issue"]

    # Extract Date of Issue based on keywords
    extracted_doi = extract_doi_from_keywords(lines, keywords_doi)

    # Define keywords for Place of Issue (POI)
    keywords_poi = ["place", "Place", "issued", "Issued", "place of issue", "Place of Issue"]

    # Extract Place of Issue based on keywords
    extracted_poi = extract_poi_from_keywords(lines, keywords_poi)

    # Print the extracted DOI and POI
    print("\n--- Extracted Date of Issue ---")
    print(f"Date of Issue: {extracted_doi}")

    print("\n--- Extracted Place of Issue ---")
    print(f"Place of Issue: {extracted_poi}")

    # Combine MRZ data, DOI, and POI into one response
    response_data = {**mrz_data, "Date of Issue": extracted_doi, "Place of Issue": extracted_poi}

    return jsonify(response_data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
